"""
방안 3: Ligand-Only Baseline 재정의
=====================================
protein 엣지를 baseline에서 제외하고 interaction vs ligand만 비교.

기존: baseline = interaction / (ligand + interaction + protein)
수정: baseline = interaction / (ligand + interaction)  ← protein 제외

protein = 0.0인 구조적 문제를 제거하고 순수하게
interaction 엣지가 ligand 엣지보다 얼마나 더 중요한지 비교.

사용법:
  python analysis/run_stats_binary.py \
    --xai_dir results/pipeline/xai/holdout/seed42 \
    --output_dir results/stats_binary
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from scipy import stats as scipy_stats

TOPK_LIST = [5, 10, 15, 20, 25]
MODELS = [
    "GEMS_B6AEPL_CleanSplit",
    "GEMS_B6AEPL_PDBbind",
    "GC_GNN_CleanSplit",
    "GC_GNN_PDBbind",
]
GROUPS = ["low", "medium", "high"]


# ── 핵심 지표 (protein 제외 baseline) ─────────────────────────────────────────

def compute_binary_topk(df: pd.DataFrame) -> dict:
    """
    protein 엣지를 제외한 이진 분석:
    baseline = interaction / (interaction + ligand)
    lift     = (top-k interaction 비율) / baseline
    """
    # protein 엣지 제외
    df_bin = df[df["type"].isin(["interaction", "ligand"])].copy()
    total  = len(df_bin)
    if total == 0:
        return {}

    n_inter  = (df_bin["type"] == "interaction").sum()
    baseline = n_inter / total  # interaction / (interaction + ligand)

    if baseline <= 0 or baseline >= 1:
        return {}

    df_sorted = df.sort_values("abs_shapley", ascending=False).reset_index(drop=True)

    stats = {}
    for k in TOPK_LIST:
        topk     = df_sorted.head(k)
        # top-k 중 interaction 비율 (전체 엣지 기준 top-k)
        n_i_topk = (topk["type"] == "interaction").sum()
        n_l_topk = (topk["type"] == "ligand").sum()
        ratio_i  = n_i_topk / k
        ratio_l  = n_l_topk / k
        lift_i   = ratio_i / baseline
        lift_l   = ratio_l / (1 - baseline) if (1 - baseline) > 0 else float("nan")

        stats[k] = {
            "interaction":          float(ratio_i),
            "ligand":               float(ratio_l),
            "baseline_interaction": float(baseline),
            "lift_interaction":     float(lift_i),
            "lift_ligand":          float(lift_l),
        }
    return stats


# ── BH 보정 ───────────────────────────────────────────────────────────────────

def _bh_correction(p_values: list, alpha: float = 0.05) -> list:
    n = len(p_values)
    if n == 0:
        return []
    arr        = np.array(p_values, dtype=float)
    sorted_idx = np.argsort(arr)
    sorted_p   = arr[sorted_idx]
    bh_crit    = np.arange(1, n + 1) / n * alpha
    sig_mask   = sorted_p <= bh_crit
    reject     = np.zeros(n, dtype=bool)
    if np.any(sig_mask):
        cutoff = int(np.where(sig_mask)[0][-1])
        reject[sorted_idx[:cutoff + 1]] = True
    return reject.tolist()


# ── 통계 검정 ──────────────────────────────────────────────────────────────────

def run_statistics(all_results: dict) -> dict:
    stat = {}

    # [1] Lift > 1 (Wilcoxon one-sided, complex-level)
    # 통계 단위: complex 1개 = lift 값 1개 (edge 단위 아님)
    # effect size: matched-pairs rank-biserial r = 2*W / (n*(n+1)/2) - 1
    wilcoxon = {}
    p_vals, keys = [], []
    for model in MODELS:
        for grp in GROUPS:
            for k in TOPK_LIST:
                lifts = [
                    s["topk_stats"][k]["lift_interaction"]
                    for s in all_results.get(model, {}).get(grp, {}).get("per_sample", [])
                    if k in s["topk_stats"] and not np.isnan(s["topk_stats"][k]["lift_interaction"])
                ]
                if len(lifts) < 5:
                    continue
                try:
                    diffs = [v - 1.0 for v in lifts]
                    W, p  = scipy_stats.wilcoxon(diffs, alternative="greater")
                    n     = len(lifts)
                    r_rb  = float(2 * W / (n * (n + 1) / 2) - 1)  # rank-biserial r
                    keys.append((model, grp, k))
                    p_vals.append(float(p))
                    wilcoxon[f"{model}|{grp}|k{k}"] = {
                        "median_lift": round(float(np.median(lifts)), 4),
                        "iqr_lift":    round(float(np.percentile(lifts, 75) - np.percentile(lifts, 25)), 4),
                        "p":           round(float(p), 6),
                        "effect_r":    round(r_rb, 4),
                        "n":           n,
                    }
                except Exception as e:
                    wilcoxon[f"{model}|{grp}|k{k}"] = {"error": str(e)}

    bh = _bh_correction(p_vals)
    for i, key in enumerate(keys):
        wilcoxon[f"{key[0]}|{key[1]}|k{key[2]}"]["sig_bh"] = bool(bh[i])
    stat["lift_vs_one"] = wilcoxon

    # [2] 모델 간 차이 (Friedman test, paired design, 대표 k=15)
    # 동일 holdout 복합체에 대해 4개 모델 평가 → paired design → Friedman 적절
    k = 15
    friedman = {}
    for grp in GROUPS:
        id_lifts = {}
        for model in MODELS:
            samples = all_results.get(model, {}).get(grp, {}).get("per_sample", [])
            id_lifts[model] = {
                s["id"]: s["topk_stats"][k]["lift_interaction"]
                for s in samples
                if k in s["topk_stats"] and not np.isnan(s["topk_stats"][k]["lift_interaction"])
            }
        common_ids = sorted(
            set.intersection(*[set(d.keys()) for d in id_lifts.values()])
        ) if id_lifts else []
        if len(common_ids) < 5:
            friedman[grp] = {"skipped": f"공통 샘플 부족 ({len(common_ids)}개)"}
            continue
        try:
            groups_aligned = [
                [id_lifts[m][pid] for pid in common_ids] for m in MODELS
            ]
            chi2, p = scipy_stats.friedmanchisquare(*groups_aligned)
            friedman[grp] = {
                "chi2": round(float(chi2), 4),
                "p":    round(float(p), 6),
                "sig":  bool(p < 0.05),
                "n_common": len(common_ids),
            }
        except Exception as e:
            friedman[grp] = {"error": str(e)}
    stat["between_model"] = friedman

    # [3] Within-model JT 트렌드 (k=15)
    jt = {}
    for model in MODELS:
        samples_dict = {}
        for grp in GROUPS:
            vals = [
                s["topk_stats"][k]["lift_interaction"]
                for s in all_results.get(model, {}).get(grp, {}).get("per_sample", [])
                if k in s["topk_stats"] and not np.isnan(s["topk_stats"][k]["lift_interaction"])
            ]
            if vals:
                samples_dict[grp] = vals

        ordered = [samples_dict[g] for g in GROUPS if g in samples_dict and len(samples_dict[g]) >= 2]
        if len(ordered) < 2:
            jt[model] = {"skipped": "그룹 부족"}
            continue
        try:
            ns = [len(g) for g in ordered]
            N  = sum(ns)
            J  = sum(
                1.0 if xj > xi else (0.5 if xj == xi else 0.0)
                for i in range(len(ordered))
                for j in range(i + 1, len(ordered))
                for xi in ordered[i]
                for xj in ordered[j]
            )
            E_J   = (N * N - sum(n * n for n in ns)) / 4.0
            Var_J = (N * N * (2 * N + 3) - sum(n * n * (2 * n + 3) for n in ns)) / 72.0
            z     = (J - E_J) / Var_J ** 0.5 if Var_J > 0 else 0.0
            p     = float(scipy_stats.norm.sf(z))
            jt[model] = {
                "J":           round(J, 2),
                "z":           round(z, 4),
                "p":           round(p, 6),
                "sig":         bool(p < 0.05),
                "n_per_group": {g: len(samples_dict[g]) for g in GROUPS if g in samples_dict},
                "note":        "단일 seed 결과. 다중 비교 보정 전 값." if p < 0.05 else "",
            }
        except Exception as e:
            jt[model] = {"error": str(e)}
    stat["within_model_jt"] = jt

    return stat


# ── 출력 ───────────────────────────────────────────────────────────────────────

def print_summary(all_results: dict, stat: dict):
    k = 15
    print(f"\n{'='*70}")
    print(f"  Binary Baseline 분석 (protein 제외, k={k})")
    print(f"  baseline = interaction / (interaction + ligand)")
    print(f"{'='*70}")

    print(f"\n{'모델':<35} {'그룹':<8} {'baseline':>10} {'median lift':>12} {'p':>10} {'BH':>6}")
    print("-" * 80)
    for model in MODELS:
        for grp in GROUPS:
            key = f"{model}|{grp}|k{k}"
            val = stat["lift_vs_one"].get(key, {})
            samples = all_results.get(model, {}).get(grp, {}).get("per_sample", [])
            baselines = [s["topk_stats"][k]["baseline_interaction"]
                         for s in samples if k in s["topk_stats"]]
            bl = np.mean(baselines) if baselines else float("nan")
            sig = "★" if val.get("sig_bh") else " "
            ml  = val.get("median_lift", float("nan"))
            p   = val.get("p", float("nan"))
            print(f"{sig} {model:<33} {grp:<8} {bl:>10.3f} {ml:>12.4f} {p:>10.6f} {str(val.get('sig_bh','')):>6}")

    print(f"\n[Within-model JT 트렌드, k={k}]")
    print("-" * 60)
    for model, val in stat["within_model_jt"].items():
        sig = "★" if val.get("sig") else " "
        p   = val.get("p", "?")
        print(f"{sig} {model:<40} p={p}  {val.get('conclusion','')}")

    print(f"\n[모델 간 차이 — Friedman test (paired), k={k}]")
    print("-" * 40)
    for grp, val in stat["between_model"].items():
        if "chi2" in val:
            sig = "★" if val.get("sig") else " "
            print(f"  {sig} {grp:<10} χ²={val['chi2']:.4f}  p={val.get('p','?')}  n={val.get('n_common','?')}")
        else:
            print(f"  {grp}: {val}")


# ── 메인 ───────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, dict):  return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj: return None
    try:
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
    except: pass
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xai_dir",    default=os.path.join(_ROOT, "results/pipeline/xai/holdout/seed42"))
    parser.add_argument("--output_dir", default=os.path.join(_ROOT, "results/stats_binary"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for model in MODELS:
        all_results[model] = {}
        model_dir = os.path.join(args.xai_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for grp in GROUPS:
            grp_dir = os.path.join(model_dir, grp)
            if not os.path.isdir(grp_dir):
                continue
            per_sample = []
            for pdb_id in sorted(os.listdir(grp_dir)):
                csv_path = os.path.join(grp_dir, pdb_id, f"{pdb_id}_shapley.csv")
                if not os.path.exists(csv_path):
                    continue
                df   = pd.read_csv(csv_path)
                topk = compute_binary_topk(df)
                if topk:
                    per_sample.append({"id": pdb_id, "topk_stats": topk})
            all_results[model][grp] = {"per_sample": per_sample}
            print(f"  {model}/{grp}: {len(per_sample)} 샘플")

    stat = run_statistics(all_results)
    print_summary(all_results, stat)

    with open(os.path.join(args.output_dir, "stats_binary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe({"results": all_results, "statistics": stat}), f, indent=2, ensure_ascii=False)
    print(f"\n[완료] {args.output_dir}/stats_binary.json 저장")


if __name__ == "__main__":
    main()
