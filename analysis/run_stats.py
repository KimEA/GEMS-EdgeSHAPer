"""
XAI 결과 통계 분석 스크립트 (재실행용)
========================================
Shapley CSV에서 Lift, Wilcoxon, JT 트렌드, Friedman, BH-FDR 통계를 계산.
XAI를 다시 돌리지 않고 저장된 CSV에서 통계만 재실행할 때 사용.

통계 단위: complex 1개 = metric 값 1개 (edge 단위 집계 아님)
effect size: Wilcoxon matched-pairs rank-biserial r

사용법:
  python analysis/run_stats.py \
    --xai_dir results/pipeline/xai/holdout/seed42 \
    --output_dir results/stats
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

TOPK_LIST = [5, 10, 15, 20, 25]
MODELS = [
    "GEMS_B6AEPL_CleanSplit",
    "GEMS_B6AEPL_PDBbind",
    "GC_GNN_CleanSplit",
    "GC_GNN_PDBbind",
]
GROUPS = ["low", "medium", "high"]


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_per_sample(xai_dir: str) -> dict:
    """CSV에서 complex-level lift 값 로드."""
    all_results = {}
    for model in MODELS:
        all_results[model] = {}
        model_dir = os.path.join(xai_dir, model)
        if not os.path.isdir(model_dir):
            print(f"[skip] {model}")
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
                df = pd.read_csv(csv_path)
                n_total = len(df)
                if n_total == 0:
                    continue
                n_inter    = (df["type"] == "interaction").sum()
                n_lig      = (df["type"] == "ligand").sum()
                n_prot     = (df["type"] == "protein").sum()
                baseline_i = n_inter / n_total
                baseline_l = n_lig   / n_total
                baseline_p = n_prot  / n_total
                df_sorted  = df.sort_values("abs_shapley", ascending=False).reset_index(drop=True)
                topk = {}
                for k in TOPK_LIST:
                    top = df_sorted.head(k)
                    ri  = (top["type"] == "interaction").sum() / k
                    rl  = (top["type"] == "ligand").sum()      / k
                    rp  = (top["type"] == "protein").sum()     / k
                    topk[k] = {
                        "interaction":          float(ri),
                        "ligand":               float(rl),
                        "protein":              float(rp),
                        "baseline_interaction": float(baseline_i),
                        "baseline_ligand":      float(baseline_l),
                        "baseline_protein":     float(baseline_p),
                        "lift_interaction":     float(ri / baseline_i) if baseline_i > 0 else float("nan"),
                        "lift_ligand":          float(rl / baseline_l) if baseline_l > 0 else float("nan"),
                        "lift_protein":         float(rp / baseline_p) if baseline_p > 0 else float("nan"),
                    }
                per_sample.append({"id": pdb_id, "topk_stats": topk})
            all_results[model][grp] = {"per_sample": per_sample}
            print(f"  {model}/{grp}: {len(per_sample)} 샘플")
    return all_results


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


# ── 통계 검정 ─────────────────────────────────────────────────────────────────

def run_statistics(all_results: dict) -> dict:
    stat = {}

    # [1] Lift > 1 (Wilcoxon one-sided, complex-level)
    # effect size: matched-pairs rank-biserial r = 2W/(n(n+1)/2) - 1
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
                    r_rb  = float(2 * W / (n * (n + 1) / 2) - 1)
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
    # 동일 holdout 복합체에 4개 모델 적용 → paired design → Friedman 적절
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
                "chi2":     round(float(chi2), 4),
                "p":        round(float(p), 6),
                "sig":      bool(p < 0.05),
                "n_common": len(common_ids),
            }
        except Exception as e:
            friedman[grp] = {"error": str(e)}
    stat["between_model_friedman"] = friedman

    # [3] Within-model JT 트렌드 (k=15)
    # 해석 주의: 단일 seed, 경계적 p-value 가능성 있음
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
            ns  = [len(g) for g in ordered]
            N   = sum(ns)
            J   = sum(
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
                "J":          round(J, 2),
                "z":          round(z, 4),
                "p":          round(p, 6),
                "sig":        bool(p < 0.05),
                "n_per_group": {g: len(samples_dict[g]) for g in GROUPS if g in samples_dict},
                "note":       "단일 seed 결과. 다중 비교 보정 전 값." if p < 0.05 else "",
            }
        except Exception as e:
            jt[model] = {"error": str(e)}
    stat["within_model_jt"] = jt

    return stat


# ── 출력 ─────────────────────────────────────────────────────────────────────

def print_summary(stat: dict):
    k = 15
    print(f"\n{'='*75}")
    print(f"  Lift@{k} 통계 (complex-level Wilcoxon, effect size = rank-biserial r)")
    print(f"{'='*75}")
    print(f"{'모델':<35} {'그룹':<8} {'median lift':>12} {'IQR':>8} {'r':>6} {'p':>10} {'BH':>5}")
    print("-" * 80)
    for model in MODELS:
        for grp in GROUPS:
            key = f"{model}|{grp}|k{k}"
            v   = stat["lift_vs_one"].get(key, {})
            if "median_lift" not in v:
                continue
            sig = "★" if v.get("sig_bh") else " "
            print(f"{sig} {model:<33} {grp:<8} "
                  f"{v['median_lift']:>12.4f} {v['iqr_lift']:>8.4f} "
                  f"{v['effect_r']:>6.3f} {v['p']:>10.6f} {str(v.get('sig_bh','')):>5}")

    print(f"\n[모델 간 차이 — Friedman test (paired), k={k}]")
    print("-" * 50)
    for grp, v in stat["between_model_friedman"].items():
        if "chi2" in v:
            sig = "★" if v["sig"] else " "
            print(f"  {sig} {grp:<10} χ²={v['chi2']:.4f}  p={v['p']:.6f}  n={v['n_common']}")
        else:
            print(f"  {grp}: {v}")

    print(f"\n[Within-model JT 트렌드, k={k}]")
    print("-" * 65)
    for model, v in stat["within_model_jt"].items():
        if "p" not in v:
            print(f"  {model}: {v}")
            continue
        sig = "★" if v["sig"] else " "
        note = " ⚠ 단일 seed" if v.get("note") else ""
        print(f"  {sig} {model:<40} z={v['z']:.3f}  p={v['p']:.6f}{note}")


# ── 직렬화 ───────────────────────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, dict):  return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj: return None
    try:
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
    except Exception: pass
    return obj


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xai_dir",    default=os.path.join(_ROOT, "results/pipeline/xai/holdout/seed42"))
    parser.add_argument("--output_dir", default=os.path.join(_ROOT, "results/stats"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[xai_dir] {args.xai_dir}")

    all_results = load_per_sample(args.xai_dir)
    stat        = run_statistics(all_results)
    print_summary(stat)

    out_path = os.path.join(args.output_dir, "stat_tests.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(stat), f, indent=2, ensure_ascii=False)
    print(f"\n[완료] {out_path} 저장")


if __name__ == "__main__":
    main()
