"""
방안 1: Contact Validation 분석
================================
Top-k Shapley 엣지 중 실제 근접 접촉(≤ 임계거리)인 interaction 엣지 비율을 계산.

distance_A 컬럼 = Cβ 거리 (Å). extract_edge_distances.py 실행 후 사용 가능.
그래프 구성 cutoff(5Å)와 동일한 threshold 사용을 권장.

지표:
  - contact_precision@k : Top-k 중 close-contact interaction 엣지 비율
  - baseline_contact    : 전체 interaction 엣지 중 close-contact 비율
  - contact_lift@k      : contact_precision / baseline_contact

사용법:
  python analysis/contact_validation.py \
    --xai_dir results/pipeline/xai/holdout/seed42 \
    --output_dir results/contact_validation \
    --threshold 5.0
"""

import os, json, argparse
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from scipy import stats as scipy_stats

TOPK_LIST   = [5, 10, 15, 20, 25]
MODELS      = [
    "GEMS_B6AEPL_CleanSplit",
    "GEMS_B6AEPL_PDBbind",
    "GC_GNN_CleanSplit",
    "GC_GNN_PDBbind",
]
GROUPS      = ["low", "medium", "high"]


# ── 핵심 지표 계산 ─────────────────────────────────────────────────────────────

def compute_contact_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """
    한 샘플의 Contact Validation 지표 계산.

    Args:
        df        : Shapley CSV (edge_idx, src, dst, type, shapley, abs_shapley, distance_A)
        threshold : 근접 접촉 거리 임계값 (Å)

    Returns:
        dict with contact_precision@k, baseline_contact, contact_lift@k
    """
    if df.empty or "distance_A" not in df.columns:
        return {}

    # interaction 엣지만 추출
    inter = df[df["type"] == "interaction"].copy()
    if inter.empty:
        return {}

    # baseline: 전체 엣지 중 close-contact interaction 비율 (랜덤 선택 기준점)
    # → original lift_interaction의 baseline(n_inter/n_total)과 동일한 구조
    n_total      = len(df)
    n_close_all  = (inter["distance_A"] <= threshold).sum()
    baseline     = n_close_all / n_total if n_total > 0 else float("nan")

    # Top-k: abs_shapley 기준 전체 엣지 정렬
    df_sorted = df.sort_values("abs_shapley", ascending=False).reset_index(drop=True)

    result = {"baseline_contact": baseline}

    for k in TOPK_LIST:
        topk = df_sorted.head(k)
        topk_inter = topk[topk["type"] == "interaction"]

        # contact_precision@k: Top-k 중 close-contact interaction 비율
        n_close_topk       = (topk_inter["distance_A"] <= threshold).sum()
        contact_precision  = n_close_topk / k

        # contact_lift@k
        contact_lift = (contact_precision / baseline) if baseline > 0 else float("nan")

        result[f"contact_precision_{k}"] = float(contact_precision)
        result[f"contact_lift_{k}"]      = float(contact_lift)

        # 추가: interaction-only top-k (interaction 엣지만 순위 매길 때)
        inter_sorted  = inter.sort_values("abs_shapley", ascending=False).reset_index(drop=True)
        topk_inter_only = inter_sorted.head(k)
        n_close_inter_topk = (topk_inter_only["distance_A"] <= threshold).sum()
        result[f"inter_topk_precision_{k}"] = float(n_close_inter_topk / k) if k <= len(inter) else float("nan")

    return result


# ── 통계 검정 ──────────────────────────────────────────────────────────────────

def _bh_correction(p_values: list, alpha: float = 0.05) -> list:
    n = len(p_values)
    if n == 0:
        return []
    arr = np.array(p_values, dtype=float)
    sorted_idx = np.argsort(arr)
    sorted_p   = arr[sorted_idx]
    bh_crit    = np.arange(1, n + 1) / n * alpha
    sig_mask   = sorted_p <= bh_crit
    reject     = np.zeros(n, dtype=bool)
    if np.any(sig_mask):
        cutoff = int(np.where(sig_mask)[0][-1])
        reject[sorted_idx[:cutoff + 1]] = True
    return reject.tolist()


def run_statistics(all_metrics: dict, k: int = 15) -> dict:
    """
    모델 × 그룹 간 contact_lift 통계 검정.
    """
    results = {}

    # 1) Lift > 1 검정 (Wilcoxon one-sided)
    wilcoxon = {}
    p_vals, keys = [], []
    for model in MODELS:
        for grp in GROUPS:
            lifts = [
                m[f"contact_lift_{k}"]
                for m in all_metrics.get(model, {}).get(grp, [])
                if not np.isnan(m.get(f"contact_lift_{k}", float("nan")))
            ]
            if len(lifts) < 5:
                continue
            try:
                stat, p = scipy_stats.wilcoxon(
                    [v - 1.0 for v in lifts], alternative="greater"
                )
                keys.append((model, grp))
                p_vals.append(float(p))
                wilcoxon[f"{model}|{grp}"] = {
                    "median_lift": round(float(np.median(lifts)), 4),
                    "p": round(float(p), 6),
                    "n": len(lifts),
                }
            except Exception as e:
                wilcoxon[f"{model}|{grp}"] = {"error": str(e)}

    bh = _bh_correction(p_vals)
    for i, key in enumerate(keys):
        wilcoxon[f"{key[0]}|{key[1]}"]["sig_bh"] = bool(bh[i])

    results["lift_vs_one"] = wilcoxon

    # 2) 모델 간 차이 (Kruskal-Wallis)
    kw = {}
    for grp in GROUPS:
        groups_data = []
        for model in MODELS:
            lifts = [
                m[f"contact_lift_{k}"]
                for m in all_metrics.get(model, {}).get(grp, [])
                if not np.isnan(m.get(f"contact_lift_{k}", float("nan")))
            ]
            if lifts:
                groups_data.append(lifts)
        if len(groups_data) >= 2:
            try:
                stat, p = scipy_stats.kruskal(*groups_data)
                kw[grp] = {"H": round(float(stat), 4), "p": round(float(p), 6),
                           "sig": bool(p < 0.05)}
            except Exception as e:
                kw[grp] = {"error": str(e)}
    results["between_model"] = kw

    return results


# ── 요약 출력 ──────────────────────────────────────────────────────────────────

def print_summary(all_metrics: dict, stat_results: dict, k: int = 15):
    print(f"\n{'='*70}")
    print(f"  Contact Validation 결과 (k={k}, 임계거리 기반 근접 접촉)")
    print(f"{'='*70}")

    print(f"\n{'모델':<35} {'그룹':<8} {'baseline':>10} {'precision':>10} {'lift':>8} {'n':>5}")
    print("-" * 75)
    for model in MODELS:
        for grp in GROUPS:
            samples = all_metrics.get(model, {}).get(grp, [])
            if not samples:
                continue
            baselines  = [m["baseline_contact"] for m in samples if not np.isnan(m.get("baseline_contact", float("nan")))]
            precisions = [m.get(f"contact_precision_{k}", float("nan")) for m in samples]
            lifts      = [m.get(f"contact_lift_{k}", float("nan")) for m in samples]
            precisions = [v for v in precisions if not np.isnan(v)]
            lifts      = [v for v in lifts      if not np.isnan(v)]
            if not lifts:
                continue
            print(f"  {model:<33} {grp:<8} "
                  f"{np.mean(baselines):>10.3f} "
                  f"{np.mean(precisions):>10.3f} "
                  f"{np.mean(lifts):>8.3f} "
                  f"{len(lifts):>5}")

    print(f"\n[Lift > 1.0 검정, k={k}]")
    print(f"{'모델':<35} {'그룹':<8} {'median lift':>12} {'p':>10} {'BH sig':>8}")
    print("-" * 75)
    for key, val in stat_results.get("lift_vs_one", {}).items():
        model, grp = key.split("|")
        sig = "★" if val.get("sig_bh") else " "
        p   = val.get("p", float("nan"))
        ml  = val.get("median_lift", float("nan"))
        print(f"{sig} {model:<33} {grp:<8} {ml:>12.4f} {p:>10.6f} {str(val.get('sig_bh','')):>8}")

    print(f"\n[모델 간 차이, k={k}]")
    for grp, val in stat_results.get("between_model", {}).items():
        sig = "★" if val.get("sig") else " "
        print(f"  {sig} {grp:<10} H={val.get('H','?'):.4f}  p={val.get('p','?'):.6f}")


# ── 메인 ───────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, dict):  return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj: return None
    try:
        import numpy as np
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
    except ImportError:
        pass
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xai_dir",    default=os.path.join(_ROOT, "results/pipeline/xai/holdout/seed42"))
    parser.add_argument("--output_dir", default=os.path.join(_ROOT, "results/contact_validation"))
    parser.add_argument("--threshold",  type=float, default=5.0,
                        help="근접 접촉 거리 임계값 (Å, 기본 5.0 = 그래프 구성 cutoff)")
    parser.add_argument("--k",          type=int, default=15,
                        help="통계 요약에 사용할 대표 k 값")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}
    for model in MODELS:
        all_metrics[model] = {}
        model_dir = os.path.join(args.xai_dir, model)
        if not os.path.isdir(model_dir):
            print(f"[skip] {model}")
            continue
        for grp in GROUPS:
            grp_dir = os.path.join(model_dir, grp)
            if not os.path.isdir(grp_dir):
                continue
            samples = []
            for pdb_id in sorted(os.listdir(grp_dir)):
                csv_path = os.path.join(grp_dir, pdb_id, f"{pdb_id}_shapley.csv")
                if not os.path.exists(csv_path):
                    continue
                df  = pd.read_csv(csv_path)
                met = compute_contact_metrics(df, args.threshold)
                if met:
                    met["pdb_id"] = pdb_id
                    samples.append(met)
            all_metrics[model][grp] = samples
            print(f"  {model}/{grp}: {len(samples)} 샘플")

    stat_results = run_statistics(all_metrics, k=args.k)
    print_summary(all_metrics, stat_results, k=args.k)

    # 결과 저장
    out = {
        "threshold_A": args.threshold,
        "metrics":     _json_safe(all_metrics),
        "statistics":  _json_safe(stat_results),
    }
    out_path = os.path.join(args.output_dir, f"contact_validation_thr{args.threshold}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[완료] {out_path} 저장")


if __name__ == "__main__":
    main()
