"""
방안 2: Residue-Level Shapley 집계
=====================================
interaction 엣지의 Shapley 값을 단백질 잔기(residue) 단위로 집계.

- 각 단백질 노드(dst >= n_lig)로 연결된 interaction 엣지의 abs_shapley 합산
- 잔기별 중요도 순위 분석
- 모델 간 잔기 중요도 일치도(Spearman 상관) 비교

사용법:
  python analysis/residue_analysis.py \
    --xai_dir results/pipeline/xai/holdout/seed42 \
    --output_dir results/residue_analysis
"""

import os, json, argparse
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    "GEMS_B6AEPL_CleanSplit",
    "GEMS_B6AEPL_PDBbind",
    "GC_GNN_CleanSplit",
    "GC_GNN_PDBbind",
]
GROUPS = ["low", "medium", "high"]


# ── 잔기 집계 ──────────────────────────────────────────────────────────────────

def aggregate_by_residue(df: pd.DataFrame) -> pd.DataFrame:
    """
    interaction 엣지의 abs_shapley를 dst 노드(단백질 잔기)로 집계.

    Returns:
        DataFrame with columns [residue_node, total_shapley, n_edges, mean_shapley]
    """
    inter = df[df["type"] == "interaction"].copy()
    if inter.empty:
        return pd.DataFrame()

    agg = inter.groupby("dst").agg(
        total_shapley=("abs_shapley", "sum"),
        n_edges=("abs_shapley", "count"),
        mean_shapley=("abs_shapley", "mean"),
        mean_distance=("distance_A", "mean") if "distance_A" in inter.columns else ("abs_shapley", "count"),
    ).reset_index().rename(columns={"dst": "residue_node"})

    agg["rank"] = agg["total_shapley"].rank(ascending=False, method="min").astype(int)
    return agg.sort_values("rank")


def compute_sample_stats(df: pd.DataFrame) -> dict:
    """샘플 하나의 잔기 집계 요약 통계."""
    res = aggregate_by_residue(df)
    if res.empty:
        return {}

    top1  = res.iloc[0]
    top3  = res.head(3)
    n_res = len(res)

    result = {
        "n_residues":          n_res,
        "top1_shapley":        float(top1["total_shapley"]),
        "top3_shapley_sum":    float(top3["total_shapley"].sum()),
        "total_shapley_sum":   float(res["total_shapley"].sum()),
        "concentration_top1":  float(top1["total_shapley"] / res["total_shapley"].sum()),
        "concentration_top3":  float(top3["total_shapley"].sum() / res["total_shapley"].sum()),
    }
    if "mean_distance" in res.columns:
        result["top1_mean_distance"] = float(top1["mean_distance"])
        result["top3_mean_distance"] = float(top3["mean_distance"].mean())
    return result


# ── 모델 간 잔기 일치도 ────────────────────────────────────────────────────────

def compute_model_agreement(all_residue_data: dict) -> dict:
    """
    동일 샘플에 대해 두 모델이 중요하게 본 잔기 순위의 Spearman 상관.
    all_residue_data: {model: {pdb_id: residue_df}}
    """
    model_pairs = [
        ("GEMS_B6AEPL_CleanSplit", "GEMS_B6AEPL_PDBbind"),
        ("GC_GNN_CleanSplit",      "GC_GNN_PDBbind"),
        ("GEMS_B6AEPL_CleanSplit", "GC_GNN_CleanSplit"),
        ("GEMS_B6AEPL_PDBbind",    "GC_GNN_PDBbind"),
    ]

    results = {}
    for m1, m2 in model_pairs:
        corrs = []
        pdb_ids = set(all_residue_data.get(m1, {}).keys()) & \
                  set(all_residue_data.get(m2, {}).keys())
        for pdb_id in pdb_ids:
            df1 = all_residue_data[m1][pdb_id]
            df2 = all_residue_data[m2][pdb_id]
            if df1.empty or df2.empty:
                continue
            common = set(df1["residue_node"]) & set(df2["residue_node"])
            if len(common) < 3:
                continue
            s1 = df1.set_index("residue_node").loc[list(common), "total_shapley"]
            s2 = df2.set_index("residue_node").loc[list(common), "total_shapley"]
            r, p = scipy_stats.spearmanr(s1, s2)
            if not np.isnan(r):
                corrs.append(float(r))

        results[f"{m1} vs {m2}"] = {
            "mean_spearman": round(float(np.mean(corrs)), 4) if corrs else float("nan"),
            "std_spearman":  round(float(np.std(corrs)),  4) if corrs else float("nan"),
            "n_samples":     len(corrs),
        }
    return results


# ── 출력 ───────────────────────────────────────────────────────────────────────

def print_summary(all_stats: dict, agreement: dict):
    print(f"\n{'='*70}")
    print("  Residue-Level Shapley 집계 결과")
    print(f"{'='*70}")

    print(f"\n{'모델':<35} {'그룹':<8} {'평균 잔기수':>10} {'Top1 집중도':>12} {'Top3 집중도':>12}")
    print("-" * 80)
    for model in MODELS:
        for grp in GROUPS:
            samples = all_stats.get(model, {}).get(grp, [])
            if not samples:
                continue
            n_res  = [s["n_residues"]         for s in samples if s]
            top1c  = [s["concentration_top1"] for s in samples if s]
            top3c  = [s["concentration_top3"] for s in samples if s]
            print(f"  {model:<33} {grp:<8} "
                  f"{np.mean(n_res):>10.1f} "
                  f"{np.mean(top1c):>12.3f} "
                  f"{np.mean(top3c):>12.3f}")

    print(f"\n[모델 간 잔기 중요도 일치도 (Spearman r)]")
    print("-" * 60)
    for pair, val in agreement.items():
        r = val['mean_spearman']
        n = val['n_samples']
        print(f"  {pair:<50} r={r:.4f}  (n={n})")


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
    parser.add_argument("--output_dir", default=os.path.join(_ROOT, "results/residue_analysis"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_stats        = {}
    all_residue_data = {}  # {model: {pdb_id: residue_df}}

    for model in MODELS:
        all_stats[model]        = {}
        all_residue_data[model] = {}
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
                res = aggregate_by_residue(df)
                if not res.empty:
                    all_residue_data[model][pdb_id] = res
                stat = compute_sample_stats(df)
                if stat:
                    stat["pdb_id"] = pdb_id
                    samples.append(stat)
            all_stats[model][grp] = samples
            print(f"  {model}/{grp}: {len(samples)} 샘플")

    agreement = compute_model_agreement(all_residue_data)
    print_summary(all_stats, agreement)

    out = {
        "summary_stats": _json_safe(all_stats),
        "model_agreement": _json_safe(agreement),
    }
    out_path = os.path.join(args.output_dir, "residue_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[완료] {out_path} 저장")


if __name__ == "__main__":
    main()
