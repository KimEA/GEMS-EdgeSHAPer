"""
다중 시드 Shapley CSV 병합 스크립트
=====================================
seed42 XAI 결과(results/pipeline/xai/holdout/seed42)와
seeds 123, 2024 결과(results/pipeline_seeds/xai/holdout/seed{N})를
복합체별로 abs_shapley 평균 → 단일 merged CSV 생성.

병합 결과는 run_stats.py, contact_validation.py 등 기존 분석 스크립트에
바로 입력 가능한 형식으로 저장됨.

사용법:
  python analysis/merge_seeds.py \
    --seed_dirs results/pipeline/xai/holdout/seed42 \
                results/pipeline_seeds/xai/holdout/seed123 \
                results/pipeline_seeds/xai/holdout/seed2024 \
    --output_dir results/pipeline_merged/xai/holdout
"""

import os, argparse
from typing import List, Optional
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    "GEMS_B6AEPL_CleanSplit",
    "GEMS_B6AEPL_PDBbind",
    "GC_GNN_CleanSplit",
    "GC_GNN_PDBbind",
]
GROUPS = ["low", "medium", "high"]


def merge_sample(csv_paths: List[str]) -> Optional[pd.DataFrame]:
    """
    동일 복합체의 여러 시드 CSV를 읽어 abs_shapley 평균.
    구조 열(edge_idx, src, dst, type, distance_A 등)은 첫 번째 시드 기준.
    """
    dfs = []
    for path in csv_paths:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if not dfs:
        return None
    if len(dfs) == 1:
        out = dfs[0].copy()
        out["n_seeds_merged"] = 1
        return out

    # 기준 df (첫 번째 시드)
    base = dfs[0].copy()
    n_rows = len(base)

    # 시드별 abs_shapley 집계
    abs_vals = np.array([
        df["abs_shapley"].values if len(df) == n_rows else np.full(n_rows, np.nan)
        for df in dfs
    ])
    shapley_vals = np.array([
        df["shapley"].values if "shapley" in df.columns and len(df) == n_rows
        else np.full(n_rows, np.nan)
        for df in dfs
    ])

    base["abs_shapley"] = np.nanmean(abs_vals, axis=0)
    if "shapley" in base.columns:
        base["shapley"] = np.nanmean(shapley_vals, axis=0)
    base["n_seeds_merged"] = (~np.isnan(abs_vals)).sum(axis=0)

    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_dirs", nargs="+", required=True,
        help="시드별 XAI 디렉터리 목록 (예: results/pipeline/xai/holdout/seed42 results/pipeline_seeds/xai/holdout/seed123)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(_ROOT, "results/pipeline_merged/xai/holdout/merged"),
    )
    args = parser.parse_args()

    n_seeds = len(args.seed_dirs)
    print(f"병합할 시드 수: {n_seeds}")
    for d in args.seed_dirs:
        print(f"  {d}")

    total_merged = 0
    for model in MODELS:
        for grp in GROUPS:
            out_grp = os.path.join(args.output_dir, model, grp)

            # 모든 시드에서 pdb_id 수집
            pdb_ids = set()
            for sd in args.seed_dirs:
                grp_dir = os.path.join(sd, model, grp)
                if os.path.isdir(grp_dir):
                    pdb_ids |= {
                        d for d in os.listdir(grp_dir)
                        if os.path.isdir(os.path.join(grp_dir, d))
                    }

            if not pdb_ids:
                continue

            for pdb_id in sorted(pdb_ids):
                csv_paths = [
                    os.path.join(sd, model, grp, pdb_id, f"{pdb_id}_shapley.csv")
                    for sd in args.seed_dirs
                ]
                merged = merge_sample(csv_paths)
                if merged is None:
                    continue

                out_dir = os.path.join(out_grp, pdb_id)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{pdb_id}_shapley.csv")
                merged.to_csv(out_path, index=False)
                total_merged += 1

            print(f"  {model}/{grp}: {len(pdb_ids)} 샘플 병합 완료")

    print(f"\n[완료] 총 {total_merged}개 샘플 병합")
    print(f"  출력 위치: {args.output_dir}")
    print(f"\n이후 분석 실행:")
    print(f"  python analysis/run_stats.py --xai_dir {args.output_dir}")
    print(f"  python analysis/contact_validation.py --xai_dir {args.output_dir}")


if __name__ == "__main__":
    main()
