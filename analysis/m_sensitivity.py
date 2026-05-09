"""
M sensitivity 분석 — EdgeSHAPer Monte Carlo 수렴 검증
======================================================
동일 복합체에 M=10,20,50,100,200을 적용해 Shapley 순위 안정성 확인.
M=50이 M=100 대비 Spearman ρ ≥ 0.95이면 충분한 것으로 판정.

전제:
  - GEMS_B6AEPL_CleanSplit seed42 체크포인트가 results/pipeline/checkpoints/에 존재
  - B6AEPL CleanSplit 데이터셋 및 id_split.json 존재

사용법 (서버):
  python analysis/m_sensitivity.py \
    --dataset    /workspace/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt \
    --ckpt       results/pipeline/checkpoints/gems_b6aepl_cleansplit_seed42_best.pt \
    --split_json results/pipeline/id_split.json \
    --n_samples  10 \
    --output_dir results/m_sensitivity

기준 판정 (M_ref=100 기준):
  Spearman ρ(M=50, M=100) ≥ 0.95 → M=50 사용 가능
  Spearman ρ(M=50, M=100) < 0.95 → M=100 이상 권장
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

M_VALUES = [10, 20, 50, 100, 200]
M_REF    = 100    # 기준 M
M_TARGET = 50     # 검증 대상 M
RHO_THRESHOLD = 0.95


def run_sensitivity(model, graphs, M_values, device, seed=42, output_dir=None):
    """
    n_samples개 복합체 × len(M_values)개 M에 대해 Shapley 계산.
    복합체 1개 완료마다 output_dir/m_sensitivity_raw.json에 중간 저장.

    Returns:
        results: {pdb_id: {M: [abs_shapley 리스트]}}
    """
    from pipeline.xai_analyzer import EdgeSHAPer4GEMS

    # 이전 중간 저장 파일 있으면 이어서 실행
    results = {}
    if output_dir:
        raw_path = os.path.join(output_dir, "m_sensitivity_raw.json")
        if os.path.exists(raw_path):
            with open(raw_path) as f:
                saved = json.load(f)
            results = {pid: {int(M): vals for M, vals in by_m.items()}
                       for pid, by_m in saved.items()}
            print(f"  [재개] 이전 저장 데이터 {len(results)}개 복합체 로드")

    for i, graph in enumerate(graphs):
        pdb_id = graph.id
        if pdb_id in results:
            print(f"  [{i+1}/{len(graphs)}] {pdb_id} — 이미 완료, 건너뜀")
            continue
        print(f"  [{i+1}/{len(graphs)}] {pdb_id} ({graph.edge_index.shape[1]} edges)")
        results[pdb_id] = {}
        explainer = EdgeSHAPer4GEMS([model], graph, device)
        for M in M_values:
            phi = explainer.explain(M=M, seed=seed + i)
            results[pdb_id][M] = [abs(v) for v in phi]
            print(f"    M={M:3d}: done")

        # 복합체 1개 완료마다 중간 저장
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            raw = {pid: {str(M): vals for M, vals in by_m.items()}
                   for pid, by_m in results.items()}
            with open(os.path.join(output_dir, "m_sensitivity_raw.json"), "w") as f:
                json.dump(raw, f, indent=2)
            print(f"    → 중간 저장 완료 ({len(results)}/{len(graphs)})")

    return results


def compute_stability(results, M_ref, M_values):
    """
    각 (복합체, M) 쌍에 대해 M_ref 기준 Spearman ρ 계산.

    Returns:
        DataFrame: [pdb_id, M, spearman_rho, p_value, n_edges]
    """
    rows = []
    ref_key = M_ref
    for pdb_id, phi_by_m in results.items():
        if ref_key not in phi_by_m:
            continue
        ref_phi = phi_by_m[ref_key]
        n_edges = len(ref_phi)
        for M in M_values:
            if M == ref_key or M not in phi_by_m:
                continue
            rho, p = spearmanr(phi_by_m[M], ref_phi)
            rows.append({
                "pdb_id":      pdb_id,
                "M":           M,
                "spearman_rho": round(float(rho), 4),
                "p_value":      round(float(p),   6),
                "n_edges":      n_edges,
            })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, M_ref: int, M_target: int, threshold: float):
    print(f"\n{'='*65}")
    print(f"  M Sensitivity 결과 (기준: M={M_ref})")
    print(f"  판정 기준: Spearman ρ ≥ {threshold} → 사용 가능")
    print(f"{'='*65}")

    print(f"\n{'M':>6} | {'mean ρ':>8} | {'min ρ':>7} | {'std ρ':>7} | {'n':>4} | 판정")
    print("-" * 50)
    for M, grp in df.groupby("M"):
        mean_rho = grp["spearman_rho"].mean()
        min_rho  = grp["spearman_rho"].min()
        std_rho  = grp["spearman_rho"].std()
        n        = len(grp)
        ok = "✓ 사용 가능" if mean_rho >= threshold else "✗ 부족"
        marker = "★" if M == M_target else " "
        print(f"{marker}{M:>5} | {mean_rho:>8.4f} | {min_rho:>7.4f} | {std_rho:>7.4f} | {n:>4} | {ok}")

    # 복합체별 상세
    target_rows = df[df["M"] == M_target]
    if not target_rows.empty:
        print(f"\n[M={M_target} 복합체별 Spearman ρ vs M={M_ref}]")
        print("-" * 45)
        for _, row in target_rows.iterrows():
            flag = "✓" if row["spearman_rho"] >= threshold else "✗"
            print(f"  {flag} {row['pdb_id']:<12} ρ={row['spearman_rho']:.4f}  "
                  f"(n_edges={row['n_edges']})")

        mean_rho = target_rows["spearman_rho"].mean()
        verdict  = "★ M=50 사용 가능 (full pipeline 진행)" if mean_rho >= threshold \
                   else "✗ M=100 이상 권장 (M=100으로 재설정 후 진행)"
        print(f"\n{'='*65}")
        print(f"  최종 판정: {verdict}")
        print(f"  mean ρ(M={M_target} vs M={M_ref}) = {mean_rho:.4f}")
        print(f"{'='*65}")


def plot_sensitivity(df: pd.DataFrame, M_values: list, M_ref: int,
                     M_target: int, threshold: float, output_dir: str):
    """M sensitivity 수렴 곡선 저장."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    for pdb_id, grp in df.groupby("pdb_id"):
        grp_s = grp.sort_values("M")
        ax.plot(grp_s["M"], grp_s["spearman_rho"],
                color="gray", alpha=0.4, linewidth=1)

    summary = df.groupby("M")["spearman_rho"].agg(["mean", "std"]).reset_index()
    ax.plot(summary["M"], summary["mean"], color="#1F77B4",
            linewidth=2.5, marker="o", markersize=7, label="Mean ρ")
    ax.fill_between(summary["M"],
                    summary["mean"] - summary["std"],
                    summary["mean"] + summary["std"],
                    alpha=0.2, color="#1F77B4", label="Mean ± SD")

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold ρ = {threshold}")
    if M_target in df["M"].values:
        ax.axvline(M_target, color="orange", linestyle=":", linewidth=1.5,
                   label=f"M = {M_target} (target)")

    for _, row in summary.iterrows():
        ax.annotate(f"{row['mean']:.3f}", (row["M"], row["mean"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#1F77B4")

    ax.set_xlabel("M (Monte Carlo iterations)", fontsize=11)
    ax.set_ylabel(f"Spearman ρ vs. M={M_ref}", fontsize=11)
    ax.set_title("EdgeSHAPer M Sensitivity\n(Shapley rank stability)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(sorted(df["M"].unique()))
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "m_sensitivity_plot.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True,
                        help="B6AEPL_train_cleansplit.pt 경로")
    parser.add_argument("--ckpt",       required=True,
                        help="GEMS_B6AEPL_CleanSplit seed42 체크포인트")
    parser.add_argument("--split_json", required=True,
                        help="id_split.json 경로")
    parser.add_argument("--n_samples",  type=int, default=10,
                        help="테스트할 복합체 수 (기본 10, 약 1-2시간)")
    parser.add_argument("--output_dir", default=os.path.join(_ROOT, "results/m_sensitivity"))
    parser.add_argument("--M_values",   default="10,20,50,100,200",
                        help="테스트할 M 값 목록 (쉼표 구분)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    import torch
    from pipeline.trainer import load_gems_checkpoint
    from pipeline.data_loader import load_gems_dataset, load_id_split, apply_id_split

    M_values = [int(m) for m in args.M_values.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"M 테스트 값: {M_values}")
    print(f"샘플 수: {args.n_samples}")

    # 모델 로드
    print(f"\n체크포인트 로드: {args.ckpt}")
    model = load_gems_checkpoint(args.ckpt, device)
    model.eval()

    # 데이터 로드 → holdout (test) 샘플만 사용
    print(f"\n데이터셋 로드: {args.dataset}")
    dataset = load_gems_dataset(args.dataset)
    train_ids, val_ids, test_ids = load_id_split(args.split_json)
    _, _, test_ds = apply_id_split(dataset, train_ids, val_ids, test_ids)

    # n_samples개 선택 (seed 고정)
    rng = np.random.default_rng(args.seed)
    n   = min(args.n_samples, len(test_ds))
    idx = rng.choice(len(test_ds), size=n, replace=False)
    graphs = [test_ds[int(i)] for i in idx]
    print(f"\n선택된 복합체 {n}개: {[g.id for g in graphs]}")

    # M sensitivity 실행
    print(f"\nM sensitivity 실행 중... (복합체당 {len(M_values)}×M 계산)")
    results = run_sensitivity(model, graphs, M_values, device, seed=args.seed,
                              output_dir=args.output_dir)

    # 안정성 분석
    df = compute_stability(results, M_ref=M_REF, M_values=M_values)
    print_summary(df, M_ref=M_REF, M_target=M_TARGET, threshold=RHO_THRESHOLD)

    # 결과 저장
    csv_path = os.path.join(args.output_dir, "m_sensitivity.csv")
    df.to_csv(csv_path, index=False)

    # raw Shapley 값 저장 (재분석용)
    raw = {
        pdb_id: {str(M): vals for M, vals in by_m.items()}
        for pdb_id, by_m in results.items()
    }
    with open(os.path.join(args.output_dir, "m_sensitivity_raw.json"), "w") as f:
        json.dump(raw, f, indent=2)

    # 시각화
    plot_sensitivity(df, M_values=M_values, M_ref=M_REF, M_target=M_TARGET,
                     threshold=RHO_THRESHOLD, output_dir=args.output_dir)

    print(f"\n[완료] {args.output_dir}/m_sensitivity.csv 저장")


if __name__ == "__main__":
    main()
