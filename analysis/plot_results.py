"""
논문용 시각화 스크립트
========================
run_stats.py / contact_validation.py / residue_analysis.py / m_sensitivity.py
실행 후 생성된 CSV/JSON을 읽어 논문 그림을 일괄 생성.

생성 그림:
  fig1_lift_boxplot.png          — Lift@15 분포 (모델×그룹 박스플롯, 주요 결과)
  fig2_lift_topk_trend.png       — Lift@k 추이 (k=5~25, 모델별 선 그래프)
  fig3_model_comparison.png      — 4모델 Interaction 비율 비교
  fig4_m_sensitivity.png         — M sensitivity 수렴 곡선
  fig5_contact_validation.png    — Contact lift@15 (distance 기반 검증)
  fig6_residue_concentration.png — 잔기 Shapley 집중도
  fig7_distance_distribution.png — Top-k vs 전체 interaction edge 거리 분포
                                   (extract_edge_distances.py 실행 후 distance_A 컬럼 필요)
  fig8_topk_graphs/              — 복합체별 단백질-리간드 상호작용 그래프 (NetworkX)
                                   노드: 빨강(리간드) / 하늘색(단백질)
                                   Top-k 엣지: 타입별 색상, 두께 3 / 나머지: 회색, 두께 1.5

사용법:
  python analysis/plot_results.py \
    --xai_dir    results/pipeline/xai/holdout/seed42 \
    --stats_json results/stats/stat_tests.json \
    --contact_json results/contact_validation/contact_validation_thr5.0.json \
    --residue_json results/residue_analysis/residue_analysis.json \
    --msens_csv  results/m_sensitivity/m_sensitivity.csv \
    --output_dir results/figures \
    --topk_graph_k 5,10,15 \
    --topk_graph_n 3
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ── 상수 ───────────────────────────────────────────────────────────────────────
MODELS = [
    "GEMS_B6AEPL_CleanSplit",
    "GEMS_B6AEPL_PDBbind",
    "GC_GNN_CleanSplit",
    "GC_GNN_PDBbind",
]
MODEL_SHORT = {
    "GEMS_B6AEPL_CleanSplit": "GEMS\n(Clean)",
    "GEMS_B6AEPL_PDBbind":    "GEMS\n(PDBbind)",
    "GC_GNN_CleanSplit":      "GC-GNN\n(Clean)",
    "GC_GNN_PDBbind":         "GC-GNN\n(PDBbind)",
}
GROUPS = ["low", "medium", "high"]
GROUP_LABELS  = {"low": "Low\n(pKi<6)", "medium": "Medium\n(6.5≤pKi≤7.5)", "high": "High\n(pKi>8)"}
GROUP_COLORS  = {"low": "#74C476", "medium": "#FD8D3C", "high": "#E6550D"}
MODEL_COLORS  = {
    "GEMS_B6AEPL_CleanSplit": "#1F77B4",
    "GEMS_B6AEPL_PDBbind":    "#AEC7E8",
    "GC_GNN_CleanSplit":      "#FF7F0E",
    "GC_GNN_PDBbind":         "#FFBB78",
}
TOPK_LIST = [5, 10, 15, 20, 25]
K_REP     = 15
DPI       = 200


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_lift_data(xai_dir: str) -> dict:
    """xai_dir에서 복합체별 lift_interaction 값 로드."""
    data = {}
    for model in MODELS:
        data[model] = {}
        for grp in GROUPS:
            lifts_by_k = {k: [] for k in TOPK_LIST}
            grp_dir = os.path.join(xai_dir, model, grp)
            if not os.path.isdir(grp_dir):
                continue
            for pdb_id in sorted(os.listdir(grp_dir)):
                csv_path = os.path.join(grp_dir, pdb_id, f"{pdb_id}_shapley.csv")
                if not os.path.exists(csv_path):
                    continue
                df = pd.read_csv(csv_path)
                n_total = len(df)
                if n_total == 0:
                    continue
                n_inter   = (df["type"] == "interaction").sum()
                baseline  = n_inter / n_total
                if baseline <= 0:
                    continue
                df_s = df.sort_values("abs_shapley", ascending=False).reset_index(drop=True)
                for k in TOPK_LIST:
                    ri   = (df_s.head(k)["type"] == "interaction").sum() / k
                    lift = ri / baseline
                    if not np.isnan(lift):
                        lifts_by_k[k].append(float(lift))
            data[model][grp] = lifts_by_k
    return data


def _load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Figure 1: Lift@K_REP 박스플롯 ────────────────────────────────────────────

def fig1_lift_boxplot(lift_data: dict, stats_json: dict, output_dir: str):
    """
    4개 모델 × 3개 그룹 lift@15 분포를 4-패널 박스플롯으로 표현.
    BH 유의 그룹에 ★ 표시.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle(f"Interaction Edge Lift@{K_REP} Distribution\n"
                 f"(Shapley Top-{K_REP} vs. random baseline)",
                 fontsize=13, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        boxes, labels, colors, sig_flags = [], [], [], []
        for grp in GROUPS:
            vals = lift_data.get(model, {}).get(grp, {}).get(K_REP, [])
            boxes.append(vals if vals else [np.nan])
            labels.append(GROUP_LABELS[grp])
            colors.append(GROUP_COLORS[grp])

            # BH 유의 여부
            key = f"{model}|{grp}|k{K_REP}"
            sig = False
            if stats_json:
                entry = stats_json.get("lift_vs_one", {}).get(key, {})
                sig   = bool(entry.get("sig_bh", False))
            sig_flags.append(sig)

        bp = ax.boxplot(boxes, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2},
                        whiskerprops={"linewidth": 1.2},
                        capprops={"linewidth": 1.2},
                        flierprops={"marker": ".", "markersize": 4, "alpha": 0.5})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # ★ 유의 표시
        for i, (sig, vals) in enumerate(zip(sig_flags, boxes)):
            if sig and vals and not all(np.isnan(v) for v in vals):
                top = np.nanpercentile(vals, 95)
                ax.text(i + 1, top + 0.03, "★", ha="center", fontsize=14, color="red")

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(MODEL_SHORT[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Affinity Group", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(f"Lift@{K_REP} (interaction)", fontsize=10)

    legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], alpha=0.7,
                                     label=g.capitalize()) for g in GROUPS]
    legend_patches.append(Line2D([0], [0], color="gray", linestyle="--", label="Lift = 1.0 (random)"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(output_dir, "fig1_lift_boxplot.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[fig1] {path}")


# ── Figure 2: Lift@k 추이 ──────────────────────────────────────────────────────

def fig2_lift_topk_trend(lift_data: dict, output_dir: str):
    """
    k=5→25에 따른 lift_interaction 중앙값 추이 (모델별 패널, 그룹별 선).
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle("Interaction Lift@k Trend (k = 5 → 25)",
                 fontsize=13, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        for grp in GROUPS:
            medians, q25s, q75s = [], [], []
            for k in TOPK_LIST:
                vals = lift_data.get(model, {}).get(grp, {}).get(k, [])
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    medians.append(np.median(vals))
                    q25s.append(np.percentile(vals, 25))
                    q75s.append(np.percentile(vals, 75))
                else:
                    medians.append(np.nan)
                    q25s.append(np.nan)
                    q75s.append(np.nan)

            color = GROUP_COLORS[grp]
            ax.plot(TOPK_LIST, medians, marker="o", color=color,
                    linewidth=2, markersize=6, label=grp.capitalize())
            ax.fill_between(TOPK_LIST, q25s, q75s, color=color, alpha=0.15)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(TOPK_LIST)
        ax.set_xlabel("k", fontsize=10)
        ax.set_title(MODEL_SHORT[model], fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Median Lift (interaction)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_lift_topk_trend.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[fig2] {path}")


# ── Figure 3: 모델 비교 그룹 박스플롯 ─────────────────────────────────────────

def fig3_model_comparison(lift_data: dict, output_dir: str):
    """
    같은 그룹 내 4개 모델 lift@15를 나란히 비교하는 3-패널 그룹 박스플롯.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle(f"Model Comparison: Lift@{K_REP} by Affinity Group",
                 fontsize=13, fontweight="bold")

    for ax, grp in zip(axes, GROUPS):
        boxes, tick_labels = [], []
        for model in MODELS:
            vals = lift_data.get(model, {}).get(grp, {}).get(K_REP, [])
            boxes.append(vals if vals else [np.nan])
            tick_labels.append(MODEL_SHORT[model])

        bp = ax.boxplot(boxes, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2},
                        whiskerprops={"linewidth": 1.2},
                        capprops={"linewidth": 1.2},
                        flierprops={"marker": ".", "markersize": 4, "alpha": 0.5})
        for patch, model in zip(bp["boxes"], MODELS):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.8)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(range(1, len(MODELS) + 1))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(f"{grp.capitalize()} Affinity\n{GROUP_LABELS[grp]}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Model", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(f"Lift@{K_REP} (interaction)", fontsize=10)

    legend_patches = [mpatches.Patch(color=MODEL_COLORS[m], alpha=0.8,
                                     label=MODEL_SHORT[m].replace("\n", " "))
                      for m in MODELS]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(output_dir, "fig3_model_comparison.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[fig3] {path}")


# ── Figure 4: M sensitivity ────────────────────────────────────────────────────

def fig4_m_sensitivity(msens_csv: str, output_dir: str):
    """
    M sensitivity 수렴 곡선 (Spearman ρ vs M, 기준 M=100).
    """
    if not msens_csv or not os.path.exists(msens_csv):
        print("[fig4] m_sensitivity.csv 없음 — 건너뜀")
        return

    df = pd.read_csv(msens_csv)
    M_values = sorted(df["M"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))

    # 복합체별 선 (회색, 반투명)
    for pdb_id, grp in df.groupby("pdb_id"):
        grp_sorted = grp.sort_values("M")
        ax.plot(grp_sorted["M"], grp_sorted["spearman_rho"],
                color="gray", alpha=0.35, linewidth=1)

    # 평균 ± std 밴드
    summary = df.groupby("M")["spearman_rho"].agg(["mean", "std"]).reset_index()
    ax.plot(summary["M"], summary["mean"], color="#1F77B4",
            linewidth=2.5, marker="o", markersize=7, label="Mean ρ")
    ax.fill_between(summary["M"],
                    summary["mean"] - summary["std"],
                    summary["mean"] + summary["std"],
                    alpha=0.2, color="#1F77B4", label="Mean ± SD")

    # 판정 기준선
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1.5,
               label="Threshold ρ = 0.95")

    # M=50 수직선
    if 50 in M_values:
        ax.axvline(50, color="orange", linestyle=":", linewidth=1.5,
                   label="M = 50 (used)")

    ax.set_xlabel("M (Monte Carlo iterations)", fontsize=11)
    ax.set_ylabel("Spearman ρ vs. M=100", fontsize=11)
    ax.set_title("EdgeSHAPer M Sensitivity\n(Shapley rank stability)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(M_values)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 각 M의 mean ρ 값 표시
    for _, row in summary.iterrows():
        ax.annotate(f"{row['mean']:.3f}",
                    (row["M"], row["mean"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#1F77B4")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_m_sensitivity.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[fig4] {path}")


# ── Figure 5: Contact validation ──────────────────────────────────────────────

def fig5_contact_validation(contact_json: dict, output_dir: str):
    """
    모델×그룹별 contact_lift@15 막대 그래프 (mean ± std).
    """
    if not contact_json:
        print("[fig5] contact_validation JSON 없음 — 건너뜀")
        return

    metrics = contact_json.get("metrics", {})
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle(f"Contact Validation: Contact Lift@{K_REP}\n"
                 f"(Fraction of top-{K_REP} interaction edges within distance threshold)",
                 fontsize=12, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        means, stds, ns = [], [], []
        for grp in GROUPS:
            samples = metrics.get(model, {}).get(grp, [])
            key = f"contact_lift_{K_REP}"
            vals = [s[key] for s in samples
                    if key in s and not np.isnan(s.get(key, float("nan")))]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals)  if vals else 0)
            ns.append(len(vals))

        x = np.arange(len(GROUPS))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[GROUP_COLORS[g] for g in GROUPS],
                      alpha=0.8, edgecolor="black", linewidth=0.7,
                      error_kw={"linewidth": 1.5})
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        for i, (m, n) in enumerate(zip(means, ns)):
            ax.text(i, -0.08, f"n={n}", ha="center", fontsize=8, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=9)
        ax.set_title(MODEL_SHORT[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Affinity Group", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel(f"Contact Lift@{K_REP}", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig5_contact_validation.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[fig5] {path}")


# ── Figure 6: Residue 집중도 ───────────────────────────────────────────────────

def fig6_residue_concentration(residue_json: dict, output_dir: str):
    """
    모델×그룹별 잔기 Shapley 집중도 (Top-1, Top-3) 그룹 막대 그래프.
    """
    if not residue_json:
        print("[fig6] residue_analysis JSON 없음 — 건너뜀")
        return

    summary = residue_json.get("summary_stats", {})
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle("Residue-Level Shapley Concentration\n"
                 "(Fraction of total interaction Shapley in top residues)",
                 fontsize=12, fontweight="bold")

    bar_width = 0.35
    for ax, model in zip(axes, MODELS):
        top1_means, top3_means = [], []
        top1_stds,  top3_stds  = [], []
        for grp in GROUPS:
            samples = summary.get(model, {}).get(grp, [])
            c1 = [s["concentration_top1"] for s in samples if s and "concentration_top1" in s]
            c3 = [s["concentration_top3"] for s in samples if s and "concentration_top3" in s]
            top1_means.append(np.mean(c1) if c1 else 0)
            top3_means.append(np.mean(c3) if c3 else 0)
            top1_stds.append(np.std(c1)  if c1 else 0)
            top3_stds.append(np.std(c3)  if c3 else 0)

        x = np.arange(len(GROUPS))
        ax.bar(x - bar_width / 2, top1_means, bar_width, yerr=top1_stds,
               capsize=4, label="Top-1 residue",
               color="#1F77B4", alpha=0.8, edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 1.2})
        ax.bar(x + bar_width / 2, top3_means, bar_width, yerr=top3_stds,
               capsize=4, label="Top-3 residues",
               color="#FF7F0E", alpha=0.8, edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 1.2})

        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=9)
        ax.set_title(MODEL_SHORT[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Affinity Group", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Shapley Concentration (fraction)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig6_residue_concentration.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[fig6] {path}")


# ── Figure 7: Top-k vs 전체 interaction edge 거리 분포 ────────────────────────

def load_distance_data(xai_dir: str, k: int = 15) -> dict:
    """
    interaction 엣지의 거리 분포 수집.

    Returns:
        {model: {
            "all":   [distance_A 값, ...],   # 전체 interaction 엣지
            "topk":  [distance_A 값, ...],   # abs_shapley 상위 k개 중 interaction
            "botk":  [distance_A 값, ...],   # abs_shapley 하위 k개 중 interaction
        }}
        distance_A 컬럼 없으면 None 반환
    """
    data = {}
    has_distance = False

    for model in MODELS:
        all_dists, topk_dists, botk_dists = [], [], []
        for grp in GROUPS:
            grp_dir = os.path.join(xai_dir, model, grp)
            if not os.path.isdir(grp_dir):
                continue
            for pdb_id in sorted(os.listdir(grp_dir)):
                csv_path = os.path.join(grp_dir, pdb_id, f"{pdb_id}_shapley.csv")
                if not os.path.exists(csv_path):
                    continue
                df = pd.read_csv(csv_path)
                if "distance_A" not in df.columns:
                    continue
                has_distance = True

                inter = df[df["type"] == "interaction"].dropna(subset=["distance_A"])
                if inter.empty:
                    continue

                # 전체 interaction 엣지 거리
                all_dists.extend(inter["distance_A"].tolist())

                # abs_shapley 기준 전체 정렬 → top-k / bottom-k 중 interaction 엣지
                df_s = df.sort_values("abs_shapley", ascending=False).reset_index(drop=True)
                topk_inter = df_s.head(k)[df_s.head(k)["type"] == "interaction"]
                botk_inter = df_s.tail(k)[df_s.tail(k)["type"] == "interaction"]

                topk_dists.extend(topk_inter["distance_A"].dropna().tolist())
                botk_dists.extend(botk_inter["distance_A"].dropna().tolist())

        data[model] = {"all": all_dists, "topk": topk_dists, "botk": botk_dists}

    if not has_distance:
        return None
    return data


def fig7_distance_distribution(xai_dir: str, output_dir: str, k: int = 15,
                                threshold: float = 5.0):
    """
    Top-k vs 전체 interaction edge Cβ 거리 분포.

    모델별 패널 (4개):
      - 파란 실선:  Top-k 중 interaction 엣지 거리 분포 (고 Shapley)
      - 회색 점선:  전체 interaction 엣지 거리 분포 (baseline)
      - 주황 점선:  Bottom-k 중 interaction 엣지 거리 분포 (저 Shapley)
      - 빨간 수직선: graph 구성 cutoff (5Å)

    해석:
      Top-k 선이 왼쪽(단거리)으로 치우칠수록 모델이 화학적으로 가까운
      상호작용 엣지를 실제로 더 중요하게 봄 → XAI 결과의 화학적 타당성 지지
    """
    print("\n[fig7] 거리 분포 데이터 로드...")
    dist_data = load_distance_data(xai_dir, k=k)

    if dist_data is None:
        print("[fig7] distance_A 컬럼 없음 — extract_edge_distances.py 먼저 실행 필요. 건너뜀.")
        return

    # KDE 계산용
    try:
        from scipy.stats import gaussian_kde
        use_kde = True
    except ImportError:
        use_kde = False

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle(
        f"Interaction Edge Cβ-Distance Distribution\n"
        f"Top-{k} (high Shapley) vs. All interaction edges vs. Bottom-{k} (low Shapley)",
        fontsize=12, fontweight="bold",
    )

    for ax, model in zip(axes, MODELS):
        d = dist_data.get(model, {})
        all_d  = [v for v in d.get("all",  []) if not np.isnan(v)]
        topk_d = [v for v in d.get("topk", []) if not np.isnan(v)]
        botk_d = [v for v in d.get("botk", []) if not np.isnan(v)]

        if not all_d:
            ax.set_title(MODEL_SHORT[model] + "\n(데이터 없음)", fontsize=9)
            continue

        x_max = np.percentile(all_d, 99)
        x_min = 0
        xs = np.linspace(x_min, x_max, 300)

        def _plot_kde(vals, color, lw, ls, label, alpha_fill=0.0):
            if len(vals) < 5:
                return
            if use_kde:
                kde = gaussian_kde(vals, bw_method="scott")
                ys  = kde(xs)
                ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, label=label)
                if alpha_fill > 0:
                    ax.fill_between(xs, ys, alpha=alpha_fill, color=color)
            else:
                # KDE 없을 때 histogram으로 대체
                ax.hist(vals, bins=30, density=True, color=color, alpha=0.4,
                        histtype="stepfilled", label=label)

        _plot_kde(all_d,  "#888888", 1.5, "--", f"All interaction (n={len(all_d):,})")
        _plot_kde(botk_d, "#FF7F0E", 1.5, ":",  f"Bottom-{k} Shapley (n={len(botk_d):,})")
        _plot_kde(topk_d, "#1F77B4", 2.2, "-",  f"Top-{k} Shapley (n={len(topk_d):,})",
                  alpha_fill=0.12)

        # graph cutoff 수직선
        ax.axvline(threshold, color="red", linestyle="-.", linewidth=1.2,
                   label=f"Graph cutoff ({threshold}Å)")

        # 중앙값 표시
        for vals, color in [(topk_d, "#1F77B4"), (all_d, "#888888")]:
            if vals:
                med = np.median(vals)
                ax.axvline(med, color=color, linewidth=0.8, linestyle=":",
                           alpha=0.6)

        ax.set_xlabel("Cβ Distance (Å)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(MODEL_SHORT[model], fontsize=10, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right")

        # 통계 텍스트
        if topk_d and all_d:
            med_topk = np.median(topk_d)
            med_all  = np.median(all_d)
            ax.text(0.03, 0.97,
                    f"Median\nTop-k: {med_topk:.1f}Å\nAll: {med_all:.1f}Å",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                          "edgecolor": "gray", "alpha": 0.8})

    plt.tight_layout()
    path = os.path.join(output_dir, "fig7_distance_distribution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[fig7] {path}")


# ── Figure 8: Top-k 엣지 NetworkX 그래프 ─────────────────────────────────────

def _build_nx_graph(df):
    """
    Shapley CSV → NetworkX 그래프 + 노드 역할 딕셔너리 반환.

    노드 역할 추론:
      - "ligand"  엣지에 등장하는 노드 → 리간드
      - "protein" 엣지에 등장하는 노드 → 단백질
      - "interaction" 전용 노드 → ligand/protein 집합으로 귀속 시도 후 unknown
    """
    import networkx as nx

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(int(row["src"]), int(row["dst"]),
                   edge_type=str(row.get("type", "")),
                   abs_shapley=float(row["abs_shapley"]))

    ligand_nodes, protein_nodes = set(), set()
    for _, row in df.iterrows():
        src, dst = int(row["src"]), int(row["dst"])
        if row.get("type") == "ligand":
            ligand_nodes.update([src, dst])
        elif row.get("type") == "protein":
            protein_nodes.update([src, dst])

    return G, {"ligand": ligand_nodes, "protein": protein_nodes}


def fig8_topk_graph(xai_dir: str, output_dir: str,
                    k_values: list = None, n_max: int = 3):
    """
    복합체별 단백질-리간드 상호작용 그래프 (NetworkX spring_layout).

    저장 경로:
      output_dir/fig8_topk_graphs/{model}/{group}/{pdb_id}_topk{k}.png

    시각화 규칙:
      - 노드: 빨강 = 리간드 원자, 하늘색 = 단백질 잔기, 회색 = 미분류
      - Top-k 엣지: darkgreen(interaction) / darkred(ligand) / darkblue(protein), 두께 3
      - 나머지 엣지: lightgrey, 두께 1.5

    Args:
        xai_dir   : XAI holdout 시드 디렉터리 (model/group/pdb_id 구조)
        output_dir: 최상위 출력 디렉터리 (fig8_topk_graphs 하위 폴더 자동 생성)
        k_values  : Top-k 값 목록 (기본 [15])
        n_max     : 모델×그룹당 최대 복합체 수 (기본 3)
    """
    try:
        import networkx as nx
    except ImportError:
        print("[fig8] networkx 미설치 — pip install networkx 후 재실행. 건너뜀.")
        return

    if k_values is None:
        k_values = [15]

    EDGE_TYPE_COLORS = {
        "interaction": "darkgreen",
        "ligand":      "darkred",
        "protein":     "darkblue",
    }

    graph_dir = os.path.join(output_dir, "fig8_topk_graphs")
    total = 0

    for model in MODELS:
        for grp in GROUPS:
            grp_dir = os.path.join(xai_dir, model, grp)
            if not os.path.isdir(grp_dir):
                continue

            pdb_ids = sorted(os.listdir(grp_dir))[:n_max]
            for pdb_id in pdb_ids:
                csv_path = os.path.join(grp_dir, pdb_id, f"{pdb_id}_shapley.csv")
                if not os.path.exists(csv_path):
                    continue

                df = pd.read_csv(csv_path)
                if df.empty or "src" not in df.columns or "dst" not in df.columns:
                    continue

                try:
                    G, node_roles = _build_nx_graph(df)
                except Exception as e:
                    print(f"  [fig8] {pdb_id} 그래프 생성 실패: {e}")
                    continue

                if len(G.nodes) == 0:
                    continue

                # 레이아웃은 k 공통으로 한 번만 계산
                pos = nx.spring_layout(G, seed=42, k=0.5)

                # 노드 색상 결정
                node_list   = list(G.nodes())
                node_colors = []
                for n in node_list:
                    if n in node_roles["ligand"]:
                        node_colors.append("red")
                    elif n in node_roles["protein"]:
                        node_colors.append("lightblue")
                    else:
                        node_colors.append("lightgray")

                for k in k_values:
                    # Top-k 엣지 집합 (방향 무관)
                    df_s = df.sort_values("abs_shapley", ascending=False).reset_index(drop=True)
                    topk_rows = df_s.head(k)
                    topk_pairs = set(
                        zip(topk_rows["src"].astype(int), topk_rows["dst"].astype(int))
                    )
                    topk_pairs |= {(d, s) for s, d in topk_pairs}

                    topk_edge_list,   topk_edge_colors = [], []
                    other_edge_list = []

                    for u, v, data in G.edges(data=True):
                        if (u, v) in topk_pairs:
                            topk_edge_list.append((u, v))
                            topk_edge_colors.append(
                                EDGE_TYPE_COLORS.get(data.get("edge_type", ""), "black")
                            )
                        else:
                            other_edge_list.append((u, v))

                    fig, ax = plt.subplots(figsize=(8, 8))

                    # 그리기: 비-top-k 먼저 (뒤), top-k 나중 (앞), 노드 마지막
                    if other_edge_list:
                        nx.draw_networkx_edges(
                            G, pos, edgelist=other_edge_list,
                            edge_color="lightgrey", width=1.5, alpha=0.5, ax=ax)
                    if topk_edge_list:
                        nx.draw_networkx_edges(
                            G, pos, edgelist=topk_edge_list,
                            edge_color=topk_edge_colors, width=3.0, ax=ax)
                    nx.draw_networkx_nodes(
                        G, pos, nodelist=node_list,
                        node_color=node_colors, node_size=60, alpha=0.9, ax=ax)

                    ax.set_title(
                        f"{pdb_id}  —  {MODEL_SHORT[model].replace(chr(10), ' ')}  "
                        f"[{grp}]  |  Top-{k} edges highlighted",
                        fontsize=10, fontweight="bold")

                    legend_handles = [
                        mpatches.Patch(color="red",       label="Ligand node"),
                        mpatches.Patch(color="lightblue", label="Protein node"),
                        mpatches.Patch(color="lightgray", label="Unknown node"),
                        Line2D([0], [0], color="darkgreen", linewidth=2.5,
                               label="Interaction (top-k)"),
                        Line2D([0], [0], color="darkred",   linewidth=2.5,
                               label="Ligand (top-k)"),
                        Line2D([0], [0], color="darkblue",  linewidth=2.5,
                               label="Protein (top-k)"),
                        Line2D([0], [0], color="lightgrey", linewidth=1.5,
                               label="Non-top-k edge"),
                    ]
                    ax.legend(handles=legend_handles, loc="upper right",
                              fontsize=8, framealpha=0.85)
                    ax.axis("off")

                    out_subdir = os.path.join(graph_dir, model, grp)
                    os.makedirs(out_subdir, exist_ok=True)
                    out_path = os.path.join(out_subdir, f"{pdb_id}_topk{k}.png")
                    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
                    plt.close()
                    total += 1

    print(f"[fig8] {total}개 그래프 저장 → {graph_dir}")


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xai_dir",
        default=os.path.join(_ROOT, "results/pipeline/xai/holdout/seed42"),
        help="XAI holdout seed 디렉터리 (model/group/pdb_id 구조)")
    parser.add_argument("--stats_json",
        default=os.path.join(_ROOT, "results/stats/stat_tests.json"))
    parser.add_argument("--contact_json",
        default=os.path.join(_ROOT, "results/contact_validation/contact_validation_thr5.0.json"))
    parser.add_argument("--residue_json",
        default=os.path.join(_ROOT, "results/residue_analysis/residue_analysis.json"))
    parser.add_argument("--msens_csv",
        default=os.path.join(_ROOT, "results/m_sensitivity/m_sensitivity.csv"))
    parser.add_argument("--output_dir",
        default=os.path.join(_ROOT, "results/figures"))
    parser.add_argument("--k", type=int, default=15,
        help="대표 k 값 (기본 15)")
    parser.add_argument("--threshold", type=float, default=5.0,
        help="그래프 구성 cutoff 거리 (Å, 기본 5.0)")
    parser.add_argument("--topk_graph_k", default="15",
        help="fig8 NetworkX 그래프에 사용할 k 값 목록 (쉼표 구분, 기본 '15')")
    parser.add_argument("--topk_graph_n", type=int, default=3,
        help="fig8: 모델×그룹당 최대 복합체 수 (기본 3)")
    parser.add_argument("--skip_topk_graph", action="store_true",
        help="fig8 NetworkX 그래프 생성 건너뜀 (복합체 수가 많을 때)")
    args = parser.parse_args()

    global K_REP
    K_REP = args.k

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 55)
    print("  논문용 시각화 생성")
    print(f"  xai_dir : {args.xai_dir}")
    print(f"  output  : {args.output_dir}")
    print("=" * 55)

    # 데이터 로드
    print("\n[데이터 로드] XAI CSV → lift 값 계산...")
    lift_data    = load_lift_data(args.xai_dir)
    stats_json   = _load_json(args.stats_json)
    contact_json = _load_json(args.contact_json)
    residue_json = _load_json(args.residue_json)

    # 그림 생성
    print("\n[그림 생성]")
    fig1_lift_boxplot(lift_data, stats_json, args.output_dir)
    fig2_lift_topk_trend(lift_data, args.output_dir)
    fig3_model_comparison(lift_data, args.output_dir)
    fig4_m_sensitivity(args.msens_csv, args.output_dir)
    fig5_contact_validation(contact_json, args.output_dir)
    fig6_residue_concentration(residue_json, args.output_dir)
    fig7_distance_distribution(args.xai_dir, args.output_dir,
                               k=args.k, threshold=args.threshold)

    if not args.skip_topk_graph:
        topk_k_values = [int(v) for v in args.topk_graph_k.split(",")]
        fig8_topk_graph(args.xai_dir, args.output_dir,
                        k_values=topk_k_values, n_max=args.topk_graph_n)

    print(f"\n[완료] 그림 저장 위치: {args.output_dir}")
    print("  fig1_lift_boxplot.png          — 주요 결과 (논문 Figure 핵심)")
    print("  fig2_lift_topk_trend.png       — k 민감도 확인")
    print("  fig3_model_comparison.png      — 모델 간 비교")
    print("  fig4_m_sensitivity.png         — M=50 검증")
    print("  fig5_contact_validation.png    — 화학적 접촉 검증")
    print("  fig6_residue_concentration.png — 잔기 집중도")
    print("  fig7_distance_distribution.png — Top-k vs 전체 interaction edge 거리 분포")
    print("  fig8_topk_graphs/              — 복합체별 단백질-리간드 그래프 (NetworkX)")


if __name__ == "__main__":
    main()
