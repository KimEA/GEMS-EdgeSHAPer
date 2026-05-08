#!/bin/bash
# =============================================================================
# Step A: GC_GNN Cβ 재학습 (seed42) + 전체 모델 XAI (seed42)
# =============================================================================
# 실행 순서:
#   1. GC_GNN seed42 재학습 (Cβ edge_weight 적용)
#   2. 4개 모델 XAI (GEMS seed42 체크포인트 재사용, GC_GNN은 방금 재학습된 것)
#
# 사용법:
#   bash scripts/run_gcn_seed42.sh
# =============================================================================

set -e  # 에러 시 즉시 종료

# ── 경로 설정 (서버 환경에 맞게 수정) ─────────────────────────────────────────
DATA_DIR="/workspace/GEMS_datasets"
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # GEMS-XAI 루트
OUTPUT_DIR="${WORK_DIR}/results/pipeline"
SPLIT_JSON="${OUTPUT_DIR}/id_split.json"

CLEANSPLIT_AEPL="${DATA_DIR}/B6AEPL_train_cleansplit.pt"
PDBBIND_AEPL="${DATA_DIR}/B6AEPL_train_pdbbind.pt"
CASF_AEPL="${DATA_DIR}/B6AEPL_casf2016.pt"
CASF_AEPL_INDEP="${DATA_DIR}/B6AEPL_casf2016_indep.pt"
CLEANSPLIT_AE0L="${DATA_DIR}/B6AE0L_train_cleansplit.pt"
PDBBIND_AE0L="${DATA_DIR}/B6AE0L_train_pdbbind.pt"
CASF_AE0L="${DATA_DIR}/B6AE0L_casf2016.pt"
CASF_AE0L_INDEP="${DATA_DIR}/B6AE0L_casf2016_indep.pt"

M=50
MAX_PER_GROUP=100
NUM_WORKERS=4

# ── 환경 확인 ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Step A: GC_GNN 재학습 (Cβ, seed=42) + XAI"
echo "  작업 디렉터리: ${WORK_DIR}"
echo "  출력 디렉터리: ${OUTPUT_DIR}"
echo "============================================================"

cd "${WORK_DIR}"

# ── 0. M sensitivity 검증 (GEMS seed42 체크포인트 사전 필요) ──────────────────
echo ""
echo "------------------------------------------------------------"
echo "[0/3] M sensitivity 검증 (M=50 vs M=100, 10개 샘플, ~1시간)"
echo "  ★ 이 결과로 M=50 사용 가능 여부 판단 후 계속 진행"
echo "------------------------------------------------------------"
python analysis/m_sensitivity.py \
    --dataset    "${CLEANSPLIT_AEPL}" \
    --ckpt       "${OUTPUT_DIR}/checkpoints/gems_b6aepl_cleansplit_seed42_best.pt" \
    --split_json "${SPLIT_JSON}" \
    --n_samples  10 \
    --output_dir "${OUTPUT_DIR}/../m_sensitivity"

echo ""
echo "  [확인] 위 판정 결과를 보고 M=50이 사용 가능하면 계속 진행하세요."
echo "  [중단] M=100 권장이면 아래 주석에서 M 값을 변경하세요."
echo ""

# ── 1. GC_GNN seed42 재학습 ───────────────────────────────────────────────────
echo ""
echo "------------------------------------------------------------"
echo "[1/2] GC_GNN Cβ 재학습 (seed=42)"
echo "------------------------------------------------------------"
python retrain_gcn.py \
    --cleansplit_b6aepl "${CLEANSPLIT_AEPL}" \
    --pdbbind_b6aepl   "${PDBBIND_AEPL}" \
    --split_json       "${SPLIT_JSON}" \
    --output_dir       "${OUTPUT_DIR}" \
    --seed 42

echo ""
echo "[완료] GC_GNN seed42 재학습 완료"
echo "  체크포인트: ${OUTPUT_DIR}/checkpoints/gcn_cleansplit_seed42_best.pt"
echo "  체크포인트: ${OUTPUT_DIR}/checkpoints/gcn_pdbbind_seed42_best.pt"

# ── 2. 전체 모델 XAI (seed42, 학습 스킵) ────────────────────────────────────
echo ""
echo "------------------------------------------------------------"
echo "[2/2] 4개 모델 XAI 실행 (seed=42, M=${M})"
echo "------------------------------------------------------------"
python run_pipeline.py \
    --cleansplit_b6aepl     "${CLEANSPLIT_AEPL}" \
    --cleansplit_b6ae0l     "${CLEANSPLIT_AE0L}" \
    --pdbbind_b6aepl        "${PDBBIND_AEPL}" \
    --pdbbind_b6ae0l        "${PDBBIND_AE0L}" \
    --casf2016_b6aepl       "${CASF_AEPL}" \
    --casf2016_b6aepl_indep "${CASF_AEPL_INDEP}" \
    --casf2016_b6ae0l       "${CASF_AE0L}" \
    --casf2016_b6ae0l_indep "${CASF_AE0L_INDEP}" \
    --split_json            "${SPLIT_JSON}" \
    --output_dir            "${OUTPUT_DIR}" \
    --seeds 42 \
    --skip_train \
    --M             ${M} \
    --max_per_group ${MAX_PER_GROUP} \
    --num_workers   ${NUM_WORKERS}

echo ""
echo "------------------------------------------------------------"
echo "[3/3] 엣지 거리 정보 추가 (contact_validation 전처리)"
echo "------------------------------------------------------------"
python analysis/extract_edge_distances.py \
    --xai_dir    "${OUTPUT_DIR}/xai/holdout" \
    --split_json "${SPLIT_JSON}" \
    --dataset    "${CLEANSPLIT_AEPL}"

echo ""
echo "============================================================"
echo "  Step A 완료!"
echo "  XAI 결과: ${OUTPUT_DIR}/xai/holdout/seed42/"
echo "  거리 정보 추가 완료 (contact_validation 사용 가능)"
echo "============================================================"
