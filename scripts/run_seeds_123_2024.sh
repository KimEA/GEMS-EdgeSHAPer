#!/bin/bash
# =============================================================================
# Step B: 추가 시드 학습 + XAI (seed=123, 2024) — 백그라운드 실행용
# =============================================================================
# 실행 순서:
#   1. 6개 모델 학습 (GEMS×4 + GC_GNN×2) × 2 시드 (123, 2024)
#   2. 4개 모델(GEMS_B6AEPL×2, GC_GNN×2) XAI — 시드별 실행 후 집계
#
# 전제 조건:
#   - results/pipeline/id_split.json 존재 (seed42 파이프라인 실행 후)
#   - Step A (run_gcn_seed42.sh) 완료 불필요 — 독립 실행 가능
#
# 사용법:
#   nohup bash scripts/run_seeds_123_2024.sh > logs/seeds_123_2024.log 2>&1 &
#   echo "PID: $!"
# =============================================================================

set -e

# ── 경로 설정 (서버 환경에 맞게 수정) ─────────────────────────────────────────
DATA_DIR="/workspace/GEMS_datasets"
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${WORK_DIR}/results/pipeline_seeds"   # seed42와 별도 디렉터리
SPLIT_JSON="${WORK_DIR}/results/pipeline/id_split.json"   # 기존 분할 재사용

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

# ── 시작 ──────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Step B: 추가 시드 학습 + XAI (seed=123, 2024)"
echo "  시작 시각: $(date)"
echo "  작업 디렉터리: ${WORK_DIR}"
echo "  출력 디렉터리: ${OUTPUT_DIR}"
echo "  데이터 분할: ${SPLIT_JSON}"
echo "============================================================"

# 분할 파일 존재 확인
if [ ! -f "${SPLIT_JSON}" ]; then
    echo "[오류] id_split.json 없음: ${SPLIT_JSON}"
    echo "  먼저 run_gcn_seed42.sh 또는 seed42 파이프라인을 실행하세요."
    exit 1
fi

cd "${WORK_DIR}"
mkdir -p logs

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
    --seeds 123,2024 \
    --M             ${M} \
    --max_per_group ${MAX_PER_GROUP} \
    --num_workers   ${NUM_WORKERS} \
    --gems_epochs   2000 \
    --gems_patience 100 \
    --gcn_epochs    100 \
    --gcn_patience  50

echo ""
echo "------------------------------------------------------------"
echo "[2/2] 엣지 거리 정보 추가 (contact_validation 전처리)"
echo "------------------------------------------------------------"
python analysis/extract_edge_distances.py \
    --xai_dir    "${OUTPUT_DIR}/xai/holdout" \
    --split_json "${SPLIT_JSON}" \
    --dataset    "${CLEANSPLIT_AEPL}"

echo ""
echo "============================================================"
echo "  Step B 완료!"
echo "  종료 시각: $(date)"
echo "  XAI 결과: ${OUTPUT_DIR}/xai/holdout/"
echo "    seed123  → ${OUTPUT_DIR}/xai/holdout/seed123/"
echo "    seed2024 → ${OUTPUT_DIR}/xai/holdout/seed2024/"
echo ""
echo "  이후 로컬에서 분석 스크립트 실행:"
echo "    python analysis/run_stats.py --xai_dir ${OUTPUT_DIR}/xai/holdout"
echo "============================================================"
