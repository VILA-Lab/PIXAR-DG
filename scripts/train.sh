#!/usr/bin/env bash
#   Base model : pretrains/PIXAR-7B
#   Train data : data/pixar_qg_70k_3353x1   (73,353 tampered = 70,000 Qwen-Image
#                + 3,353 Gemini-2.5; built by preprocess/make_qg_70k_subsets.py)
#
# After training, merge the DeepSpeed checkpoint and evaluate:
#   bash scripts/merge.sh --exp_name ours --base_model pretrains/PIXAR-7B
#   bash scripts/eval.sh  --model outputs/merged/ours --dataset_dir data/<test_set>
#
# Usage:
#   conda activate pixar
#   bash scripts/train.sh
#   bash scripts/train.sh --gpus 4,5 --port 12771
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="pretrains/PIXAR-7B"
DATASET_DIR="data/pixar_qg_70k_3353x1"
VISION_PRETRAINED="pretrains/sam_vit_h_4b8939.pth"
LOG_BASE_DIR="outputs/runs"
EXP_NAME="ours"

GPUS="0,1"
PORT=12770
LR="2e-5"
EPOCHS=4
STEPS=2500
REAL_RATIO=0.5
SEED=0
TARGET_EFFECTIVE_BATCH=8
SCHEDULE='{"0-1":{"gemini":0},"1-end":{"gemini":2}}'

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)     VERSION="$2";     shift 2 ;;
        --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
        --exp_name)    EXP_NAME="$2";    shift 2 ;;
        --gpus)        GPUS="$2";        shift 2 ;;
        --port)        PORT="$2";        shift 2 ;;
        --lr)          LR="$2";          shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --steps)       STEPS="$2";       shift 2 ;;
        --seed)        SEED="$2";        shift 2 ;;
        --real_ratio)  REAL_RATIO="$2";  shift 2 ;;
        --schedule)    SCHEDULE="$2";    shift 2 ;;
        *) echo "[WARN] unknown arg: $1"; shift ;;
    esac
done

NUM_GPUS=$(awk -F',' '{print NF}' <<<"${GPUS}")
BATCH=$(( TARGET_EFFECTIVE_BATCH / NUM_GPUS )); [ "${BATCH}" -lt 2 ] && BATCH=2

[ -d "${VERSION}" ]           || { echo "[ERROR] base model missing: ${VERSION}"; exit 1; }
[ -d "${DATASET_DIR}/train" ] || { echo "[ERROR] dataset missing: ${DATASET_DIR}/train"; exit 1; }
[ -f "${VISION_PRETRAINED}" ] || { echo "[ERROR] SAM weights missing: ${VISION_PRETRAINED}"; exit 1; }

# Distributed defaults — override via the environment if your cluster differs.
unset CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p "${LOG_BASE_DIR}"

echo "=================================================================="
echo "  Train ${EXP_NAME}"
echo "    init   : ${VERSION}"
echo "    data   : ${DATASET_DIR}"
echo "    shape  : ${EPOCHS} ep x ${STEPS} steps  (batch ${BATCH} x ${NUM_GPUS} GPU)"
echo "    lr     : ${LR}  (constant)   real_ratio: ${REAL_RATIO}"
echo "    LI     : ${SCHEDULE}"
echo "=================================================================="

deepspeed --include "localhost:${GPUS}" --master_port "${PORT}" train_PIXAR.py \
    --version                 "${VERSION}" \
    --dataset_dir             "${DATASET_DIR}" \
    --vision_pretrained       "${VISION_PRETRAINED}" \
    --val_dataset             "${DATASET_DIR}" \
    --batch_size              "${BATCH}" \
    --grad_accumulation_steps 1 \
    --epochs                  "${EPOCHS}" \
    --steps_per_epoch         "${STEPS}" \
    --lr                      "${LR}" \
    --scheduler_type          constant \
    --dice_loss_weight        1.0 \
    --obj_loss_weight         0.5 \
    --text_loss_weight        3.0 \
    --seg_prompt_mode         fuse \
    --mask_type               ours \
    --precision               bf16 \
    --exp_name                "${EXP_NAME}" \
    --log_base_dir            "${LOG_BASE_DIR}" \
    --balance_training \
    --real_ratio              "${REAL_RATIO}" \
    --source_weights_schedule "${SCHEDULE}" \
    --train_seed              "${SEED}" \
    --no_eval \
    --no_auto_resume \
    --use_mm_start_end \
    --train_mask_decoder

echo ""
echo "[DONE] checkpoint -> ${LOG_BASE_DIR}/${EXP_NAME}/final_checkpoint"
echo "       next: bash scripts/merge.sh --exp_name ${EXP_NAME} --base_model ${VERSION}"
