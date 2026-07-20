#!/usr/bin/env bash
# Merge a DeepSpeed ZeRO checkpoint + LoRA adapter into a standalone
# HuggingFace model that can be evaluated or shared.
#
#   1. zero_to_fp32.py : ZeRO shards -> a single fp32 pytorch_model.bin
#   2. merge_lora_weights_and_save_hf_model.py : fold LoRA into the base model
#   3. clean up the ZeRO + fp32 intermediates (unless --keep_raw)
#
# Usage:
#   bash scripts/merge.sh --exp_name ours --base_model pretrains/PIXAR-7B
#   bash scripts/merge.sh --exp_name ours --base_model pretrains/PIXAR-7B \
#                         --ckpt_path outputs/runs/ours/final_checkpoint --keep_raw
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
VISION_PRETRAINED="pretrains/sam_vit_h_4b8939.pth"

EXP_NAME=""
BASE_MODEL=""
KEEP_RAW=0
CKPT_OVERRIDE=""
RUNS_DIR="outputs/runs"
FP32_DIR="outputs/fp32"
MERGED_DIR="outputs/merged"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)             EXP_NAME="$2";      shift 2 ;;
        --base_model)           BASE_MODEL="$2";    shift 2 ;;
        --keep_raw)             KEEP_RAW=1;         shift   ;;
        --ckpt_path|--ckpt_dir) CKPT_OVERRIDE="$2"; shift 2 ;;
        *) echo "[WARN] unknown arg: $1"; shift ;;
    esac
done

[ -n "${EXP_NAME}" ] && [ -n "${BASE_MODEL}" ] || { echo "[ERROR] --exp_name and --base_model are required"; exit 1; }

if [ -n "${CKPT_OVERRIDE}" ]; then
    CKPT_DIR="${CKPT_OVERRIDE}"
else
    CKPT_DIR="${RUNS_DIR}/${EXP_NAME}/final_checkpoint"
fi
[ -d "${CKPT_DIR}" ]                    || { echo "[ERROR] checkpoint not found: ${CKPT_DIR}"; exit 1; }
[ -f "${CKPT_DIR}/zero_to_fp32.py" ]   || { echo "[ERROR] zero_to_fp32.py not in ${CKPT_DIR}"; exit 1; }

FP32="${FP32_DIR}/${EXP_NAME}"
SAVE_PATH="${MERGED_DIR}/${EXP_NAME}"

echo "=== merge ${EXP_NAME} : ${CKPT_DIR} + ${BASE_MODEL} -> ${SAVE_PATH} ==="

mkdir -p "${FP32}"
${PYTHON} "${CKPT_DIR}/zero_to_fp32.py" "${CKPT_DIR}" "${FP32}"
if   [ -f "${FP32}/pytorch_model.bin" ];            then WEIGHT="${FP32}/pytorch_model.bin"
elif [ -f "${FP32}/pytorch_model.bin.index.json" ]; then WEIGHT="${FP32}"
else echo "[ERROR] zero_to_fp32 produced no weights in ${FP32}"; ls "${FP32}"; exit 1; fi

mkdir -p "${SAVE_PATH}"
${PYTHON} merge_lora_weights_and_save_hf_model.py \
    --version           "${BASE_MODEL}" \
    --weight            "${WEIGHT}" \
    --save_path         "${SAVE_PATH}" \
    --vision_pretrained "${VISION_PRETRAINED}" \
    --precision         bf16 \
    --use_mm_start_end \
    --train_mask_decoder

[ -f "${SAVE_PATH}/config.json" ] || { echo "[ERROR] merge failed (no config.json)"; exit 1; }

if [ "${KEEP_RAW}" -eq 0 ]; then
    find "${FP32}" -mindepth 1 -delete 2>/dev/null || true
    rmdir "${FP32}" 2>/dev/null || true
    echo "[cleanup] removed fp32 intermediate (pass --keep_raw to retain ZeRO/fp32 artifacts)"
fi

echo "[DONE] merged model -> ${SAVE_PATH}"
