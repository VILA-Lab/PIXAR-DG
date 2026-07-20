#!/usr/bin/env bash
# Evaluate a merged model on a (multi-generator) PIXAR test set.
#
# Produces metrics.json with overall accuracy, pixel-level gIoU / cIoU / F1, and
# a per-generator breakdown (per_model_metrics). Per-generator tampered accuracy
# is per_model_metrics[<gen>].recall; per-generator localization is .giou / .ciou.
#
# Usage:
#   bash scripts/eval.sh --model outputs/merged/ours --dataset_dir data/<test_set>
#   bash scripts/eval.sh --model outputs/merged/ours --dataset_dir data/<test_set> --gpus 0,1
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
VISION_PRETRAINED="pretrains/sam_vit_h_4b8939.pth"

MODEL=""
DATASET_DIR=""
GPUS="0,1,2,3"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";       shift 2 ;;
        --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
        --gpus)        GPUS="$2";        shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        *) echo "[WARN] unknown arg: $1"; shift ;;
    esac
done

[ -n "${MODEL}" ] && [ -n "${DATASET_DIR}" ] || { echo "[ERROR] --model and --dataset_dir are required"; exit 1; }
[ -d "${MODEL}" ]              || { echo "[ERROR] model not found: ${MODEL}"; exit 1; }
[ -d "${DATASET_DIR}/test" ]  || { echo "[ERROR] test split not found: ${DATASET_DIR}/test"; exit 1; }
[ -n "${OUTPUT_DIR}" ] || OUTPUT_DIR="outputs/logs/eval_$(basename "${MODEL}")"

mkdir -p "${OUTPUT_DIR}"
unset CUDA_VISIBLE_DEVICES

echo "=== eval ${MODEL} on ${DATASET_DIR}/test -> ${OUTPUT_DIR} ==="
${PYTHON} test_parallel.py \
    --version           "${MODEL}" \
    --dataset_dir       "${DATASET_DIR}" \
    --vision_pretrained "${VISION_PRETRAINED}" \
    --gpus              "${GPUS}" \
    --output_dir        "${OUTPUT_DIR}" \
    --split             test \
    --seg_prompt_mode   fuse \
    --precision         bf16 \
    --obj_threshold     0.5 \
    --max_new_tokens    128 \
    --use_mm_start_end \
    --train_mask_decoder \
    --save_generated_text \
    --mapping_json      "${DATASET_DIR}/mapping.json" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo "[DONE] metrics -> ${OUTPUT_DIR}/metrics.json"
