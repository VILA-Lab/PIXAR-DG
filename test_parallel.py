"""
test_parallel.py — Multi-GPU parallel evaluation for PIXAR model.

Splits the test set into N equal chunks and evaluates each chunk on a
separate GPU in parallel. Raw intermediate counts from each worker are
merged and all final metrics are recomputed exactly — identical to running
test.py on the full set.

Usage:
    python test_parallel.py \\
      --version /path/to/model \\
      --dataset_dir /path/to/dataset \\
      --vision_pretrained /path/to/sam.pth \\
      --gpus 2,3,4,5 \\
      --output_dir ./evaluation/logs/my_eval_parallel \\
      [--seg_prompt_mode fuse] [--precision bf16] [--save_generated_text]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

from utils.metrics import (
    MetricsAccumulator,
    GroupAccumulator,
    merge_raw_counts,
    compute_metrics,
    compute_group_metrics,
    compute_cross_table_metrics,
    print_metrics_report,
    print_group_report,
    metrics_for_json,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PIXAR Parallel Evaluation (Multi-GPU)"
    )
    parser.add_argument("--version", required=True, type=str,
                        help="Path to merged model (base + finetune weights)")
    parser.add_argument("--precision", default="fp16", type=str,
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--split", default="validation", type=str)
    parser.add_argument("--output_dir", default="./test_output_parallel", type=str)
    parser.add_argument("--workers", default=4, type=int)

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str,
                        choices=["llava_v1", "llava_llama_2"])

    parser.add_argument("--num_obj_classes", type=int, default=81)
    parser.add_argument("--obj_threshold", type=float, default=0.5)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_generated_text", action="store_true", default=False)
    parser.add_argument("--text_output_file", type=str, default="generated_texts.json")

    parser.add_argument("--seg_prompt_mode", type=str, default="fuse",
                        choices=["seg_only", "text_only", "fuse"])
    parser.add_argument("--generate_text_in_seg_only", action="store_true", default=False,
                        help="Generate text tokens even in seg_only mode (default: disabled)")

    # Parallel-specific
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU IDs, e.g. '2,3,4,5'")

    # Subset evaluation (optional; default=None preserves full-eval behavior)
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="If set, subsample dataset to this many samples using stratified "
             "sampling (real/tampered ratio preserved). Must be >= 2. "
             "Default: evaluate all samples (behavior unchanged).",
    )
    parser.add_argument(
        "--sample_seed", type=int, default=42,
        help="Random seed for dataset shuffling and stratified sampling. "
             "Default: 42 (behavior unchanged).",
    )

    parser.add_argument(
        "--mapping_json", type=str, default=None,
        help="Path to mapping.json for per-model/per-op breakdown. "
             "Default: auto-detected as pixar_0.05/mapping.json relative to "
             "--dataset_dir. Pass 'none' to disable.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers for per-model/per-op breakdown
# ---------------------------------------------------------------------------

def _find_mapping_json(dataset_dir: str) -> str | None:
    """
    Locate the mapping.json that describes <dataset_dir>.

    Order: (1) <dataset_dir>/mapping.json, else (2) walk up to a pixar_0.05/mapping.json.
    Without (1), a sibling pixar_0.05 mapping could be picked up silently (wrong generator set).
    """
    d = os.path.abspath(dataset_dir)
    direct = os.path.join(d, "mapping.json")
    if os.path.isfile(direct):
        return direct
    for _ in range(6):
        candidate = os.path.join(d, "pixar_0.05", "mapping.json")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:  # reached filesystem root
            break
        d = parent
    return None


def _parse_model_op(type_str: str) -> tuple[str, str]:
    """
    Parse a mapping.json 'type' string into (model, operation).

    Example: 'gemini3_coco_val_inter_replacement_1' → ('gemini3', 'inter_replacement_1')
    """
    parts = type_str.split("_coco_val_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return type_str, "unknown"



# ---------------------------------------------------------------------------
# Worker: runs in a spawned subprocess, one per GPU
# ---------------------------------------------------------------------------

def evaluate_worker(gpu, chunk_id, num_chunks, args, output_dir):
    """
    Load the model on `gpu`, evaluate indices [start, end), and save
    raw intermediate counts to output_dir/raw_chunk_{chunk_id}.json.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Local imports here so each spawned process initialises CUDA cleanly
    import tqdm
    import transformers
    from model.PIXAR import PIXARForCausalLM
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from utils.PIXAR_Set import CustomDataset
    from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

    print(f"[Chunk {chunk_id}] GPU {gpu}: loading tokenizer...", flush=True)

    # ---- Tokenizer ----
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # ---- Model ----
    torch_dtype = {"fp16": torch.half, "bf16": torch.bfloat16}.get(
        args.precision, torch.float32
    )
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "obj_token_idx": args.obj_token_idx,
        "num_obj_classes": args.num_obj_classes,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_prompt_mode": args.seg_prompt_mode,
    }
    model = PIXARForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=torch_dtype)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    model.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    print(f"[Chunk {chunk_id}] GPU {gpu}: model loaded.", flush=True)

    # ---- Dataset ----
    test_dataset = CustomDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        split=args.split,
        precision=args.precision,
        image_size=args.image_size,
    )

    # ---- Chunk index range ----
    import random
    all_indices = list(range(len(test_dataset)))
    random.seed(args.sample_seed)   # fixed seed → every worker gets the same shuffle
    random.shuffle(all_indices)

    # ---- Optional stratified subsampling (only when --max_samples is set) ----
    if args.max_samples is not None:
        N = len(all_indices)
        if args.max_samples < 2:
            raise ValueError(
                f"--max_samples must be >= 2 (got {args.max_samples}); "
                f"dataset has 2 non-empty classes (real, tampered)."
            )
        if args.max_samples < N:
            real_idx     = [i for i in all_indices if test_dataset.cls_labels[i] == 0]
            tampered_idx = [i for i in all_indices if test_dataset.cls_labels[i] == 2]
            N_r, N_t = len(real_idx), len(tampered_idx)
            # Proportional allocation; difference goes to tampered → exact sum == max_samples
            n_real     = round(args.max_samples * N_r / N)
            n_tampered = args.max_samples - n_real
            # Clamp to available
            n_real     = min(n_real,     N_r)
            n_tampered = min(n_tampered, N_t)
            # Backfill deficit to the other class when one class is exhausted
            deficit = args.max_samples - n_real - n_tampered
            if deficit > 0:
                extra_r    = min(deficit, N_r - n_real)
                n_real    += extra_r
                deficit   -= extra_r
                n_tampered = min(n_tampered + deficit, N_t)
            assert n_real > 0 and n_tampered > 0, (
                f"Stratified allocation gave empty class "
                f"(n_real={n_real}, n_tampered={n_tampered}). "
                f"Increase --max_samples."
            )
            all_indices = real_idx[:n_real] + tampered_idx[:n_tampered]
            random.seed(args.sample_seed)   # re-seed so re-shuffle is identical across workers
            random.shuffle(all_indices)
            print(
                f"[Chunk {chunk_id}] Subsampled: {len(all_indices)} samples "
                f"(real={n_real}/{N_r}, tampered={n_tampered}/{N_t}, seed={args.sample_seed})",
                flush=True,
            )

    chunk_size = (len(all_indices) + num_chunks - 1) // num_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, len(all_indices))
    indices = all_indices[start:end]
    print(
        f"[Chunk {chunk_id}] GPU {gpu}: indices {start}~{end-1} "
        f"({len(indices)}/{len(all_indices)} samples)",
        flush=True,
    )

    # ---- Default prompt ----
    default_prompt = (
        "Can you identify whether this image is real, fully synthetic, or tampered? "
        "If it is tampered, please (1) classify which object was modified and "
        "(2) output a mask for the modified regions."
    )

    # ---- Metric accumulators ----
    acc = MetricsAccumulator()

    # ---- Per-model / per-op accumulators (requires mapping_json) ----
    from collections import defaultdict
    mapping: dict = {}
    mapping_path = args.mapping_json
    if mapping_path is None:
        mapping_path = _find_mapping_json(args.dataset_dir)
    if mapping_path and mapping_path.lower() != "none" and os.path.isfile(mapping_path):
        with open(mapping_path) as _f:
            mapping = json.load(_f)
        print(f"[Chunk {chunk_id}] Loaded mapping.json ({len(mapping)} entries): {mapping_path}", flush=True)
    else:
        print(f"[Chunk {chunk_id}] No mapping.json found; skipping per-model/op breakdown.", flush=True)
    per_model: dict[str, GroupAccumulator] = defaultdict(GroupAccumulator)
    per_op:    dict[str, GroupAccumulator] = defaultdict(GroupAccumulator)
    per_model_per_op: dict[str, dict[str, GroupAccumulator]] = defaultdict(lambda: defaultdict(GroupAccumulator))

    # ---- Real-time text output file ----
    gt_path = os.path.join(output_dir, f"generated_texts_chunk_{chunk_id}.jsonl")
    gt_file = open(gt_path, "w", encoding="utf-8") if args.save_generated_text else None

    # ---- Evaluation loop ----
    for sample_idx in tqdm.tqdm(indices, desc=f"GPU{gpu} chunk{chunk_id}"):
        item = test_dataset[sample_idx]
        (image_path, image, image_clip, conversations, mask, soft_mask,
         labels, cls_labels, resize, _, _, _, has_text, obj_label_vec) = item

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + default_prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "[CLS] [OBJ] [SEG] ")
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        image_clip = image_clip.unsqueeze(0).cuda()
        image = image.unsqueeze(0).cuda()
        if args.precision == "fp16":
            image_clip = image_clip.half(); image = image.half()
        elif args.precision == "bf16":
            image_clip = image_clip.bfloat16(); image = image.bfloat16()

        resize_list = [resize]
        original_size_list = [labels.shape[-2:]]

        generate_text = (
            args.seg_prompt_mode != "seg_only"
            or args.generate_text_in_seg_only
        )
        with torch.no_grad():
            output_ids, pred_masks, obj_preds, cls_info = model.evaluate(
                image_clip, image, input_ids, resize_list, original_size_list,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
                cls_label=cls_labels,
                generate_text=generate_text,
            )

        # Decode text
        input_token_len = input_ids.shape[1]
        new_tokens = output_ids[0][input_token_len:]
        new_tokens = new_tokens[new_tokens != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(new_tokens, skip_special_tokens=False)
        text_output = text_output.replace("\n", " ").replace("  ", " ").strip()

        if cls_labels == 0:
            gt_text_description = ""
        elif cls_labels == 1:
            gt_text_description = ""
        else:
            conv_str = conversations[0]
            seg_marker = "[SEG] "
            seg_pos = conv_str.find(seg_marker)
            if seg_pos >= 0:
                gt_text_description = conv_str[seg_pos + len(seg_marker):].split("</s>")[0].strip()
                hardcoded_prefix = "The image is tampered."
                if gt_text_description.startswith(hardcoded_prefix):
                    remaining = gt_text_description[len(hardcoded_prefix):].strip()
                    gt_text_description = (
                        f"This image is tampered. {remaining}" if remaining else ""
                    )
            else:
                gt_text_description = ""

        if gt_file is not None:
            gt_file.write(json.dumps({
                "image_path": image_path,
                "generated_text": text_output,
                "gt_text_description": gt_text_description,
                "ground_truth_label": int(cls_labels),
                "predicted_class": cls_info["predicted_class"],
                "predicted_label": cls_info["label"],
            }, ensure_ascii=False) + "\n")
            gt_file.flush()

        # ------ Classification ------
        predicted_class = cls_info["predicted_class"]
        acc.update_cls(predicted_class, int(cls_labels))

        # ------ Segmentation (tampered only) ------
        if cls_labels == 2:
            gt_mask       = soft_mask.int().cuda()
            pred_mask_bin = (pred_masks[0] > 0).int().cuda()

            with torch.no_grad():
                pm = pred_masks[0].float().cuda()
                pred_scores = (
                    torch.sigmoid(pm) if (pm.min() < 0 or pm.max() > 1.0)
                    else pm.clamp(0, 1)
                )

            seg_result = acc.update_seg(pred_mask_bin, gt_mask, pred_scores)

            # Per-group IoU + pixel update (seg_result always non-None for tampered samples)
            if mapping and seg_result is not None:
                img_name = os.path.basename(image_path)
                if img_name in mapping:
                    model_name, op_name = _parse_model_op(mapping[img_name]["type"])
                    inter_np, union_np, acc_iou_np, seg_n, pix_TP_i, pix_FP_i, pix_FN_i = seg_result
                    is_tampered_pred = (predicted_class == 2)
                    per_model[model_name].update(
                        is_tampered_pred, inter_np, union_np, acc_iou_np, seg_n,
                        pix_TP_i, pix_FP_i, pix_FN_i,
                    )
                    per_op[op_name].update(
                        is_tampered_pred, inter_np, union_np, acc_iou_np, seg_n,
                        pix_TP_i, pix_FP_i, pix_FN_i,
                    )
                    per_model_per_op[model_name][op_name].update(
                        is_tampered_pred, inter_np, union_np, acc_iou_np, seg_n,
                        pix_TP_i, pix_FP_i, pix_FN_i,
                    )
                else:
                    print(f"[Chunk {chunk_id}] WARNING: {img_name} not in mapping.json; skipping group update.", flush=True)
        elif cls_labels == 2 and mapping:
            # Tampered sample but seg not run → still count for recall
            img_name = os.path.basename(image_path)
            if img_name in mapping:
                model_name, op_name = _parse_model_op(mapping[img_name]["type"])
                is_tampered_pred = (predicted_class == 2)
                per_model[model_name].update(is_tampered_pred)
                per_op[op_name].update(is_tampered_pred)
                per_model_per_op[model_name][op_name].update(is_tampered_pred)
            else:
                print(f"[Chunk {chunk_id}] WARNING: {img_name} not in mapping.json; skipping group update.", flush=True)

        # ------ OBJ (tampered only) ------
        if cls_labels == 2:
            probs_obj = obj_preds.unsqueeze(0) if obj_preds.dim() == 1 else obj_preds
            acc.update_obj(probs_obj.cuda(), obj_label_vec.cuda(), threshold=args.obj_threshold)

    # ---- Save raw counts ----
    raw = acc.to_dict()
    raw["per_model"] = {k: v.to_dict() for k, v in per_model.items()}
    raw["per_op"]    = {k: v.to_dict() for k, v in per_op.items()}
    raw["per_model_per_op"] = {
        m: {o: g.to_dict() for o, g in ops.items()}
        for m, ops in per_model_per_op.items()
    }
    raw_path = os.path.join(output_dir, f"raw_chunk_{chunk_id}.json")
    with open(raw_path, "w") as f:
        json.dump(raw, f)
    print(f"[Chunk {chunk_id}] Raw counts saved → {raw_path}", flush=True)

    if gt_file is not None:
        gt_file.close()
        print(f"[Chunk {chunk_id}] Generated texts saved → {gt_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",")]
    num_chunks = len(gpus)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Parallel evaluation: {num_chunks} chunks on GPUs {gpus}")
    print(f"Model:      {args.version}")
    print(f"Dataset:    {args.dataset_dir}")
    print(f"Output dir: {args.output_dir}")

    ctx = mp.get_context("spawn")
    procs = []
    for chunk_id, gpu in enumerate(gpus):
        p = ctx.Process(
            target=evaluate_worker,
            args=(gpu, chunk_id, num_chunks, args, args.output_dir),
        )
        p.start()
        procs.append(p)
        print(f"  Launched chunk {chunk_id} on GPU {gpu} (PID={p.pid})")

    print("Waiting for all chunks to finish...")
    failed = []
    for chunk_id, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            failed.append(chunk_id)
            print(
                f"  [ERROR] Chunk {chunk_id} (GPU {gpus[chunk_id]}) "
                f"failed with exitcode {p.exitcode}"
            )

    if failed:
        raise RuntimeError(
            f"Chunks {failed} failed. "
            f"Check {args.output_dir}/raw_chunk_*.json for which chunks completed."
        )

    # ---- Merge ----
    print("\nAll chunks done. Merging results...")
    raws = []
    for i in range(num_chunks):
        raw_path = os.path.join(args.output_dir, f"raw_chunk_{i}.json")
        with open(raw_path) as f:
            raws.append(json.load(f))

    merged  = merge_raw_counts(raws)
    metrics = compute_metrics(merged)
    print_metrics_report(metrics, metrics["total_samples"], num_chunks)

    # Per-model / per-op breakdown
    per_model_m = compute_group_metrics(merged.get("per_model", {}))
    per_op_m    = compute_group_metrics(merged.get("per_op",    {}))
    if per_model_m:
        print_group_report("Per-Model Breakdown (tampered samples)", per_model_m)
    if per_op_m:
        print_group_report("Per-Operation Breakdown (tampered samples)", per_op_m)

    # Optionally merge generated text files (JSONL per chunk → single JSON)
    if args.save_generated_text:
        all_texts = []
        for i in range(num_chunks):
            gt_path = os.path.join(args.output_dir, f"generated_texts_chunk_{i}.jsonl")
            if os.path.exists(gt_path):
                with open(gt_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_texts.append(json.loads(line))
        out_path = os.path.join(args.output_dir, args.text_output_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_texts, f, indent=2, ensure_ascii=False)
        print(f"Generated texts saved to: {out_path}")

    # Per-model × per-op cross-table
    cross_raw = merged.get("per_model_per_op", {})
    all_models = sorted(merged.get("per_model", {}).keys())
    all_ops    = sorted(merged.get("per_op",    {}).keys())
    cross_m = compute_cross_table_metrics(cross_raw, all_models, all_ops) if (all_models and all_ops) else {}

    # Save final metrics.json (includes per-group breakdown when available)
    save_metrics = metrics_for_json(metrics)
    if per_model_m:
        save_metrics["per_model_metrics"] = per_model_m
    if per_op_m:
        save_metrics["per_op_metrics"] = per_op_m
    if cross_m:
        save_metrics["per_model_per_op_metrics"] = cross_m
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

