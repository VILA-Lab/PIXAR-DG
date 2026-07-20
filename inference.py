#!/usr/bin/env python3
"""
Run the tampering detector on ordinary image paths and save per-example outputs.

Loads a merged checkpoint and, for each input image, saves the generated text
description, the CLS-head decision (real / tampered), the OBJ-head categories,
and the predicted localization mask.

The 7B and 13B models share one interface: point `--version` at a merged 7B or
13B checkpoint — no other change is needed.

Example:
    python inference.py \
        --version outputs/merged/ours_7b \
        --vision_pretrained pretrains/sam_vit_h_4b8939.pth \
        --seg_prompt_mode fuse \
        --image_paths path/to/image.png \
        --output_dir example_outputs
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.PIXAR import PIXARForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)


CLS_LABELS_3WAY = {
    0: "real",
    1: "fully synthetic",
    2: "tampered",
}

OBJ_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "background",
]

DEFAULT_PROMPT = (
    "Can you identify whether this image is real, fully synthetic, or tampered? "
    "If it is tampered, please (1) classify which object was modified and "
    "(2) output a mask for the modified regions."
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="PIXAR example pipeline for ordinary image path lists."
    )
    parser.add_argument("--version", required=True, type=str,
                        help="Path to the merged PIXAR model directory.")
    parser.add_argument("--vision_pretrained", required=True, type=str,
                        help="Path to SAM ViT-H checkpoint, e.g. sam_vit_h_4b8939.pth.")
    parser.add_argument("--image_paths", nargs="*", default=[],
                        help="One or more local image paths.")
    parser.add_argument("--image_list", default=None, type=str,
                        help="Text/JSON/JSONL file containing local image paths.")
    parser.add_argument("--output_dir", default="./example_outputs", type=str,
                        help="Directory where per-sample outputs will be saved.")

    parser.add_argument("--prompt", default=DEFAULT_PROMPT, type=str,
                        help="Prompt used for all images.")
    parser.add_argument("--precision", default="bf16", type=str,
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--num_obj_classes", default=81, type=int)
    parser.add_argument("--obj_threshold", default=0.5, type=float)
    parser.add_argument("--obj_top_k", default=5, type=int)
    parser.add_argument("--mask_threshold", default=0.5, type=float,
                        help="Threshold for saving the binary predicted mask.")
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--seg_prompt_mode", default="seg_only", type=str,
                        choices=["seg_only", "text_only", "fuse"])
    parser.add_argument("--conv_type", default="llava_v1", type=str,
                        choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--use_mm_start_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_mask_decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--copy_input", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy the original input file into each sample directory.")
    parser.add_argument("--copy_gt_mask", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy the ground-truth mask into each sample directory when it can be found.")
    parser.add_argument("--gt_mask_dir", default=None, type=str,
                        help="Optional directory containing GT masks named <image_stem>_mask.*.")
    return parser.parse_args(argv)


def load_image_paths(args):
    def item_to_path(item):
        if isinstance(item, dict):
            return item.get("image_path") or item.get("path") or item.get("file")
        return item

    paths = list(args.image_paths or [])
    if args.image_list:
        list_path = Path(args.image_list)
        suffix = list_path.suffix.lower()
        if suffix == ".json":
            data = json.loads(list_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("images") or data.get("image_paths") or []
            paths.extend(str(p) for p in (item_to_path(x) for x in data) if p)
        elif suffix == ".jsonl":
            with list_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    path = item_to_path(item)
                    if path:
                        paths.append(str(path))
        else:
            with list_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        paths.append(line)

    unique_paths = []
    seen = set()
    for p in paths:
        key = os.path.abspath(os.path.expanduser(p))
        if key not in seen:
            unique_paths.append(key)
            seen.add(key)
    if not unique_paths:
        raise ValueError("No images provided. Use --image_paths or --image_list.")
    return unique_paths


def preprocess_sam_image(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
):
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    return F.pad(x, (0, padw, 0, padh))


def dtype_from_precision(precision):
    if precision == "fp16":
        return torch.half
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def tensor_to_precision(tensor, precision):
    if precision == "fp16":
        return tensor.half()
    if precision == "bf16":
        return tensor.bfloat16()
    return tensor.float()


def build_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
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
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
            special_tokens=True,
        )
    return tokenizer


def build_model(args, tokenizer, device):
    torch_dtype = dtype_from_precision(args.precision)
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
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    quantized = args.load_in_8bit or args.load_in_4bit
    if quantized and not torch.cuda.is_available():
        raise RuntimeError("8-bit/4-bit loading requires CUDA.")
    if args.load_in_4bit:
        load_kwargs.update({
            "torch_dtype": torch.half,
            "device_map": {"": 0},
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        load_kwargs.update({
            "torch_dtype": torch.half,
            "device_map": {"": 0},
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual_model"],
            ),
        })

    model = PIXARForCausalLM.from_pretrained(
        args.version,
        **load_kwargs,
        **model_args,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=torch_dtype, device=device)
    model.resize_token_embeddings(len(tokenizer))

    if quantized:
        model.get_model().visual_model.to(dtype=torch_dtype, device=device)
    else:
        model = model.to(device=device, dtype=torch_dtype)

    model.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    return model


def make_prompt(args):
    conv = conversation_lib.default_conversation.copy()
    conv.messages = []
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + args.prompt
    if args.use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "[CLS] [OBJ] [SEG] ")
    return conv.get_prompt()


def decode_generated_text(tokenizer, output_ids, input_len):
    new_tokens = output_ids[0][input_len:]
    new_tokens = new_tokens[new_tokens != IMAGE_TOKEN_INDEX]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.replace("\n", " ").replace("  ", " ").strip()


def binary_cls_from_info(cls_info):
    raw_probs = cls_info.get("probabilities", {})
    real_prob = float(raw_probs.get("real", 0.0))
    full_prob = float(raw_probs.get("fully synthetic", 0.0))
    tampered_prob = float(raw_probs.get("tampered", 0.0))
    binary_probs = {
        "real": real_prob,
        "tampered": full_prob + tampered_prob,
    }

    raw_pred_class = int(cls_info["predicted_class"])
    raw_pred_label = CLS_LABELS_3WAY.get(raw_pred_class, cls_info.get("label", "unknown"))
    binary_pred_label = (
        "tampered"
        if binary_probs["tampered"] >= binary_probs["real"]
        else "real"
    )
    binary_pred_class = 2 if binary_pred_label == "tampered" else 0

    return {
        "predicted_class": binary_pred_class,
        "predicted_label": binary_pred_label,
        "probabilities": binary_probs,
        "raw_3way": {
            "predicted_class": raw_pred_class,
            "predicted_label": raw_pred_label,
            "probabilities": {
                "real": real_prob,
                "fully synthetic": full_prob,
                "tampered": tampered_prob,
            },
        },
    }


def obj_output_from_probs(obj_preds, threshold, top_k):
    probs = obj_preds.detach().float().cpu().flatten().numpy()
    names = OBJ_CLASS_NAMES[: len(probs)]
    thresholded = [
        {"index": i, "name": names[i] if i < len(names) else str(i), "probability": float(p)}
        for i, p in enumerate(probs)
        if p >= threshold
    ]
    order = np.argsort(-probs)
    top_k = min(top_k, len(order))
    top = [
        {
            "index": int(i),
            "name": names[int(i)] if int(i) < len(names) else str(int(i)),
            "probability": float(probs[int(i)]),
        }
        for i in order[:top_k]
    ]
    raw = {
        names[i] if i < len(names) else str(i): float(probs[i])
        for i in range(len(probs))
    }
    return {
        "threshold": threshold,
        "top_k": top,
        "thresholded": thresholded,
        "raw_probabilities": raw,
    }


def prepare_image_tensors(image_path, clip_processor, transform, args, device):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]

    image_clip = clip_processor.preprocess(image_rgb, return_tensors="pt")["pixel_values"][0]
    image_clip = tensor_to_precision(image_clip.unsqueeze(0).to(device), args.precision)

    sam_image = transform.apply_image(image_rgb)
    resize = sam_image.shape[:2]
    sam_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
    sam_tensor = preprocess_sam_image(sam_tensor, img_size=args.image_size).unsqueeze(0)
    sam_tensor = tensor_to_precision(sam_tensor.to(device), args.precision)

    return image_bgr, image_rgb, image_clip, sam_tensor, resize, original_size


def save_localization(sample_dir, image_rgb, pred_masks, mask_threshold):
    if not pred_masks:
        raise RuntimeError("No localization mask returned by model.evaluate().")

    mask_logits = pred_masks[0].detach().float().cpu()
    if mask_logits.dim() == 3:
        mask_logits = mask_logits[0]
    mask_np = mask_logits.numpy()
    if mask_np.min() < 0.0 or mask_np.max() > 1.0:
        prob = torch.sigmoid(mask_logits).numpy()
    else:
        prob = np.clip(mask_np, 0.0, 1.0)
    binary = prob >= mask_threshold

    prob_path = sample_dir / "predicted_mask_prob.npy"
    mask_path = sample_dir / "predicted_mask.png"
    overlay_path = sample_dir / "overlay.png"

    np.save(prob_path, prob.astype(np.float32))
    cv2.imwrite(str(mask_path), (binary.astype(np.uint8) * 255))

    overlay = image_rgb.copy().astype(np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)
    alpha = 0.45
    overlay[binary] = overlay[binary] * (1.0 - alpha) + red * alpha
    overlay_bgr = cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    return {
        "mask_threshold": mask_threshold,
        "mask_png": str(mask_path),
        "mask_prob_npy": str(prob_path),
        "overlay_png": str(overlay_path),
        "shape": list(prob.shape),
        "positive_pixel_fraction": float(binary.mean()),
    }


def safe_sample_dir(output_dir, index, image_path):
    stem = Path(image_path).stem
    digest = hashlib.sha1(os.path.abspath(image_path).encode("utf-8")).hexdigest()[:8]
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:80]
    return output_dir / f"sample_{index:04d}_{safe_stem}_{digest}"


def find_gt_mask_path(image_path, args):
    image_path = Path(image_path)
    stem = image_path.stem
    suffixes = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    candidates = []

    if args.gt_mask_dir:
        mask_dir = Path(args.gt_mask_dir)
        candidates.extend(mask_dir / f"{stem}_mask{s}" for s in suffixes)
        candidates.extend(mask_dir / f"{stem}{s}" for s in suffixes)
    else:
        # Common PIXAR layout: <dataset>/<split>/tampered/foo.png
        # with GT masks at <dataset>/<split>/masks/foo_mask.png.
        split_dir = image_path.parent.parent
        if image_path.parent.name in {"tampered", "real", "full_synthetic"}:
            mask_dir = split_dir / "masks"
            candidates.extend(mask_dir / f"{stem}_mask{s}" for s in suffixes)
            candidates.extend(mask_dir / f"{stem}{s}" for s in suffixes)

        local_mask_dir = image_path.parent / "masks"
        candidates.extend(local_mask_dir / f"{stem}_mask{s}" for s in suffixes)
        candidates.extend(image_path.with_name(f"{stem}_mask{s}") for s in suffixes)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def copy_gt_mask(sample_dir, image_path, args):
    if not args.copy_gt_mask:
        return None

    gt_mask = find_gt_mask_path(image_path, args)
    if gt_mask is None:
        print(f"  WARNING: GT mask not found for {image_path}", file=sys.stderr, flush=True)
        return None

    output_path = sample_dir / "gt_mask.png"
    shutil.copy2(gt_mask, output_path)
    return {
        "source_mask": str(gt_mask),
        "gt_mask_png": str(output_path),
    }


def run_one_image(index, image_path, args, tokenizer, model, clip_processor, transform, device):
    sample_dir = safe_sample_dir(Path(args.output_dir), index, image_path)
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_bgr, image_rgb, image_clip, sam_tensor, resize, original_size = prepare_image_tensors(
        image_path, clip_processor, transform, args, device
    )

    input_copy_path = sample_dir / "input.png"
    cv2.imwrite(str(input_copy_path), image_bgr)
    if args.copy_input:
        original_copy = sample_dir / ("source" + Path(image_path).suffix.lower())
        try:
            shutil.copy2(image_path, original_copy)
        except OSError:
            original_copy = None
    else:
        original_copy = None
    gt_mask_copy = copy_gt_mask(sample_dir, image_path, args)

    full_prompt = make_prompt(args)
    input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output_ids, pred_masks, obj_preds, cls_info = model.evaluate(
            image_clip,
            sam_tensor,
            input_ids,
            [resize],
            [original_size],
            max_new_tokens=args.max_new_tokens,
            tokenizer=tokenizer,
            cls_label=2,
            generate_text=True,
        )

    text_output = decode_generated_text(tokenizer, output_ids, input_ids.shape[1])
    (sample_dir / "text.txt").write_text(text_output + "\n", encoding="utf-8")

    cls_output = binary_cls_from_info(cls_info)
    obj_output = obj_output_from_probs(obj_preds, args.obj_threshold, args.obj_top_k)
    localization = save_localization(sample_dir, image_rgb, pred_masks, args.mask_threshold)

    result = {
        "sample_index": index,
        "image_path": image_path,
        "input_png": str(input_copy_path),
        "source_copy": str(original_copy) if original_copy else None,
        "gt_mask": gt_mask_copy,
        "assumed_ground_truth_label": "tampered",
        "text": text_output,
        "cls_head": cls_output,
        "obj_head": obj_output,
        "localization": localization,
        "settings": {
            "seg_prompt_mode": args.seg_prompt_mode,
            "max_new_tokens": args.max_new_tokens,
            "obj_threshold": args.obj_threshold,
            "obj_top_k": args.obj_top_k,
            "mask_threshold": args.mask_threshold,
            "copy_gt_mask": args.copy_gt_mask,
            "gt_mask_dir": args.gt_mask_dir,
        },
    }

    result_path = sample_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    cls_path = sample_dir / "cls_head.json"
    obj_path = sample_dir / "obj_head.json"
    cls_path.write_text(json.dumps(cls_output, indent=2, ensure_ascii=False), encoding="utf-8")
    obj_path.write_text(json.dumps(obj_output, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


def write_summary(output_dir, results):
    output_dir = Path(output_dir)
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "image_path",
                "predicted_label",
                "p_real",
                "p_tampered",
                "top_obj",
                "top_obj_prob",
                "result_json",
                "overlay_png",
                "mask_png",
                "mask_prob_npy",
            ],
        )
        writer.writeheader()
        for r in results:
            top_obj = r["obj_head"]["top_k"][0] if r["obj_head"]["top_k"] else {}
            writer.writerow({
                "sample_index": r["sample_index"],
                "image_path": r["image_path"],
                "predicted_label": r["cls_head"]["predicted_label"],
                "p_real": r["cls_head"]["probabilities"]["real"],
                "p_tampered": r["cls_head"]["probabilities"]["tampered"],
                "top_obj": top_obj.get("name", ""),
                "top_obj_prob": top_obj.get("probability", ""),
                "result_json": str(safe_sample_dir(output_dir, r["sample_index"], r["image_path"]) / "result.json"),
                "overlay_png": r["localization"]["overlay_png"],
                "mask_png": r["localization"]["mask_png"],
                "mask_prob_npy": r["localization"]["mask_prob_npy"],
            })


def main(argv):
    args = parse_args(argv)
    image_paths = load_image_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(args)
    model = build_model(args, tokenizer, device)
    clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    results = []
    errors = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path}", flush=True)
        try:
            result = run_one_image(
                index,
                image_path,
                args,
                tokenizer,
                model,
                clip_processor,
                transform,
                device,
            )
            results.append(result)
        except Exception as exc:
            error = {"sample_index": index, "image_path": image_path, "error": str(exc)}
            errors.append(error)
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)

    write_summary(output_dir, results)
    if errors:
        (output_dir / "errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nSaved {len(results)} result(s) to {output_dir}")
    if errors:
        print(f"{len(errors)} image(s) failed; see {output_dir / 'errors.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
