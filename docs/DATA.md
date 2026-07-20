# Data

The training set is built on top of the public **PIXAR benchmark**; we do not
redistribute images. This guide covers the on-disk format and the construction
scripts under `preprocess/`.

## Dataset format

Every dataset (the PIXAR benchmark and all derived subsets) uses the same layout:

```
<dataset>/
├── train/
│   ├── real/            # authentic source images
│   ├── tampered/        # edited images
│   ├── masks/           # binary edit masks
│   ├── soft_masks/      # pixel-difference maps  M_τ = |I_orig − I_gen| > τ   (τ = 0.05)
│   ├── metadata/        # per-image JSON: {"cls": [...], "text": "..."}
│   └── full_synthetic/  # kept for format compatibility (may be empty)
├── validation/          # same structure
├── test/  -> validation # symlink
└── mapping.json
```

`mapping.json` maps each tampered filename to its record (comments below are
explanatory, not literal JSON):

```jsonc
{
  "tampered_<hash>.png": {
    "entry":  "<coco/source id>",          // uniqueness key for the train/val split
    "real":   "original_<id>.png",
    "type":   "<generator>_coco_<split>_<op>",
    "split":  "train" | "validation",
    "source": "qwen" | "gemini" | ...,     // generator
    "bg":     false
  }
}
```

Derived subsets symlink images back to the source dataset (no copies); only
`mapping.json` is rewritten.

## Build pipeline

All scripts default to `./data` and accept `--help`.

### 1. Re-split the PIXAR benchmark

```bash
python preprocess/split_pixar.py \
  --src data/PIXAR_preprocessed/test_full_0.05/full_0.05 \
  --dst data/pixar_0.05
```

Deterministic entry-level train/validation split (seed = 42).

### 2. Headline training set — 70K Qwen + 3,353 Gemini-2.5

```bash
python preprocess/make_qg_70k_subsets.py \
  --variant 3353x1 \
  --qwen_src   data/PIXAR_preprocessed/train_0.05/ours_0.05 \
  --gemini_src data/pixar_0.05 \
  --out_dir    data
```

Produces `data/pixar_qg_70k_3353x1` (73,353 train tampered). The Qwen pool is
selected round-robin across edit operations; entries overlapping the validation
split are removed (leak prevention). Validation is symlinked from `pixar_0.05`.

The same script builds the **base-source size variants** used in the data-scale
study — pass `--variant 30k_3353x1`, `150k_3353x1`, or `380k_3353x1` (each pairs
N Qwen samples with the same 3,353 Gemini-2.5 companion set).

## What you must supply

| Item | Source |
|:---|:---|
| PIXAR benchmark (Qwen train pool + multi-generator test) | public PIXAR release |
| SAM ViT-H weights | `sam_vit_h_4b8939.pth` |
| Base detector (PIXAR-7B / 13B) | HuggingFace |

Given the PIXAR benchmark, every dataset in the paper is reproducible from the
scripts above — no proprietary raw generator outputs are required for the main
Qwen + Gemini-2.5 training set.
