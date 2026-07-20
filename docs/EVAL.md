# Evaluation

```bash
bash scripts/eval.sh --model outputs/merged/ours_7b --dataset_dir data/pixar_0.05
```

This runs `test_parallel.py` across the GPUs in `--gpus` (default `0,1,2,3`),
splitting the test split into shards and merging the per-shard results into a
single `metrics.json` in the output directory.

## Which test set

`data/pixar_0.05` (built by `preprocess/split_pixar.py`, with `test/` symlinked
to `validation/`) is the directly reproducible multi-generator test set. The
paper's headline OOD results use the four held-out generators **GPT-Image-2.0,
Gemini-3.1, FLUX.2, Seedream 4.5** from the PIXAR benchmark release; **Qwen-Image**
and **Gemini-2.5** are the in-domain generators. Evaluate on whichever PIXAR test
split contains these generators and read the per-generator breakdown below.

## metrics.json

**Top-level fields are global** — computed over *all* samples of *all* generators
in the test set. They are **not** the per-table numbers (the paper tables average
over the four OOD generators only — see below).

| Key | Meaning |
|:---|:---|
| `accuracy` | global classification accuracy (%) |
| `pixel_recall`, `pixel_f1` | global pixel-level recall / F1 |
| `giou`, `ciou` | `giou` = mean per-image IoU; `ciou` = cumulative (dataset-level) IoU |
| `per_model_metrics` | per-generator breakdown (see below) |
| `per_op_metrics` | per edit-operation breakdown |
| `total_samples`, `n_gt_real`, `n_gt_tampered` | counts |

`per_model_metrics[<generator>]`:

| Field | Meaning |
|:---|:---|
| `recall` | per-generator **binary tampered accuracy** (× 100 for %) — the *Binary Acc.* table column |
| `giou`, `ciou` | per-generator localization — the *gIoU* / *cIoU* columns |
| `pixel_f1` | per-generator pixel F1 |
| `n`, `correct` | sample count, correct count |

## Reproducing the paper tables

A table cell for a metric `m` is the **mean over the four OOD generators** of the
corresponding `per_model_metrics[<gen>].<field>`:

- **Binary Acc.** → mean of `recall` over {GPT-Image-2.0, Gemini-3.1, FLUX.2, Seedream 4.5}.
- **gIoU / cIoU** → mean of `giou` / `ciou` over the same four.
- **Pixel Recall / Pixel F1** (Table 1) are pixel-level quantities — use the
  per-generator pixel fields, not the binary `recall`. Do **not** read the
  top-level `pixel_recall` / `pixel_f1` (those are global over all generators).
- **In-domain** uses Qwen-Image and Gemini-2.5 instead.

> Test-composition note: in the PIXAR release, *GPT-Image-2.0* is a weighted blend
> of two quality tiers (⅓ high + ⅔ medium) and *Gemini-3.1* corresponds to the
> `gemini31flash` split. Aggregate `per_model_metrics` accordingly for an exact
> match.

## Text description quality (optional)

`scripts/eval.sh` passes `--save_generated_text`, so each run also writes
`generated_texts.json`, which can be scored for semantic similarity against the
reference descriptions with your preferred text metric.
