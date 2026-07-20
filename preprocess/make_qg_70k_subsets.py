#!/usr/bin/env python3
"""
make_qg_70k_subsets.py

Build the pixar_qg_70k_* dataset family (qwen pool + nested gemini pool, all symlinked).

Usage
-----
  python preprocess/make_qg_70k_subsets.py --variant all [--dry_run]
  python preprocess/make_qg_70k_subsets.py --variant 3353x1
  python preprocess/make_qg_70k_subsets.py --variant 335x10 --skip_existing
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Variant table:  variant_key → (dataset_name, n_unique_gemini, n_reps, n_qwen)

VARIANTS: dict[str, tuple[str, int, int, int]] = {
    # 70K family (paper main + rep ablation)
    "3353x1":      ("pixar_qg_70k_3353x1",   3353,  1,  70_000),  # Tier-1 main
    "335x10":      ("pixar_qg_70k_335x10",    335, 10,  70_000),  # Tier-2 mid-rep
    "67x50":       ("pixar_qg_70k_67x50",      67, 50,  70_000),  # Tier-2 extreme-rep
    # Scaling study (qwen scale-up/down, gemini fixed at 3,353×1)
    "30k_3353x1":  ("pixar_qg_30k_3353x1",   3353,  1,  30_000),  # low-scale (target exposure ↑)
    "150k_3353x1": ("pixar_qg_150k_3353x1",  3353,  1, 150_000),  # mid-scale
    "380k_3353x1": ("pixar_qg_380k_3353x1",  3353,  1, 380_000),  # max-scale
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _symlink_safe(link: Path, target: Path) -> bool:
    """Create symlink. Return False if already exists (collision tolerated)."""
    try:
        link.symlink_to(target.resolve())
        return True
    except FileExistsError:
        return False


def _get_op(type_str: str) -> str:
    """
    Extract op name (handles all variants):
      'coco_train_addition'              → 'addition'
      'coco_val_motion'                  → 'motion'
      'gemini_coco_val_replacement_1'    → 'replacement_1'
      'qwen_coco_train_inter_replacement_1' → 'inter_replacement_1'
    """
    for marker in ("_coco_val_", "_coco_train_"):
        if marker in type_str:
            return type_str.split(marker, 1)[1]
    if type_str.startswith("coco_val_"):
        return type_str[len("coco_val_"):]
    if type_str.startswith("coco_train_"):
        return type_str[len("coco_train_"):]
    return type_str


def _rewrite_qwen_type(type_str: str) -> str:
    """
    Rewrite train_0.05 'coco_X_op' → 'qwen_coco_X_op' for source filter compatibility.
      'coco_train_addition' → 'qwen_coco_train_addition'
      'coco_val_motion'     → 'qwen_coco_val_motion'
    Idempotent: already-rewritten types pass through.
    """
    if type_str.startswith(("coco_train_", "coco_val_")):
        return f"qwen_{type_str}"
    return type_str


def _related_filenames(tampered_name: str) -> tuple[str, str]:
    stem = Path(tampered_name).stem
    return f"{stem}_mask.png", f"{stem}_cls.json"


def _symlink_batch(tasks: list[tuple[Path, Path]], desc: str, n_workers: int = 8) -> int:
    """Returns number of successfully created (non-collision) symlinks."""
    total = len(tasks)
    if total == 0:
        return 0
    done = 0
    created = 0
    report_every = max(1, total // 10)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_symlink_safe, lnk, tgt): i for i, (lnk, tgt) in enumerate(tasks)}
        for fut in as_completed(futures):
            if fut.result():
                created += 1
            done += 1
            if done % report_every == 0 or done == total:
                _log(f"  [{desc}] {done}/{total}  ({100 * done // total} %)  created={created}")
    return created


# ---------------------------------------------------------------------------
# Qwen 70K pool selection (with mask)
# ---------------------------------------------------------------------------

def build_qwen_pool(
    train_05_mapping: dict,
    val_entries: set[str],
    n_target: int,
    qwen_train_dir: Path,
) -> tuple[list[str], Counter]:
    """
    Round-robin select n_target qwen tampered names from train_0.05.

    Filtering:
      1. Mask entries whose `entry` ∈ val_entries (Q2 leak prevention)
      2. Mask entries whose tampered file does NOT exist on disk
         (mapping.json has 416K records but train_0.05 only has 387K files;
          ~28K phantom entries must be skipped).

    Returns (selected_pool, op_distribution).
    """
    by_op: dict[str, list[str]] = defaultdict(list)
    masked_val = 0
    masked_phantom = 0

    tampered_dir = qwen_train_dir / "tampered"
    masks_dir = qwen_train_dir / "masks"
    soft_masks_dir = qwen_train_dir / "soft_masks"
    metadata_dir = qwen_train_dir / "metadata"

    for k, v in train_05_mapping.items():
        if v["entry"] in val_entries:
            masked_val += 1
            continue
        # Check all 4 source files exist (tampered + 3 derived)
        stem = k.replace(".png", "")
        if not (tampered_dir / k).exists():
            masked_phantom += 1
            continue
        # Defensive: also check derived files (should be parallel; if any miss we skip)
        if not (soft_masks_dir / f"{stem}_mask.png").exists():
            masked_phantom += 1
            continue
        if not (masks_dir / f"{stem}_mask.png").exists():
            masked_phantom += 1
            continue
        if not (metadata_dir / f"{stem}_cls.json").exists():
            masked_phantom += 1
            continue
        by_op[_get_op(v["type"])].append(k)

    _log(f"\n  Filter stats:")
    _log(f"    masked (entry ∈ val): {masked_val:,}")
    _log(f"    masked (phantom — no disk file): {masked_phantom:,}")
    _log(f"    remaining candidates: {sum(len(v) for v in by_op.values()):,}")

    for op in by_op:
        by_op[op].sort()

    op_order = sorted(by_op.keys(), key=lambda op: len(by_op[op]))
    n_types = len(op_order)

    _log(f"\n  Qwen pool selection (α: round-robin, all ops):")
    _log(f"    Available after all filters: {sum(len(v) for v in by_op.values()):,}")
    _log(f"    Op count: {n_types}")
    for op in op_order:
        _log(f"      {op:30s}: {len(by_op[op]):>7,}")

    selected: list[str] = []
    op_counts: Counter = Counter()
    indices: dict[str, int] = {op: 0 for op in op_order}
    iters = 0

    while len(selected) < n_target:
        op = op_order[iters % n_types]
        idx = indices[op]
        if idx < len(by_op[op]):
            selected.append(by_op[op][idx])
            op_counts[op] += 1
            indices[op] = idx + 1
        iters += 1
        if all(indices[o] >= len(by_op[o]) for o in op_order):
            break

    _log(f"\n  Selected {len(selected):,} qwen entries:")
    for op in op_order:
        avail = len(by_op[op])
        sel = op_counts.get(op, 0)
        _log(f"    {op:30s}: {sel:>7,}  / {avail:>7,}  ({100*sel/avail:.1f}%)")
    if op_counts:
        m = max(op_counts.values()) / max(min(op_counts.values()), 1)
        _log(f"  Max:Min ratio: {m:.1f}×")

    return selected, op_counts


# ---------------------------------------------------------------------------
# Gemini nested pool (3353 ⊃ 335 ⊃ 67)
# ---------------------------------------------------------------------------

def build_gemini_nested_pool(pixar_05_mapping: dict, max_n: int) -> list[str]:
    """
    Returns ordered list of up to max_n gemini tampered filenames.
    Algorithm: dedup by entry (prefer bg=False) → round-robin by op-type.
    Strict nesting: pool[:N1] ⊂ pool[:N2] for N1 < N2.
    """
    gemini_train = {
        k: v for k, v in pixar_05_mapping.items()
        if v.get("split") == "train"
        and v.get("type", "").split("_coco_")[0] == "gemini"
    }

    # Dedup by entry, prefer bg=False, tie-break by filename
    by_entry: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for k, v in gemini_train.items():
        by_entry[v["entry"]].append((k, v))

    unique: list[tuple[str, dict]] = []
    for entry, candidates in by_entry.items():
        candidates.sort(key=lambda kv: (kv[1].get("bg", False), kv[0]))
        unique.append(candidates[0])

    # Group by op
    by_op: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for k, v in unique:
        by_op[_get_op(v["type"])].append((k, v))
    for op in by_op:
        by_op[op].sort(key=lambda kv: kv[0])

    op_order = sorted(by_op.keys(), key=lambda op: len(by_op[op]))
    n_types = len(op_order)

    _log(f"\n  Gemini nested pool selection:")
    _log(f"    Source: gemini train tampered = {len(gemini_train):,}")
    _log(f"    Dedup by entry → unique = {len(unique):,}")
    for op in op_order:
        _log(f"      {op:30s}: {len(by_op[op]):>5,}")

    # Round-robin
    selected: list[str] = []
    indices: dict[str, int] = {op: 0 for op in op_order}
    iters = 0
    while len(selected) < max_n:
        op = op_order[iters % n_types]
        idx = indices[op]
        if idx < len(by_op[op]):
            selected.append(by_op[op][idx][0])
            indices[op] = idx + 1
        iters += 1
        if all(indices[o] >= len(by_op[o]) for o in op_order):
            break

    return selected


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    variant: str,
    qwen_pool: list[str],
    qwen_mapping: dict,
    qwen_train_dir: Path,
    gemini_pool_full: list[str],
    gemini_mapping: dict,
    gemini_train_dir: Path,
    val_mapping: dict,
    val_dir_link_target: Path,
    out_dir: Path,
    dry_run: bool,
    skip_existing: bool,
    n_workers: int,
) -> None:
    dataset_name, n_unique_gemini, n_reps, n_qwen_target = VARIANTS[variant]
    dst = out_dir / dataset_name

    # Slice qwen_pool to the per-variant target. Larger pools strictly
    # contain smaller (deterministic round-robin), so we just take the prefix.
    qwen_pool = qwen_pool[:n_qwen_target]

    gemini_selected = gemini_pool_full[:n_unique_gemini]
    n_gemini_total = n_unique_gemini * n_reps
    n_qwen = len(qwen_pool)

    _log(f"\n{'=' * 70}")
    _log(f"  Building : {dataset_name}")
    _log(f"  variant  : {variant}  |  qwen={n_qwen:,}  |  gemini={n_unique_gemini}u × {n_reps}")
    _log(f"  total train tampered: {n_qwen + n_gemini_total:,}")
    _log(f"  dst      : {dst}")

    if dst.exists():
        if skip_existing:
            _log(f"  [skip] {dst} already exists.")
            return
        _log(f"  ERROR: destination already exists: {dst}")
        sys.exit(1)

    if dry_run:
        _log("  [dry_run] No files written.")
        return

    # ---- Directory skeleton -------------------------------------------------
    train_dst = dst / "train"
    train_dst.mkdir(parents=True, exist_ok=True)
    for subdir in ("tampered", "soft_masks", "masks", "metadata", "real"):
        (train_dst / subdir).mkdir(exist_ok=True)
    # full_synthetic kept empty for format compatibility
    (train_dst / "full_synthetic").mkdir(exist_ok=True)

    # validation/test → directory symlinks
    _symlink_safe(dst / "validation", val_dir_link_target)
    _symlink_safe(dst / "test",       val_dir_link_target)
    _log(f"  validation/, test/ → {val_dir_link_target.resolve()}")

    # ---- Symlink qwen tampered + masks + metadata ---------------------------
    _log(f"\n  Symlinking qwen files ({n_qwen:,} × 4 subdirs) ...")
    qwen_tasks: dict[str, list[tuple[Path, Path]]] = {
        s: [] for s in ("tampered", "soft_masks", "masks", "metadata")
    }
    qwen_real_files: set[str] = set()
    for name in qwen_pool:
        mask_name, meta_name = _related_filenames(name)
        qwen_tasks["tampered"].append(
            (train_dst / "tampered" / name, qwen_train_dir / "tampered" / name)
        )
        # PIXAR raw uses soft_masks/masks naming — verify on first record
        sm_src = qwen_train_dir / "soft_masks" / mask_name
        if sm_src.exists():
            qwen_tasks["soft_masks"].append((train_dst / "soft_masks" / mask_name, sm_src))
        m_src = qwen_train_dir / "masks" / mask_name
        if m_src.exists():
            qwen_tasks["masks"].append((train_dst / "masks" / mask_name, m_src))
        md_src = qwen_train_dir / "metadata" / meta_name
        if md_src.exists():
            qwen_tasks["metadata"].append((train_dst / "metadata" / meta_name, md_src))
        qwen_real_files.add(qwen_mapping[name]["real"])

    for subdir, tasks in qwen_tasks.items():
        _symlink_batch(tasks, f"qwen/{subdir}", n_workers)

    # ---- Symlink qwen real images ------------------------------------------
    _log(f"\n  Symlinking qwen real images ({len(qwen_real_files):,}) ...")
    real_tasks = [
        (train_dst / "real" / r, qwen_train_dir / "real" / r)
        for r in sorted(qwen_real_files)
    ]
    _symlink_batch(real_tasks, "qwen/real", n_workers)

    # ---- Symlink gemini tampered + masks + metadata (with reps) ------------
    _log(f"\n  Symlinking gemini files "
         f"({'original names' if n_reps == 1 else f'_rep01.._rep{n_reps:02d}'}) ...")
    gemini_tasks: dict[str, list[tuple[Path, Path]]] = {
        s: [] for s in ("tampered", "soft_masks", "masks", "metadata")
    }
    gemini_mapping_entries: dict[str, dict] = {}
    gemini_real_files: set[str] = set()

    for orig_name in gemini_selected:
        orig_mask, orig_meta = _related_filenames(orig_name)
        orig_stem = Path(orig_name).stem
        gemini_real_files.add(gemini_mapping[orig_name]["real"])

        if n_reps == 1:
            gemini_tasks["tampered"].append(
                (train_dst / "tampered" / orig_name, gemini_train_dir / "tampered" / orig_name)
            )
            sm_src = gemini_train_dir / "soft_masks" / orig_mask
            if sm_src.exists():
                gemini_tasks["soft_masks"].append((train_dst / "soft_masks" / orig_mask, sm_src))
            m_src = gemini_train_dir / "masks" / orig_mask
            if m_src.exists():
                gemini_tasks["masks"].append((train_dst / "masks" / orig_mask, m_src))
            md_src = gemini_train_dir / "metadata" / orig_meta
            if md_src.exists():
                gemini_tasks["metadata"].append((train_dst / "metadata" / orig_meta, md_src))
            gemini_mapping_entries[orig_name] = dict(gemini_mapping[orig_name])
        else:
            for rep in range(1, n_reps + 1):
                suffix = f"_rep{rep:02d}"
                rep_t = f"{orig_stem}{suffix}.png"
                rep_m = f"{orig_stem}{suffix}_mask.png"
                rep_md = f"{orig_stem}{suffix}_cls.json"

                gemini_tasks["tampered"].append(
                    (train_dst / "tampered" / rep_t, gemini_train_dir / "tampered" / orig_name)
                )
                sm_src = gemini_train_dir / "soft_masks" / orig_mask
                if sm_src.exists():
                    gemini_tasks["soft_masks"].append(
                        (train_dst / "soft_masks" / rep_m, sm_src)
                    )
                m_src = gemini_train_dir / "masks" / orig_mask
                if m_src.exists():
                    gemini_tasks["masks"].append(
                        (train_dst / "masks" / rep_m, m_src)
                    )
                md_src = gemini_train_dir / "metadata" / orig_meta
                if md_src.exists():
                    gemini_tasks["metadata"].append(
                        (train_dst / "metadata" / rep_md, md_src)
                    )
                gemini_mapping_entries[rep_t] = dict(gemini_mapping[orig_name])

    for subdir, tasks in gemini_tasks.items():
        _symlink_batch(tasks, f"gemini/{subdir}", n_workers)

    # ---- Symlink gemini real images ----------------------------------------
    _log(f"\n  Symlinking gemini real images ({len(gemini_real_files):,}) ...")
    gemini_real_tasks = [
        (train_dst / "real" / r, gemini_train_dir / "real" / r)
        for r in sorted(gemini_real_files)
    ]
    created_g = _symlink_batch(gemini_real_tasks, "gemini/real", n_workers)
    _log(f"  (collision-skipped: {len(gemini_real_tasks) - created_g})")

    # ---- mapping.json -------------------------------------------------------
    out_mapping: dict[str, dict] = {}

    # qwen entries with rewritten type + split=train
    for name in qwen_pool:
        m = dict(qwen_mapping[name])
        m["type"] = _rewrite_qwen_type(m["type"])
        m["split"] = "train"
        out_mapping[name] = m

    # gemini entries (already correctly typed) + split=train
    for name, m in gemini_mapping_entries.items():
        mm = dict(m)
        mm["split"] = "train"
        out_mapping[name] = mm

    # validation entries (carried over from pixar_0.05)
    out_mapping.update(val_mapping)

    with open(dst / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(out_mapping, f, indent=2, ensure_ascii=False)

    # ---- Summary -----------------------------------------------------------
    n_unique_real = len(qwen_real_files | gemini_real_files)
    _log(f"""
  ============================================================
  Dataset written to: {dst}

  train/
    tampered/   qwen: {n_qwen:,}  +  gemini: {n_gemini_total:,}  (all symlinks)
    real/       qwen: {len(qwen_real_files):,}  +  gemini: {len(gemini_real_files):,}
                (unique union: {n_unique_real:,})

  validation/  → {val_dir_link_target.resolve()}  (symlink)
  test/        → {val_dir_link_target.resolve()}  (symlink)

  mapping.json: {len(out_mapping):,} entries
    qwen train  : {n_qwen:,}  (type rewritten to 'qwen_coco_X_op')
    gemini train: {n_gemini_total:,}  ({n_unique_gemini} unique × {n_reps})
    val         : {len(val_mapping):,}

  Training: natural sampling (do NOT pass --balance_training).
  Seed: pool selection is deterministic.
  ============================================================
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variant",
        choices=list(VARIANTS) + ["all"],
        required=True,
        help="Which subset(s) to build: " + " | ".join(VARIANTS) + " | all",
    )
    p.add_argument(
        "--qwen_src",
        default="./data/PIXAR_preprocessed/train_0.05/ours_0.05",
    )
    p.add_argument(
        "--gemini_src",
        default="./data/pixar_0.05",
    )
    p.add_argument(
        "--out_dir",
        default="./data",
    )
    p.add_argument("--workers",       type=int, default=8)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qwen_src   = Path(args.qwen_src)
    gemini_src = Path(args.gemini_src)
    out_dir    = Path(args.out_dir)

    for path, label in [
        (qwen_src   / "mapping.json", "--qwen_src/mapping.json"),
        (gemini_src / "mapping.json", "--gemini_src/mapping.json"),
        (gemini_src / "validation",   "--gemini_src/validation"),
    ]:
        if not path.exists():
            _log(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    # ---- Load mappings ---------------------------------------------------
    _log(f"Loading {qwen_src}/mapping.json ...")
    with open(qwen_src / "mapping.json", encoding="utf-8") as f:
        qwen_mapping = json.load(f)
    _log(f"  {len(qwen_mapping):,} records")

    _log(f"Loading {gemini_src}/mapping.json ...")
    with open(gemini_src / "mapping.json", encoding="utf-8") as f:
        pixar_05_mapping = json.load(f)
    _log(f"  {len(pixar_05_mapping):,} records")

    # ---- Compute val_entries (for Q2 mask + carry-over to new mapping) ---
    val_entries = {v["entry"] for v in pixar_05_mapping.values()
                   if v.get("split") == "validation"}
    val_mapping = {k: dict(v) for k, v in pixar_05_mapping.items()
                   if v.get("split") == "validation"}
    _log(f"\n  pixar_0.05 val entries: {len(val_entries):,}, records: {len(val_mapping):,}")

    # ---- Build qwen pool sized to MAX needed by selected variants --------
    variants_to_build = list(VARIANTS) if args.variant == "all" else [args.variant]
    max_qwen_needed = max(VARIANTS[v][3] for v in variants_to_build)
    _log(f"\n=== Building qwen pool (max needed = {max_qwen_needed:,}) ===")
    qwen_pool, _ = build_qwen_pool(qwen_mapping, val_entries, max_qwen_needed,
                                    qwen_src / "train")

    # ---- Build gemini nested pool (max needed = 3353) --------------------
    _log(f"\n=== Building gemini nested pool (max {max(v[1] for v in VARIANTS.values())}) ===")
    max_n_gemini = max(v[1] for v in VARIANTS.values())
    gemini_pool_full = build_gemini_nested_pool(pixar_05_mapping, max_n_gemini)
    _log(f"  Selected: {len(gemini_pool_full)} unique gemini entries")
    if len(gemini_pool_full) < max_n_gemini:
        _log(f"  WARNING: requested {max_n_gemini}, only got {len(gemini_pool_full)}")

    # ---- Build each variant ---------------------------------------------
    for variant in variants_to_build:
        build_dataset(
            variant=variant,
            qwen_pool=qwen_pool,
            qwen_mapping=qwen_mapping,
            qwen_train_dir=qwen_src / "train",
            gemini_pool_full=gemini_pool_full,
            gemini_mapping=pixar_05_mapping,
            gemini_train_dir=gemini_src / "train",
            val_mapping=val_mapping,
            val_dir_link_target=gemini_src / "validation",
            out_dir=out_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            n_workers=args.workers,
        )

    _log("\n=== make_qg_70k_subsets: done ===")


if __name__ == "__main__":
    main()
