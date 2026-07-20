#!/usr/bin/env python3
"""
split_pixar_0_05.py

Re-partition the PIXAR validation split into train + val for pixar_0.05.
Split is by entry (COCO annotation ID) to prevent real-image leakage across splits.

Usage
-----
    python preprocess/split_pixar.py [options]

    --src      Source dataset root (must contain mapping.json and a split dir).
               Default: ./data/PIXAR_preprocessed/test_full_0.05/full_0.05
    --src_split  Which split folder inside --src to use as source.
               Default: validation
    --dst      Destination root for the new pixar_0.05 dataset.
               Default: ./data/pixar_0.05
    --val_frac Fraction of entries assigned to val.  Default: 0.10
    --seed     Random seed.  Default: 42
    --dry_run  Print stats and exit without touching the filesystem.
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _copy_file(src: Path, dst: Path) -> None:
    """Copy src → dst (dst parent must already exist)."""
    shutil.copy2(src, dst)


def _progress(iterable, desc: str, total: int):
    """Minimal progress printer (no tqdm dependency)."""
    report_every = max(1, total // 20)
    for i, item in enumerate(iterable, 1):
        yield item
        if i % report_every == 0 or i == total:
            pct = 100 * i // total
            _log(f"  [{desc}] {i}/{total}  ({pct} %)")


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def compute_split(
    mapping: dict,
    val_frac: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Returns (entry_to_split, tampered_to_split) where values are 'train'/'validation'.

    Split is done at entry (COCO annotation ID) level to prevent data leakage
    via shared real images.
    """
    entry_to_tampered: dict[str, list[str]] = defaultdict(list)
    for tampered_name, meta in mapping.items():
        entry_to_tampered[meta["entry"]].append(tampered_name)

    entries = sorted(entry_to_tampered.keys())   # sorted for determinism
    rng = random.Random(seed)
    rng.shuffle(entries)

    n_val = max(1, round(len(entries) * val_frac))
    val_entries  = set(entries[:n_val])
    train_entries = set(entries[n_val:])

    tampered_to_split: dict[str, str] = {}
    for entry, names in entry_to_tampered.items():
        split = "validation" if entry in val_entries else "train"
        for name in names:
            tampered_to_split[name] = split

    return dict(entry_to_tampered), tampered_to_split


def copy_split(
    src_split_dir: Path,
    dst_split_dir: Path,
    tampered_names: list[str],
    real_names: set[str],
    label: str,
) -> None:
    """
    Physically copy all files belonging to one split into dst_split_dir.

    Subdirectory layout mirrors the PIXAR format:
        tampered/   tampered_XXXX.png
        real/       original_XXXX.png
        soft_masks/ tampered_XXXX_mask.png
        masks/      tampered_XXXX_mask.png
        metadata/   tampered_XXXX_cls.json
        full_synthetic/  (always empty — kept for format compatibility)
    """
    subdirs = ["tampered", "real", "soft_masks", "masks", "metadata", "full_synthetic"]
    for d in subdirs:
        (dst_split_dir / d).mkdir(parents=True, exist_ok=True)

    n = len(tampered_names)

    # ---- tampered images ------------------------------------------------
    _log(f"  Copying tampered images ({n}) ...")
    for name in _progress(tampered_names, "tampered", n):
        _copy_file(src_split_dir / "tampered" / name,
                   dst_split_dir / "tampered" / name)

    # ---- real images ----------------------------------------------------
    _log(f"  Copying real images ({len(real_names)}) ...")
    real_list = sorted(real_names)
    for name in _progress(real_list, "real", len(real_list)):
        _copy_file(src_split_dir / "real" / name,
                   dst_split_dir / "real" / name)

    # ---- soft_masks  (tampered_XXXX_mask.png) ---------------------------
    _log(f"  Copying soft_masks ({n}) ...")
    for tname in _progress(tampered_names, "soft_masks", n):
        stem  = Path(tname).stem
        fname = f"{stem}_mask.png"
        src   = src_split_dir / "soft_masks" / fname
        if src.exists():
            _copy_file(src, dst_split_dir / "soft_masks" / fname)

    # ---- binary masks  (tampered_XXXX_mask.png) -------------------------
    _log(f"  Copying masks ({n}) ...")
    for tname in _progress(tampered_names, "masks", n):
        stem  = Path(tname).stem
        fname = f"{stem}_mask.png"
        src   = src_split_dir / "masks" / fname
        if src.exists():
            _copy_file(src, dst_split_dir / "masks" / fname)

    # ---- metadata  (tampered_XXXX_cls.json) -----------------------------
    _log(f"  Copying metadata ({n}) ...")
    for tname in _progress(tampered_names, "metadata", n):
        stem  = Path(tname).stem
        fname = f"{stem}_cls.json"
        src   = src_split_dir / "metadata" / fname
        if src.exists():
            _copy_file(src, dst_split_dir / "metadata" / fname)

    _log(f"  [{label}] done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src",
                   default="./data/PIXAR_preprocessed/test_full_0.05/full_0.05",
                   help="Source dataset root (contains mapping.json).")
    p.add_argument("--src_split", default="validation",
                   help="Split folder inside --src to re-partition.")
    p.add_argument("--dst", default="./data/pixar_0.05",
                   help="Destination root for the new dataset.")
    p.add_argument("--val_frac", type=float, default=0.10,
                   help="Fraction of entries assigned to validation (default 0.10).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default 42).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print stats and exit without writing anything.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src          = Path(args.src)
    dst          = Path(args.dst)
    src_split_dir = src / args.src_split

    # ---- sanity checks --------------------------------------------------
    mapping_path = src / "mapping.json"
    if not mapping_path.exists():
        _log(f"ERROR: mapping.json not found at {mapping_path}")
        sys.exit(1)
    if not src_split_dir.is_dir():
        _log(f"ERROR: source split dir not found: {src_split_dir}")
        sys.exit(1)
    if dst.exists() and not args.dry_run:
        _log(f"ERROR: destination already exists: {dst}")
        _log("Remove it first or choose a different --dst.")
        sys.exit(1)

    # ---- load mapping ---------------------------------------------------
    _log(f"Loading mapping.json from {mapping_path} ...")
    with open(mapping_path, encoding="utf-8") as f:
        mapping: dict = json.load(f)
    _log(f"  Total entries in mapping.json: {len(mapping)}")

    # ---- compute split --------------------------------------------------
    _log(f"\nComputing split  (val_frac={args.val_frac}, seed={args.seed}) ...")
    entry_to_tampered, tampered_to_split = compute_split(
        mapping, args.val_frac, args.seed
    )

    train_tampered = sorted(
        [k for k, v in tampered_to_split.items() if v == "train"]
    )
    val_tampered = sorted(
        [k for k, v in tampered_to_split.items() if v == "validation"]
    )

    train_real = {mapping[t]["real"] for t in train_tampered}
    val_real   = {mapping[t]["real"] for t in val_tampered}

    _log(f"\n  train : {len(train_tampered):>6} tampered,  {len(train_real):>5} real")
    _log(f"  val   : {len(val_tampered):>6} tampered,  {len(val_real):>5} real")
    _log(f"  test/ : symlink → validation/")

    if args.dry_run:
        _log("\n[dry_run] No files written. Exiting.")
        return

    # ---- copy train -----------------------------------------------------
    _log(f"\n=== Copying TRAIN split ===")
    copy_split(
        src_split_dir  = src_split_dir,
        dst_split_dir  = dst / "train",
        tampered_names = train_tampered,
        real_names     = train_real,
        label          = "train",
    )

    # ---- copy validation ------------------------------------------------
    _log(f"\n=== Copying VALIDATION split ===")
    copy_split(
        src_split_dir  = src_split_dir,
        dst_split_dir  = dst / "validation",
        tampered_names = val_tampered,
        real_names     = val_real,
        label          = "validation",
    )

    # ---- test → validation symlink --------------------------------------
    test_link = dst / "test"
    test_link.symlink_to("validation")
    _log(f"\nCreated symlink: {test_link} → validation/")

    # ---- write updated mapping.json -------------------------------------
    _log("\nWriting mapping.json ...")
    updated_mapping = {}
    for tampered_name, meta in mapping.items():
        updated_meta = dict(meta)
        updated_meta["split"] = tampered_to_split[tampered_name]
        updated_mapping[tampered_name] = updated_meta

    out_mapping = dst / "mapping.json"
    with open(out_mapping, "w", encoding="utf-8") as f:
        json.dump(updated_mapping, f, indent=2, ensure_ascii=False)
    _log(f"  Written: {out_mapping}  ({len(updated_mapping)} entries)")

    # ---- summary --------------------------------------------------------
    _log(f"""
============================================================
  pixar_0.05 dataset written to: {dst}

  train/
    tampered : {len(train_tampered)}
    real     : {len(train_real)}

  validation/
    tampered : {len(val_tampered)}
    real     : {len(val_real)}

  test/  →  validation/   (symlink)

  mapping.json: {len(updated_mapping)} entries  (+split field)
  seed={args.seed}, val_frac={args.val_frac}
============================================================
""")


if __name__ == "__main__":
    main()
