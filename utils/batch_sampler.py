import random
from typing import Dict, List

from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    """Balanced batch sampler with configurable real/tampered ratio (--balance_training).

    real_ratio=0.5: deterministic batch_size//2 real + batch_size//2 tampered.
    Otherwise: per-slot Bernoulli(real_ratio); the epoch-level ratio converges over batches.
    All ranks share the epoch seed then slice the batch list; full_synthetic (cls=1) excluded.
    """

    def __init__(self, dataset, batch_size, world_size=1, rank=0, real_ratio=0.5):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.real_ratio = real_ratio

        if batch_size < 2:
            raise ValueError("BalancedBatchSampler requires batch_size >= 2")
        if not 0 < real_ratio < 1:
            raise ValueError(f"real_ratio must be in (0, 1), got {real_ratio}")

        if real_ratio != 0.5:
            half = batch_size // 2
            n_real_rounded = round(batch_size * real_ratio)
            if n_real_rounded == half:
                print(
                    f"[BalancedBatchSampler] WARNING: real_ratio={real_ratio} with "
                    f"batch_size={batch_size} rounds to n_real={n_real_rounded} == "
                    f"half={half} (indistinguishable from 0.5 per batch). "
                    f"Using probability-based sampling; epoch-level ratio converges "
                    f"to {real_ratio:.3f} over many batches."
                )

        self.real_indices     = [i for i, c in enumerate(dataset.cls_labels) if c == 0]
        self.tampered_indices = [i for i, c in enumerate(dataset.cls_labels) if c == 2]

        if len(self.real_indices) == 0:
            raise ValueError("BalancedBatchSampler: no real samples found in dataset")
        if len(self.tampered_indices) == 0:
            raise ValueError("BalancedBatchSampler: no tampered samples found in dataset")

        mode_str = (
            "deterministic half/half"
            if real_ratio == 0.5
            else f"probability-based (real_ratio={real_ratio})"
        )
        print(
            f"\n[BalancedBatchSampler] rank={rank}  "
            f"real={len(self.real_indices)}  tampered={len(self.tampered_indices)}  "
            f"batch_size={batch_size}  mode={mode_str}"
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = random.Random(self.epoch * 1000)

        real_pool     = self.real_indices.copy()
        tampered_pool = self.tampered_indices.copy()
        g.shuffle(real_pool)
        g.shuffle(tampered_pool)

        if self.real_ratio == 0.5:
            all_batches = self._build_deterministic(g, real_pool, tampered_pool)
        else:
            all_batches = self._build_probabilistic(g, real_pool, tampered_pool)

        if self.world_size > 1:
            n = len(all_batches) // self.world_size
            all_batches = all_batches[self.rank * n : (self.rank + 1) * n]

        return iter(all_batches)

    def _build_deterministic(self, g, real_pool, tampered_pool):
        """bs/2 real + bs/2 tampered per batch."""
        half = self.batch_size // 2
        n_batches = len(tampered_pool) // half

        all_batches = []
        real_cursor = 0
        for b in range(n_batches):
            t_start = b * half
            t_slice = tampered_pool[t_start : t_start + half]

            r_slice = []
            while len(r_slice) < half:
                remaining = half - len(r_slice)
                available = real_pool[real_cursor : real_cursor + remaining]
                r_slice.extend(available)
                real_cursor += len(available)
                if real_cursor >= len(real_pool):
                    g.shuffle(real_pool)
                    real_cursor = 0

            batch = r_slice + t_slice
            g.shuffle(batch)
            all_batches.append(batch)

        return all_batches

    def _build_probabilistic(self, g, real_pool, tampered_pool):
        """Per-slot Bernoulli(real_ratio); both pools cycle with reshuffle."""
        expected_tampered_per_batch = self.batch_size * (1 - self.real_ratio)
        n_batches = max(1, round(len(tampered_pool) / expected_tampered_per_batch))

        all_batches = []
        real_cursor     = 0
        tampered_cursor = 0

        for _ in range(n_batches):
            batch = []
            for _ in range(self.batch_size):
                if g.random() < self.real_ratio:
                    batch.append(real_pool[real_cursor])
                    real_cursor += 1
                    if real_cursor >= len(real_pool):
                        g.shuffle(real_pool)
                        real_cursor = 0
                else:
                    batch.append(tampered_pool[tampered_cursor])
                    tampered_cursor += 1
                    if tampered_cursor >= len(tampered_pool):
                        g.shuffle(tampered_pool)
                        tampered_cursor = 0
            all_batches.append(batch)

        return all_batches

    def __len__(self):
        if self.real_ratio == 0.5:
            n_batches = len(self.tampered_indices) // (self.batch_size // 2)
        else:
            expected_tampered = self.batch_size * (1 - self.real_ratio)
            n_batches = max(1, round(len(self.tampered_indices) / expected_tampered))
        if self.world_size > 1:
            return n_batches // self.world_size
        return n_batches


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, world_size=1, rank=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank

        self.indices_by_class = {
            0: [],  # real
            1: [],  # full synthetic
            2: []   # tampered
        }

        for idx in range(len(dataset)):
            cls = dataset.cls_labels[idx]
            self.indices_by_class[cls].append(idx)

        print(f"\nRank {self.rank} - Dataset statistics:")
        for cls in self.indices_by_class:
            print(f"Class {cls}: {len(self.indices_by_class[cls])} images")

        self.original_indices = {cls: indices.copy() for cls, indices in self.indices_by_class.items()}

    def set_epoch(self, epoch):
        """Set epoch for deterministic shuffling across ranks."""
        self.epoch = epoch

    def __iter__(self):
        # Use a deterministic seed so all ranks generate the same batch list,
        # then each rank takes its own slice. This prevents NCCL deadlocks
        # caused by different ranks having different iterator lengths.
        g = random.Random(self.epoch if hasattr(self, 'epoch') else 0)

        self.indices_by_class = {cls: indices.copy() for cls, indices in self.original_indices.items()}

        all_batches = []
        for cls in sorted(self.indices_by_class.keys()):
            indices = list(self.indices_by_class[cls])
            g.shuffle(indices)

            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                batch = indices[i:i + self.batch_size]
                all_batches.append(batch)

        # Shuffle batches to ensure randomness in class distribution between batches
        g.shuffle(all_batches)

        if self.world_size > 1:
            num_batches = len(all_batches)
            num_batches_per_rank = num_batches // self.world_size
            all_batches = all_batches[self.rank * num_batches_per_rank : (self.rank + 1) * num_batches_per_rank]

        return iter(all_batches)

    def __len__(self):
        total_batches = sum(len(indices) // self.batch_size for indices in self.indices_by_class.values())
        if self.world_size > 1:
            return total_batches // self.world_size
        return total_batches


class BalancedSourceWeightedSampler(Sampler):
    """Mixed real+tampered batches with per-source oversampling on the tampered slot.

    The tampered pool is expanded by source_weights (e.g. gemini ×33), so epoch
    length scales with the schedule and weight=0 excludes a source. All ranks
    share the epoch seed then slice the batch list; full_synthetic (cls=1) excluded.
    source_weights may be mutated between epochs (see set_source_weights).

    Args:
        dataset:        training dataset exposing ``.cls_labels`` (0=real, 1=full_synthetic, 2=tampered).
        source_labels:  per-sample source string, length len(dataset).
        source_weights: source name -> non-negative int repeat count (absent => 1).
        batch_size:     per-GPU micro-batch size.
        world_size:     total distributed ranks.
        rank:           this rank's index.
        real_ratio:     fraction of real slots per batch in (0, 1). Default 0.5.
    """

    def __init__(
        self,
        dataset,
        source_labels: List[str],
        source_weights: Dict[str, int],
        batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        real_ratio: float = 0.5,
    ):
        super().__init__(dataset)
        if len(source_labels) != len(dataset):
            raise ValueError(
                f"source_labels length ({len(source_labels)}) must equal "
                f"len(dataset) ({len(dataset)})"
            )
        if batch_size < 2:
            raise ValueError("BalancedSourceWeightedSampler requires batch_size >= 2")
        if not 0 < real_ratio < 1:
            raise ValueError(f"real_ratio must be in (0, 1), got {real_ratio}")
        if any(w < 0 for w in source_weights.values()):
            raise ValueError("All source_weights values must be >= 0")

        self.dataset = dataset
        self.source_labels = source_labels
        self.source_weights = dict(source_weights)
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.real_ratio = real_ratio
        self.epoch = 0

        self.real_indices     = [i for i, c in enumerate(dataset.cls_labels) if c == 0]
        self.tampered_indices = [i for i, c in enumerate(dataset.cls_labels) if c == 2]

        if not self.real_indices:
            raise ValueError("BalancedSourceWeightedSampler: no real samples found")
        if not self.tampered_indices:
            raise ValueError("BalancedSourceWeightedSampler: no tampered samples found")

        self._tampered_expanded_len = self._compute_tampered_expanded_len(self.source_weights)

        src_counts: Dict[str, int] = {}
        for i in self.tampered_indices:
            lbl = source_labels[i]
            w = self.source_weights.get(lbl, 1)
            src_counts[lbl] = src_counts.get(lbl, 0) + w

        mode_str = (
            "deterministic half/half"
            if real_ratio == 0.5
            else f"probability-based (real_ratio={real_ratio})"
        )
        print(
            f"\n[BalancedSourceWeightedSampler] rank={rank}  "
            f"real={len(self.real_indices)}  "
            f"tampered_base={len(self.tampered_indices)}  "
            f"tampered_expanded={self._tampered_expanded_len}  "
            f"batch_size={batch_size}  mode={mode_str}  "
            f"source_weights={self.source_weights}"
        )
        print(f"[BalancedSourceWeightedSampler] per-source expanded tampered counts: {src_counts}")

    def _compute_tampered_expanded_len(self, source_weights: Dict[str, int]) -> int:
        return sum(
            source_weights.get(self.source_labels[i], 1) for i in self.tampered_indices
        )

    def set_source_weights(self, new_weights: Dict[str, int]):
        """Update weights mid-training; next __iter__ rebuilds the expanded pool."""
        if any(w < 0 for w in new_weights.values()):
            raise ValueError("All source_weights values must be >= 0")
        self.source_weights = dict(new_weights)
        self._tampered_expanded_len = self._compute_tampered_expanded_len(self.source_weights)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = random.Random(self.epoch * 1000)

        real_pool = self.real_indices.copy()
        tampered_expanded: List[int] = []
        for idx in self.tampered_indices:
            tampered_expanded.extend(
                [idx] * self.source_weights.get(self.source_labels[idx], 1)
            )

        g.shuffle(real_pool)
        g.shuffle(tampered_expanded)

        if self.real_ratio == 0.5:
            all_batches = self._build_deterministic(g, real_pool, tampered_expanded)
        else:
            all_batches = self._build_probabilistic(g, real_pool, tampered_expanded)

        # avoid contiguous gemini bursts at epoch start
        g.shuffle(all_batches)

        if self.world_size > 1:
            n = len(all_batches) // self.world_size
            all_batches = all_batches[self.rank * n : (self.rank + 1) * n]

        return iter(all_batches)

    def _build_deterministic(self, g, real_pool, tampered_expanded):
        """bs/2 real + bs/2 tampered per batch; real cycles with reshuffle."""
        half = self.batch_size // 2
        if len(tampered_expanded) < half:
            return []

        n_batches = len(tampered_expanded) // half

        all_batches = []
        real_cursor = 0
        for b in range(n_batches):
            t_slice = tampered_expanded[b * half : b * half + half]

            r_slice = []
            while len(r_slice) < half:
                remaining = half - len(r_slice)
                available = real_pool[real_cursor : real_cursor + remaining]
                r_slice.extend(available)
                real_cursor += len(available)
                if real_cursor >= len(real_pool):
                    g.shuffle(real_pool)
                    real_cursor = 0

            batch = r_slice + t_slice
            g.shuffle(batch)
            all_batches.append(batch)

        return all_batches

    def _build_probabilistic(self, g, real_pool, tampered_expanded):
        """Per-slot Bernoulli(real_ratio); both pools cycle with reshuffle."""
        expected_tampered_per_batch = self.batch_size * (1 - self.real_ratio)
        n_batches = max(1, round(len(tampered_expanded) / expected_tampered_per_batch))

        all_batches = []
        real_cursor = 0
        tampered_cursor = 0

        for _ in range(n_batches):
            batch = []
            for _ in range(self.batch_size):
                if g.random() < self.real_ratio:
                    batch.append(real_pool[real_cursor])
                    real_cursor += 1
                    if real_cursor >= len(real_pool):
                        g.shuffle(real_pool)
                        real_cursor = 0
                else:
                    batch.append(tampered_expanded[tampered_cursor])
                    tampered_cursor += 1
                    if tampered_cursor >= len(tampered_expanded):
                        g.shuffle(tampered_expanded)
                        tampered_cursor = 0
            all_batches.append(batch)

        return all_batches

    def __len__(self):
        if self.real_ratio == 0.5:
            half = self.batch_size // 2
            n_batches = self._tampered_expanded_len // half
        else:
            expected_tampered = self.batch_size * (1 - self.real_ratio)
            n_batches = max(1, round(self._tampered_expanded_len / expected_tampered))
        if self.world_size > 1:
            return n_batches // self.world_size
        return n_batches


def parse_weights_schedule(schedule_json: str, total_epochs: int) -> List[tuple]:
    """Parse --source_weights_schedule JSON into sorted [start, end) intervals.

    '{"0-2": {"gemini": 0}, "2-end": {"gemini": 33}}' -> [(0,2,{...}), (2,5,{...})].
    Intervals must be contiguous and cover [0, total_epochs); "end" => total_epochs;
    weights are non-negative ints (weight=0 excludes that source).
    """
    import json as _j
    raw = _j.loads(schedule_json)
    intervals = []
    for key, weights in raw.items():
        parts = key.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid interval key: {key!r} (expected 'N-M' or 'N-end')")
        try:
            start = int(parts[0])
            end = total_epochs if parts[1] == "end" else int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid interval key: {key!r}")
        for src, w in weights.items():
            if not isinstance(w, int) or w < 0:
                raise ValueError(
                    f"Weight for source {src!r} must be a non-negative integer, got {w!r}"
                )
        intervals.append((start, end, weights))
    intervals.sort(key=lambda x: x[0])
    if not intervals or intervals[0][0] != 0:
        raise ValueError("Schedule must start at epoch 0")
    if intervals[-1][1] != total_epochs:
        raise ValueError(
            f"Schedule must end at epoch {total_epochs}, got {intervals[-1][1]}"
        )
    for i in range(len(intervals) - 1):
        if intervals[i][1] != intervals[i + 1][0]:
            raise ValueError(
                f"Schedule gap or overlap between intervals "
                f"[{intervals[i][0]},{intervals[i][1]}) and "
                f"[{intervals[i+1][0]},{intervals[i+1][1]})"
            )
    return intervals
