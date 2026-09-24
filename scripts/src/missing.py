from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np


MODALITY_CODES = {"T": "text", "A": "audio", "V": "vision"}


@dataclass(frozen=True)
class MissingSpec:
    """Specification for one continuous missing block per selected modality."""

    pattern: str = "TAV"
    rate: float = 0.3
    position: str = "random"  # early, middle, late, random

    def modalities(self) -> tuple[str, ...]:
        keys = tuple(MODALITY_CODES[c] for c in self.pattern.upper() if c in MODALITY_CODES)
        if not keys:
            raise ValueError(f"Invalid missing pattern: {self.pattern!r}")
        return keys


def _block_start(valid_len: int, block_len: int, position: str, rng: np.random.Generator) -> int:
    max_start = max(0, valid_len - block_len)
    if position == "random":
        return int(rng.integers(0, max_start + 1))
    anchors = {"early": 1 / 6, "middle": 1 / 2, "late": 5 / 6}
    if position not in anchors:
        raise ValueError(f"Unknown position {position!r}; choose early/middle/late/random")
    center = anchors[position] * max(valid_len - 1, 1)
    return int(np.clip(round(center - block_len / 2), 0, max_start))


def apply_contiguous_missing(
    features: Mapping[str, np.ndarray],
    valid_masks: Mapping[str, np.ndarray],
    spec: MissingSpec,
    rng: np.random.Generator,
    synchronize_position: bool = True,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, tuple[int, int]]]:
    """Zero a continuous interval and return corrupted arrays and observed masks.

    Arrays represent a single sample with shape ``[time, feature_dim]``.  The
    operation never alters padding positions.  When multiple modalities are
    selected, ``synchronize_position`` uses the same normalized block center.
    """

    rate = float(np.clip(spec.rate, 0.0, 1.0))
    corrupted = {k: np.array(v, copy=True) for k, v in features.items()}
    observed = {k: np.asarray(valid_masks[k], dtype=bool).copy() for k in features}
    intervals: Dict[str, tuple[int, int]] = {}

    shared_u = float(rng.random()) if spec.position == "random" and synchronize_position else None
    for name in spec.modalities():
        valid_len = int(np.asarray(valid_masks[name], dtype=bool).sum())
        if valid_len <= 0 or rate <= 0:
            intervals[name] = (0, 0)
            continue
        block_len = min(valid_len, max(1, int(round(rate * valid_len))))
        if shared_u is not None:
            start = int(round(shared_u * max(0, valid_len - block_len)))
        else:
            start = _block_start(valid_len, block_len, spec.position, rng)
        end = start + block_len
        corrupted[name][start:end] = 0.0
        observed[name][start:end] = False
        intervals[name] = (start, end)
    return corrupted, observed, intervals


def detect_observed_mask(features: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Detect attachment-3 missing frames while preserving the known valid span."""

    nonzero = np.linalg.norm(np.asarray(features), axis=-1) > eps
    return np.asarray(valid_mask, dtype=bool) & nonzero


def sample_training_spec(
    rng: np.random.Generator,
    patterns: Iterable[str],
    rate_min: float,
    rate_max: float,
    complete_probability: float,
) -> MissingSpec | None:
    if rng.random() < complete_probability:
        return None
    pattern_list = tuple(patterns)
    if not pattern_list:
        raise ValueError("At least one missing pattern is required")
    return MissingSpec(
        pattern=str(rng.choice(pattern_list)),
        rate=float(rng.uniform(rate_min, rate_max)),
        position="random",
    )

