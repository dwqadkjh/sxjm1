from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .missing import MissingSpec, apply_contiguous_missing, detect_observed_mask, sample_training_spec


MODALITIES = ("text", "audio", "vision")


def load_pickle(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    with Path(source).open("rb") as handle:
        return pickle.load(handle)


def _as_feature_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected [N,T,D] feature array, got shape {array.shape}")
    array[~np.isfinite(array)] = 0.0
    return array


def _lengths(split_data: Mapping[str, Any], name: str, features: np.ndarray) -> np.ndarray:
    key = f"{name}_lengths"
    if key in split_data:
        lengths = np.asarray(split_data[key]).reshape(-1).astype(np.int64)
    else:
        # Text generally has no explicit lengths.  Trailing all-zero rows count as padding.
        nonzero = np.linalg.norm(features, axis=-1) > 1e-12
        lengths = np.zeros(len(features), dtype=np.int64)
        for i, row in enumerate(nonzero):
            nz = np.flatnonzero(row)
            lengths[i] = int(nz[-1] + 1) if nz.size else features.shape[1]
    return np.clip(lengths, 1, features.shape[1])


def _downsample(x: np.ndarray, valid: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if stride <= 1:
        return x, valid
    out_x, out_valid = [], []
    for start in range(0, x.shape[0], stride):
        end = min(x.shape[0], start + stride)
        chunk_mask = valid[start:end]
        if chunk_mask.any():
            out_x.append(x[start:end][chunk_mask].mean(axis=0))
            out_valid.append(True)
        else:
            out_x.append(np.zeros(x.shape[1], dtype=np.float32))
            out_valid.append(False)
    return np.asarray(out_x, dtype=np.float32), np.asarray(out_valid, dtype=bool)


class MultimodalPickleDataset(Dataset):
    """Reads the supplied aligned/unaligned pickle without changing the split."""

    def __init__(
        self,
        path: str | Path | Mapping[str, Any],
        split: str,
        corruption: str = "none",  # none, random, fixed, natural
        missing_cfg: Mapping[str, Any] | None = None,
        fixed_spec: MissingSpec | None = None,
        seed: int = 1111,
        strides: Mapping[str, int] | None = None,
    ) -> None:
        payload = load_pickle(path)
        if split not in payload:
            raise KeyError(f"Split {split!r} not found. Available: {list(payload)}")
        split_data = payload[split]
        self.split = split
        self.corruption = corruption
        self.missing_cfg = dict(missing_cfg or {})
        self.fixed_spec = fixed_spec
        self.seed = int(seed)
        self.epoch = 0
        self.strides = {**{m: 1 for m in MODALITIES}, **dict(strides or {})}

        self.features = {m: _as_feature_array(split_data[m]) for m in MODALITIES}
        n = len(self.features["text"])
        if any(len(v) != n for v in self.features.values()):
            raise ValueError("Text, audio, and vision sample counts differ")
        self.lengths = {m: _lengths(split_data, m, self.features[m]) for m in MODALITIES}

        self.ids = list(split_data.get("id", [f"{split}_{i:06d}" for i in range(n)]))
        self.raw_text = list(split_data.get("raw_text", [""] * n))
        self.regression = self._optional_label(split_data, "regression_labels", np.float32)
        self.classification = self._optional_label(split_data, "classification_labels", np.int64)
        if self.classification is not None:
            unique = set(np.unique(self.classification).tolist())
            if unique.issubset({-1, 0, 1}):
                self.classification = self.classification + 1
            if not set(np.unique(self.classification)).issubset({0, 1, 2}):
                raise ValueError("classification_labels must encode three classes as -1/0/1 or 0/1/2")

    @staticmethod
    def _optional_label(data: Mapping[str, Any], key: str, dtype: Any) -> np.ndarray | None:
        if key not in data:
            return None
        return np.asarray(data[key], dtype=dtype).reshape(-1)

    def __len__(self) -> int:
        return len(self.ids)

    def set_epoch(self, epoch: int) -> None:
        """Change online training masks while keeping the run reproducible."""
        self.epoch = int(epoch)

    @property
    def feature_dims(self) -> Dict[str, int]:
        return {m: int(self.features[m].shape[-1]) for m in MODALITIES}

    @property
    def sequence_lengths(self) -> Dict[str, int]:
        return {
            m: int(np.ceil(self.features[m].shape[1] / max(1, self.strides[m])))
            for m in MODALITIES
        }

    def _base_sample(self, index: int) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        feats: Dict[str, np.ndarray] = {}
        valid: Dict[str, np.ndarray] = {}
        for name in MODALITIES:
            raw = self.features[name][index]
            mask = np.arange(raw.shape[0]) < int(self.lengths[name][index])
            raw, mask = _downsample(raw, mask, int(self.strides[name]))
            feats[name], valid[name] = raw, mask
        return feats, valid

    def __getitem__(self, index: int) -> Dict[str, Any]:
        complete, valid = self._base_sample(index)
        rng = np.random.default_rng(self.seed + 104729 * index + 1000003 * self.epoch)
        intervals: Dict[str, tuple[int, int]] = {}

        if self.corruption == "none":
            corrupted = {k: v.copy() for k, v in complete.items()}
            observed = {k: v.copy() for k, v in valid.items()}
        elif self.corruption == "natural":
            corrupted = {k: v.copy() for k, v in complete.items()}
            observed = {k: detect_observed_mask(corrupted[k], valid[k]) for k in MODALITIES}
        else:
            if self.corruption == "fixed":
                if self.fixed_spec is None:
                    raise ValueError("fixed corruption requires fixed_spec")
                spec = self.fixed_spec
            elif self.corruption == "random":
                full_probability = float(
                    self.missing_cfg.get("full_modality_dropout_probability", 0.0)
                )
                if full_probability > 0.0 and rng.random() < full_probability:
                    patterns = self.missing_cfg.get(
                        "full_modality_dropout_patterns", ["T", "A", "V", "TA", "TV", "AV"]
                    )
                    configured_weights = self.missing_cfg.get(
                        "full_modality_dropout_weights"
                    )
                    probabilities = None
                    if configured_weights is not None:
                        if isinstance(configured_weights, Mapping):
                            probabilities = np.asarray(
                                [float(configured_weights.get(pattern, 0.0)) for pattern in patterns],
                                dtype=np.float64,
                            )
                        else:
                            probabilities = np.asarray(configured_weights, dtype=np.float64)
                        if probabilities.shape != (len(patterns),):
                            raise ValueError(
                                "full_modality_dropout_weights must match the configured patterns"
                            )
                        if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
                            raise ValueError(
                                "full_modality_dropout_weights must be non-negative with positive sum"
                            )
                        probabilities = probabilities / probabilities.sum()
                    spec = MissingSpec(
                        pattern=str(rng.choice(patterns, p=probabilities)),
                        rate=1.0,
                        position="random",
                    )
                else:
                    spec = sample_training_spec(
                        rng=rng,
                        patterns=self.missing_cfg.get("patterns", ["T", "A", "V", "TA", "TV", "AV", "TAV"]),
                        rate_min=float(self.missing_cfg.get("rate_min", 0.1)),
                        rate_max=float(self.missing_cfg.get("rate_max", 0.5)),
                        complete_probability=float(self.missing_cfg.get("complete_probability", 0.2)),
                    )
                if spec is None:
                    corrupted = {k: v.copy() for k, v in complete.items()}
                    observed = {k: v.copy() for k, v in valid.items()}
                    return self._pack(index, complete, corrupted, valid, observed, intervals)
            else:
                raise ValueError(f"Unknown corruption mode: {self.corruption}")
            corrupted, observed, intervals = apply_contiguous_missing(
                complete,
                valid,
                spec,
                rng,
                bool(self.missing_cfg.get("synchronize_position", True)),
            )
        return self._pack(index, complete, corrupted, valid, observed, intervals)

    def _pack(
        self,
        index: int,
        complete: Mapping[str, np.ndarray],
        corrupted: Mapping[str, np.ndarray],
        valid: Mapping[str, np.ndarray],
        observed: Mapping[str, np.ndarray],
        intervals: Mapping[str, tuple[int, int]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "id": str(self.ids[index]),
            "raw_text": str(self.raw_text[index]),
            "complete": {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in complete.items()},
            "corrupted": {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in corrupted.items()},
            "valid_masks": {k: torch.from_numpy(np.asarray(v, dtype=bool)) for k, v in valid.items()},
            "observed_masks": {k: torch.from_numpy(np.asarray(v, dtype=bool)) for k, v in observed.items()},
            "intervals": {k: tuple(intervals.get(k, (0, 0))) for k in MODALITIES},
        }
        if self.regression is not None:
            result["regression"] = torch.tensor(float(self.regression[index]), dtype=torch.float32)
        if self.classification is not None:
            result["classification"] = torch.tensor(int(self.classification[index]), dtype=torch.long)
        return result


def summarize_pickle(path: str | Path) -> Dict[str, Any]:
    payload = load_pickle(path)
    summary: Dict[str, Any] = {}
    for split, data in payload.items():
        row: Dict[str, Any] = {"samples": len(data.get("id", data["text"]))}
        for name in MODALITIES:
            if name in data:
                row[name] = tuple(np.asarray(data[name]).shape)
        for label in ("classification_labels", "regression_labels"):
            if label in data:
                values = np.asarray(data[label]).reshape(-1)
                row[label] = {
                    "shape": tuple(values.shape),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
        summary[split] = row
    return summary
