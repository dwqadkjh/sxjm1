from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .data import load_pickle
from .engine import model_forward
from .models.common import MODALITIES


def adaptive_alpha(metrics: Mapping[str, float]) -> float:
    f1 = max(float(metrics.get("macro_f1", 0.0)), 0.0)
    corr = max(float(metrics.get("pearson", 0.0)), 0.0)
    return f1 / (f1 + corr) if f1 + corr > 1e-12 else 0.5


def build_attachment4_payload(input_dir: str | Path) -> tuple[dict, dict[str, dict]]:
    root = Path(input_dir).resolve()
    files = sorted(root.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No attachment-4 pkl files found in {root}")
    merged: dict[str, list] = {m: [] for m in MODALITIES}
    merged.update({"id": [], "raw_text": []})
    metadata: dict[str, dict] = {}
    for file in files:
        sample = load_pickle(file)
        sample = sample.get("test", sample)
        missing = [key for key in (*MODALITIES, "id", "raw_text") if key not in sample]
        if missing:
            raise KeyError(f"{file.name} is missing fields: {missing}")
        sample_id = str(np.asarray(sample["id"]).item())
        if sample_id in metadata:
            raise ValueError(f"Duplicate attachment-4 id: {sample_id}")
        shapes = {m: np.asarray(sample[m]).shape for m in MODALITIES}
        if any(len(shape) != 2 or shape[0] != 50 for shape in shapes.values()):
            raise ValueError(f"{file.name} is not aligned_50: {shapes}")
        video = root / "videos" / f"{file.stem}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"Video matching {file.name} was not found: {video}")
        for name in MODALITIES:
            merged[name].append(np.asarray(sample[name], dtype=np.float32))
        merged["id"].append(sample_id)
        merged["raw_text"].append(str(np.asarray(sample["raw_text"]).item()))
        metadata[sample_id] = {
            "source_file": str(file),
            "video_path": str(video),
            "text_bert": np.asarray(sample.get("text_bert", [])).tolist(),
        }
    split = {m: np.stack(merged[m]) for m in MODALITIES}
    split.update({"id": merged["id"], "raw_text": merged["raw_text"]})
    return {"test": split}, metadata


def mp4_duration(path: str | Path) -> float:
    """Read the mvhd duration using only the Python standard library."""
    path = Path(path)

    def atoms(handle, start: int, end: int):
        offset = start
        while offset + 8 <= end:
            handle.seek(offset)
            size, kind = struct.unpack(">I4s", handle.read(8))
            header = 8
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
                header = 16
            elif size == 0:
                size = end - offset
            if size < header or offset + size > end:
                break
            yield kind, offset + header, offset + size
            offset += size

    with path.open("rb") as handle:
        end = path.stat().st_size
        moov = next((item for item in atoms(handle, 0, end) if item[0] == b"moov"), None)
        if moov is None:
            raise ValueError(f"MP4 moov atom not found: {path}")
        mvhd = next((item for item in atoms(handle, moov[1], moov[2]) if item[0] == b"mvhd"), None)
        if mvhd is None:
            raise ValueError(f"MP4 mvhd atom not found: {path}")
        handle.seek(mvhd[1])
        version = handle.read(1)[0]
        handle.read(3)
        if version == 1:
            handle.read(16)
            timescale = struct.unpack(">I", handle.read(4))[0]
            duration = struct.unpack(">Q", handle.read(8))[0]
        else:
            handle.read(8)
            timescale, duration = struct.unpack(">II", handle.read(8))
    if timescale <= 0:
        raise ValueError(f"Invalid MP4 timescale in {path}")
    return float(duration / timescale)


def _snapshot(output: Mapping[str, torch.Tensor], target_class: int | None = None) -> dict:
    probs = torch.softmax(output["classification"], dim=-1)
    classes = probs.argmax(dim=-1)
    target = int(classes[0].item()) if target_class is None else int(target_class)
    return {
        "intensity": output["regression"].clamp(-3.0, 3.0),
        "probs": probs,
        "class": classes,
        "target": target,
    }


def _impact(full: dict, output: Mapping[str, torch.Tensor], alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = _snapshot(output, full["target"])
    cls_delta = (
        full["probs"][:, full["target"]] - masked["probs"][:, full["target"]]
    )
    reg_delta = full["intensity"] - masked["intensity"]
    impact = alpha * cls_delta.abs() + (1.0 - alpha) * reg_delta.abs() / 6.0
    return (
        impact.detach().cpu().numpy(),
        cls_delta.detach().cpu().numpy(),
        reg_delta.detach().cpu().numpy(),
    )


def _variant_batch(batch: Mapping[str, Any], masks: list[Mapping[str, list[int]]]) -> dict:
    count = len(masks)
    variant = {
        "corrupted": {
            m: batch["corrupted"][m].expand(count, -1, -1).clone() for m in MODALITIES
        },
        "valid_masks": {
            m: batch["valid_masks"][m].expand(count, -1).clone() for m in MODALITIES
        },
        "observed_masks": {
            m: batch["observed_masks"][m].expand(count, -1).clone() for m in MODALITIES
        },
    }
    for row, specification in enumerate(masks):
        for modality, positions in specification.items():
            if positions:
                variant["observed_masks"][modality][row, positions] = False
    return variant


@torch.inference_mode()
def _masked_impacts(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    masks: list[Mapping[str, list[int]]],
    full: dict,
    alpha: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    impacts, cls_deltas, reg_deltas = [], [], []
    for start in range(0, len(masks), chunk_size):
        output = model_forward(model, _variant_batch(batch, masks[start : start + chunk_size]))
        impact, cls_delta, reg_delta = _impact(full, output, alpha)
        impacts.append(impact)
        cls_deltas.append(cls_delta)
        reg_deltas.append(reg_delta)
    return tuple(np.concatenate(values) for values in (impacts, cls_deltas, reg_deltas))


def _normalise(values: np.ndarray) -> tuple[np.ndarray, bool]:
    total = float(np.sum(values))
    return (values / total, False) if total > 1e-12 else (np.zeros_like(values), True)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return 0.0
    rank_left = np.argsort(np.argsort(left)).astype(np.float64)
    rank_right = np.argsort(np.argsort(right)).astype(np.float64)
    if rank_left.std() == 0 or rank_right.std() == 0:
        return 0.0
    return float(np.corrcoef(rank_left, rank_right)[0, 1])


def _perturb_batch(batch: Mapping[str, Any], noise_ratio: float, seed: int) -> dict:
    generator = torch.Generator(device=batch["corrupted"]["text"].device).manual_seed(seed)
    perturbed = {
        "corrupted": {m: value.clone() for m, value in batch["corrupted"].items()},
        "valid_masks": batch["valid_masks"],
        "observed_masks": batch["observed_masks"],
    }
    for modality in MODALITIES:
        value = perturbed["corrupted"][modality]
        observed = batch["observed_masks"][modality].unsqueeze(-1)
        observed_values = value[observed.expand_as(value)]
        scale = (
            observed_values.std(unbiased=False).clamp_min(1e-6)
            if observed_values.numel() else value.new_tensor(1e-6)
        ) * noise_ratio
        noise = torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype)
        perturbed["corrupted"][modality] = torch.where(observed, value + noise * scale, value)
    return perturbed


@torch.inference_mode()
def explain_sample(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    alpha: float,
    faithfulness_ratio: float = 0.2,
    stability_repeats: int = 3,
    noise_ratio: float = 0.01,
    seed: int = 1111,
    chunk_size: int = 64,
) -> dict:
    if int(batch["corrupted"]["text"].shape[0]) != 1:
        raise ValueError("explain_sample requires batch_size=1")
    output = model_forward(model, batch)
    if "reliability_weights" not in output:
        raise ValueError("The checkpoint does not expose Block-EMT reliability_weights")
    full = _snapshot(output)
    observed = {m: batch["observed_masks"][m][0].bool() for m in MODALITIES}
    length = int(batch["corrupted"]["text"].shape[1])

    modality_masks = [
        {m: torch.nonzero(observed[m], as_tuple=False).flatten().tolist()} for m in MODALITIES
    ]
    modality_impact, modality_cls, modality_reg = _masked_impacts(
        model, batch, modality_masks, full, alpha, chunk_size
    )
    modality_share, uninformative = _normalise(modality_impact)

    local_masks, local_index = [], []
    for time in range(length):
        for modality in MODALITIES:
            if bool(observed[modality][time]):
                local_masks.append({modality: [time]})
                local_index.append((time, MODALITIES.index(modality)))
    local_impact = np.zeros((length, len(MODALITIES)), dtype=np.float64)
    if local_masks:
        values, _, _ = _masked_impacts(model, batch, local_masks, full, alpha, chunk_size)
        for (time, modality), value in zip(local_index, values):
            local_impact[time, modality] = float(value)
    time_impact = local_impact.sum(axis=1)
    time_share, _ = _normalise(time_impact)

    time_valid = np.stack([observed[m].cpu().numpy() for m in MODALITIES], axis=1).any(axis=1)
    valid_indices = np.flatnonzero(time_valid)
    count = max(1, int(math.ceil(len(valid_indices) * faithfulness_ratio)))
    ordered = valid_indices[np.argsort(time_impact[valid_indices])]
    bottom, top = ordered[:count].tolist(), ordered[-count:].tolist()
    joint_masks = [
        {m: top for m in MODALITIES},
        {m: bottom for m in MODALITIES},
    ]
    deletion, _, _ = _masked_impacts(model, batch, joint_masks, full, alpha, chunk_size)

    observed_flat = np.stack([observed[m].cpu().numpy() for m in MODALITIES], axis=1).reshape(-1)
    stability = []
    for repeat in range(stability_repeats):
        noisy = _perturb_batch(batch, noise_ratio, seed + repeat)
        noisy_output = model_forward(model, noisy)
        noisy_full = _snapshot(noisy_output, full["target"])
        noisy_values, _, _ = _masked_impacts(
            model, noisy, local_masks, noisy_full, alpha, chunk_size
        )
        noisy_map = np.zeros_like(local_impact)
        for (time, modality), value in zip(local_index, noisy_values):
            noisy_map[time, modality] = float(value)
        stability.append(
            _spearman(local_impact.reshape(-1)[observed_flat], noisy_map.reshape(-1)[observed_flat])
        )

    reliability = output["reliability_weights"][0].detach().cpu().numpy()
    reliability_mean = reliability[time_valid].mean(axis=0) if time_valid.any() else np.zeros(3)
    probabilities = full["probs"][0].detach().cpu().numpy()
    return {
        "prediction": {
            "polarity_index": int(full["class"][0].item()),
            "polarity_probabilities": probabilities.tolist(),
            "intensity": float(full["intensity"][0].item()),
        },
        "alpha": float(alpha),
        "reliability": {
            "modality_mean": dict(zip(MODALITIES, reliability_mean.tolist())),
            "time_modality": reliability.tolist(),
        },
        "contribution": {
            "modality_share": dict(zip(MODALITIES, modality_share.tolist())),
            "modality_impact": dict(zip(MODALITIES, modality_impact.tolist())),
            "classification_delta": dict(zip(MODALITIES, modality_cls.tolist())),
            "intensity_delta": dict(zip(MODALITIES, modality_reg.tolist())),
            "time_modality": local_impact.tolist(),
            "time_share": time_share.tolist(),
            "uninformative": bool(uninformative),
        },
        "faithfulness": {
            "top_positions": top,
            "bottom_positions": bottom,
            "top_impact": float(deletion[0]),
            "bottom_impact": float(deletion[1]),
            "top_gt_bottom": bool(deletion[0] > deletion[1]),
            "stability_spearman": float(np.mean(stability)) if stability else None,
        },
    }


def top_windows(values: list[float], window: int = 5, count: int = 3) -> list[tuple[int, int, float]]:
    array = np.asarray(values, dtype=np.float64)
    window = min(max(1, int(window)), len(array))
    candidates = sorted(
        ((start, start + window, float(array[start : start + window].sum()))
         for start in range(len(array) - window + 1)),
        key=lambda item: item[2],
        reverse=True,
    )
    chosen: list[tuple[int, int, float]] = []
    for item in candidates:
        if all(item[1] <= old[0] or item[0] >= old[1] for old in chosen):
            chosen.append(item)
            if len(chosen) == count:
                break
    return chosen
