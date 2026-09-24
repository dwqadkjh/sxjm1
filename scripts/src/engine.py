from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch

from .losses import multitask_loss
from .metrics import sentiment_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_trained_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """Load a trusted project checkpoint and rebuild its configured model."""
    from .models import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(
        checkpoint["model_name"],
        checkpoint["feature_dims"],
        checkpoint["sequence_lengths"],
        cfg.get("model_params", {}),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _move_nested(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_nested(v, device) for k, v in value.items()}
    return value


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: _move_nested(v, device) for k, v in batch.items()}


def model_forward(model: torch.nn.Module, batch: Mapping[str, Any], complete: bool = False):
    key = "complete" if complete else "corrupted"
    masks = batch["valid_masks"] if complete else batch["observed_masks"]
    if getattr(model, "requires_valid_masks", False):
        return model(batch[key], masks, batch["valid_masks"])
    return model(batch[key], masks)


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Mapping[str, float],
    class_weights: torch.Tensor | None,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    need_complete = (
        float(loss_weights.get("consistency", 0.0)) > 0
        or float(loss_weights.get("latent_reconstruction", 0.0)) > 0
    )
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        missing_output = model_forward(model, batch)
        with torch.no_grad():
            model.eval()
            complete_output = model_forward(model, batch, complete=True) if need_complete else None
            model.train()
        loss, pieces = multitask_loss(
            missing_output, batch, loss_weights, class_weights, complete_output
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(batch["regression"].shape[0])
        count += batch_size
        for key, value in pieces.items():
            totals[key] += float(value.item()) * batch_size
    return {k: v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    loss_weights: Mapping[str, float] | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    ids, y_reg, p_reg, y_cls, p_cls = [], [], [], [], []
    total_loss, samples = 0.0, 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model_forward(model, batch)
        if "regression" in batch and "classification" in batch and loss_weights is not None:
            loss, _ = multitask_loss(output, batch, loss_weights, class_weights, None)
            batch_size = int(batch["regression"].shape[0])
            total_loss += float(loss.item()) * batch_size
            samples += batch_size
        ids.extend(raw_batch["id"])
        p_reg.append(output["regression"].cpu().numpy())
        p_cls.append(output["classification"].argmax(dim=-1).cpu().numpy())
        if "regression" in batch:
            y_reg.append(batch["regression"].cpu().numpy())
        if "classification" in batch:
            y_cls.append(batch["classification"].cpu().numpy())
    predictions = {
        "id": np.asarray(ids),
        "regression_pred": np.clip(np.concatenate(p_reg), -3.0, 3.0),
        "classification_pred": np.concatenate(p_cls),
    }
    metrics: Dict[str, float] = {}
    if y_reg and y_cls:
        predictions["regression_true"] = np.concatenate(y_reg)
        predictions["classification_true"] = np.concatenate(y_cls)
        metrics = sentiment_metrics(
            predictions["regression_true"],
            predictions["regression_pred"],
            predictions["classification_true"],
            predictions["classification_pred"],
        )
        metrics["loss"] = total_loss / max(samples, 1)
    return metrics, predictions


def class_weights_from_labels(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(3.0 * counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)
