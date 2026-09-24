from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def reconstruction_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    valid_masks: Mapping[str, torch.Tensor],
    observed_masks: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for name, pred in predictions.items():
        missing = valid_masks[name].bool() & ~observed_masks[name].bool()
        if missing.any():
            losses.append(F.smooth_l1_loss(pred[missing], targets[name][missing]))
    if not losses:
        first = next(iter(targets.values()))
        return first.sum() * 0.0
    return torch.stack(losses).mean()


def feature_consistency_loss(missing_feature: torch.Tensor, complete_feature: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(missing_feature, complete_feature.detach(), dim=-1)).mean()


def latent_reconstruction_loss(
    predictions: Mapping[str, torch.Tensor],
    complete_targets: Mapping[str, torch.Tensor],
    valid_masks: Mapping[str, torch.Tensor],
    observed_masks: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for name, pred in predictions.items():
        missing = valid_masks[name].bool() & ~observed_masks[name].bool()
        if missing.any():
            losses.append(F.smooth_l1_loss(pred[missing], complete_targets[name][missing].detach()))
    if not losses:
        return next(iter(predictions.values())).sum() * 0.0
    return torch.stack(losses).mean()


def modality_auxiliary_loss(
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
    available: torch.Tensor,
    regression_targets: torch.Tensor,
    classification_targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    modality_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: dict[str, torch.Tensor] = {}
    for index, name in enumerate(("text", "audio", "vision")):
        selected = available[:, index].bool()
        if selected.any():
            reg = F.smooth_l1_loss(
                predictions[name]["regression"][selected], regression_targets[selected]
            )
            cls = F.cross_entropy(
                predictions[name]["classification"][selected],
                classification_targets[selected],
                weight=class_weights,
            )
            losses[name] = reg + cls
    if losses:
        if modality_weights is None:
            total = torch.stack(list(losses.values())).mean()
        else:
            configured = {name: float(modality_weights.get(name, 0.0)) for name in losses}
            if any(value < 0.0 for value in configured.values()):
                raise ValueError("modality auxiliary weights must be non-negative")
            weight_sum = sum(configured.values())
            if weight_sum <= 0.0:
                raise ValueError("at least one available modality auxiliary weight must be positive")
            total = sum(configured[name] * loss for name, loss in losses.items()) / weight_sum
        return total, losses
    zero = regression_targets.sum() * 0.0
    return zero, {}


def multitask_loss(
    missing_output: Mapping[str, torch.Tensor],
    batch: Mapping[str, object],
    weights: Mapping[str, float],
    class_weights: torch.Tensor | None = None,
    complete_output: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reg = F.smooth_l1_loss(missing_output["regression"], batch["regression"])
    cls = F.cross_entropy(
        missing_output["classification"], batch["classification"], weight=class_weights
    )
    recon = reconstruction_loss(
        missing_output.get("reconstruction", {}),
        batch["complete"],
        batch["valid_masks"],
        batch["observed_masks"],
    )
    if complete_output is None:
        consistency = reg * 0.0
        latent_reconstruction = reg * 0.0
    else:
        consistency = feature_consistency_loss(
            missing_output["feature"], complete_output["feature"]
        )
        if "latent_reconstruction" in missing_output and "latent_targets" in complete_output:
            latent_reconstruction = latent_reconstruction_loss(
                missing_output["latent_reconstruction"], complete_output["latent_targets"],
                batch["valid_masks"], batch["observed_masks"],
            )
        else:
            latent_reconstruction = reg * 0.0
    task_log_vars = missing_output.get("task_log_vars")
    if task_log_vars is None:
        task_total = (
            float(weights.get("regression", 1.0)) * reg
            + float(weights.get("classification", 1.0)) * cls
        )
    else:
        cls_log_var, reg_log_var = task_log_vars.unbind()
        task_total = (
            0.5 * torch.exp(-cls_log_var) * cls + 0.5 * cls_log_var
            + 0.5 * torch.exp(-reg_log_var) * reg + 0.5 * reg_log_var
        )
    if "modality_auxiliary" in missing_output:
        modality_auxiliary, modality_auxiliary_parts = modality_auxiliary_loss(
            missing_output["modality_auxiliary"],
            missing_output["modality_available"],
            batch["regression"],
            batch["classification"],
            class_weights,
            weights.get("modality_auxiliary_weights"),
        )
    else:
        modality_auxiliary = reg * 0.0
        modality_auxiliary_parts = {}
    total = (
        task_total
        + float(weights.get("reconstruction", 0.0)) * recon
        + float(weights.get("consistency", 0.0)) * consistency
        + float(weights.get("latent_reconstruction", 0.0)) * latent_reconstruction
        + float(weights.get("modality_auxiliary", 0.0)) * modality_auxiliary
    )
    pieces = {
        "total": total.detach(),
        "regression": reg.detach(),
        "classification": cls.detach(),
        "reconstruction": recon.detach(),
        "consistency": consistency.detach(),
        "latent_reconstruction": latent_reconstruction.detach(),
        "modality_auxiliary": modality_auxiliary.detach(),
    }
    for name, value in modality_auxiliary_parts.items():
        pieces[f"modality_auxiliary_{name}"] = value.detach()
    if task_log_vars is not None:
        pieces["classification_log_var"] = task_log_vars[0].detach()
        pieces["regression_log_var"] = task_log_vars[1].detach()
    for name, value in missing_output.get("aux_losses", {}).items():
        if not torch.is_tensor(value):
            value = reg * 0.0 + float(value)
        total = total + float(weights.get(name, 0.0)) * value
        pieces[name] = value.detach()
    pieces["total"] = total.detach()
    return total, pieces
