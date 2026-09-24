from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make direct IDE execution independent of the configured working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, save_json
from src.data import MultimodalPickleDataset
from src.engine import (
    class_weights_from_labels,
    evaluate,
    resolve_device,
    seed_everything,
    train_one_epoch,
)
from src.metrics import selection_score, tune_regression_thresholds
from src.missing import MissingSpec
from src.models import build_model


def build_validation_conditions(cfg: dict) -> list[dict]:
    """Expand a validation attack matrix, with legacy single-condition fallback."""
    matrix = cfg.get("validation_matrix")
    if matrix is None:
        conditions = [
            dict(
                cfg.get(
                    "validation_missing",
                    {"pattern": "TAV", "rate": 0.3, "position": "middle"},
                )
            )
        ]
    else:
        patterns = list(matrix.get("patterns", []))
        rates = [float(value) for value in matrix.get("rates", [])]
        positions = list(matrix.get("positions", []))
        if not patterns or not rates or not positions:
            raise ValueError("validation_matrix requires non-empty patterns, rates, and positions")
        conditions = [
            {"pattern": pattern, "rate": rate, "position": position}
            for pattern, rate, position in product(patterns, rates, positions)
        ]
    extras = [dict(item) for item in cfg.get("validation_extra_conditions", [])]
    for item in extras:
        if not {"pattern", "rate", "position"}.issubset(item):
            raise ValueError(
                "each validation_extra_condition requires pattern, rate, and position"
            )
    return conditions + extras


def evaluate_validation_matrix(
    model: torch.nn.Module,
    dataset: MultimodalPickleDataset,
    loader: DataLoader,
    conditions: list[dict],
    device: torch.device,
    loss_weights: dict,
    class_weights: torch.Tensor,
) -> tuple[dict, dict, list[dict]]:
    """Evaluate every attack condition and return equal-weight condition means."""
    condition_rows: list[dict] = []
    pooled_predictions: dict[str, list[np.ndarray]] = {
        "regression_pred": [],
        "classification_true": [],
    }
    for condition in conditions:
        dataset.fixed_spec = MissingSpec(
            pattern=str(condition["pattern"]),
            rate=float(condition["rate"]),
            position=str(condition["position"]),
        )
        metrics, predictions = evaluate(model, loader, device, loss_weights, class_weights)
        condition_rows.append({**condition, **metrics})
        for key in pooled_predictions:
            pooled_predictions[key].append(predictions[key])

    metric_keys = tuple(metrics.keys())
    condition_weights = np.asarray(
        [float(row.get("selection_weight", 1.0)) for row in condition_rows],
        dtype=np.float64,
    )
    if np.any(condition_weights < 0.0) or condition_weights.sum() <= 0.0:
        raise ValueError("validation selection weights must be non-negative with positive sum")
    aggregate = {
        key: float(np.average(
            [float(row[key]) for row in condition_rows], weights=condition_weights
        ))
        for key in metric_keys
    }
    pooled = {key: np.concatenate(values) for key, values in pooled_predictions.items()}
    return aggregate, pooled, condition_rows


def effective_loss_weights(model_name: str, configured: dict) -> dict:
    result = dict(configured)
    key = model_name.lower().replace("-", "_")
    if key == "mult":
        result["reconstruction"] = 0.0
        result["consistency"] = 0.0
    elif key in {"tfr", "tfr_net", "tfrnet"}:
        result["consistency"] = 0.0
    elif key in {"misa", "misa_adapted"}:
        result["reconstruction"] = 0.0
        result["consistency"] = 0.0
        result["latent_reconstruction"] = 0.0
    return result


def resolve_data_path(value: str | None) -> Path:
    """Resolve an explicit path or locate attachment 2 for one-click execution."""
    candidates: list[Path] = []
    if value:
        supplied = Path(value).expanduser()
        candidates.append(supplied if supplied.is_absolute() else PROJECT_ROOT / supplied)
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "unaligned_50.pkl",
            PROJECT_ROOT.parent
            / "EQ"
            / "E题数据"
            / "E题数据"
            / "附件2-数据集特征文件"
            / "unaligned_50.pkl",
        ]
    )
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved
    checked = "\n".join(f"  - {path.resolve()}" for path in candidates)
    raise FileNotFoundError(f"Training data was not found. Checked:\n{checked}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MulT, TFR-Net, EMT-DLFR, Block-aware EMT, MISA, or LNLN"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "emt_dlfr_aligned_quick.json"))
    parser.add_argument("--data", default=None, help="Override data_path in config")
    parser.add_argument(
        "--model",
        choices=["mult", "tfr_net", "emt_dlfr", "block_emt", "br_emt_dlfr", "misa", "lnln"],
        default=None,
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = resolve_data_path(args.data or cfg.get("data_path"))
    model_name = args.model or cfg.get("model", "emt_dlfr")
    output_dir = Path(args.output or cfg.get("output_dir", f"outputs/{model_name}"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 1111))
    seed_everything(seed)
    device = resolve_device(str(cfg.get("device", "auto")))
    strides = cfg.get("strides", {})
    missing_cfg = cfg.get("missing", {})
    train_set = MultimodalPickleDataset(
        data_path,
        "train",
        corruption="random",
        missing_cfg=missing_cfg,
        seed=seed,
        strides=strides,
    )
    validation_conditions = build_validation_conditions(cfg)
    valid_set = MultimodalPickleDataset(
        data_path,
        "valid",
        corruption="fixed",
        missing_cfg=missing_cfg,
        fixed_spec=MissingSpec(**validation_conditions[0]),
        seed=seed + 17,
        strides=strides,
    )
    batch_size = int(cfg.get("batch_size", 32))
    workers = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = build_model(
        model_name,
        train_set.feature_dims,
        train_set.sequence_lengths,
        cfg.get("model_params", {}),
    ).to(device)
    loss_weights = effective_loss_weights(model_name, cfg.get("loss_weights", {}))
    class_weights = class_weights_from_labels(train_set.classification, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    epochs = int(cfg.get("epochs", 30))
    patience = int(cfg.get("early_stopping_patience", 7))
    best_score, stale = float("-inf"), 0
    history = []
    validation_history = []
    best_path = output_dir / "best_model.pt"

    print(
        f"Validation selection uses {len(validation_conditions)} attack conditions "
        "with configured condition weights."
    )

    for epoch in range(1, epochs + 1):
        train_set.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            class_weights,
            float(cfg.get("grad_clip", 1.0)),
        )
        valid_metrics, valid_predictions, condition_rows = evaluate_validation_matrix(
            model,
            valid_set,
            valid_loader,
            validation_conditions,
            device,
            loss_weights,
            class_weights,
        )
        validation_history.extend({"epoch": epoch, **item} for item in condition_rows)
        score = selection_score(valid_metrics, cfg.get("selection_metric_weights"))
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_loss.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
            "selection_score": score,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={train_loss['total']:.4f} "
            f"F1={valid_metrics['macro_f1']:.4f} MAE={valid_metrics['mae']:.4f} "
            f"Corr={valid_metrics['pearson']:.4f} score={score:.4f}"
        )
        if score > best_score:
            low, high, threshold_f1 = tune_regression_thresholds(
                valid_predictions["regression_pred"], valid_predictions["classification_true"]
            )
            best_score, stale = score, 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "feature_dims": train_set.feature_dims,
                    "sequence_lengths": train_set.sequence_lengths,
                    "validation_metrics": valid_metrics,
                    "validation_conditions": validation_conditions,
                    "validation_condition_metrics": condition_rows,
                    "regression_class_thresholds": [low, high],
                    "threshold_macro_f1": threshold_f1,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    keys = sorted({key for row in history for key in row})
    with (output_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    validation_keys = sorted({key for row in validation_history for key in row})
    with (output_dir / "validation_matrix_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=validation_keys)
        writer.writeheader()
        writer.writerows(validation_history)
    save_json(
        {
            "model": model_name,
            "best_score": best_score,
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint": str(best_path.resolve()),
            "validation_conditions": validation_conditions,
        },
        output_dir / "run_summary.json",
    )
    print(f"Saved: {best_path.resolve()}")


if __name__ == "__main__":
    main()
