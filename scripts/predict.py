from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import MultimodalPickleDataset
from src.engine import evaluate, load_trained_model, resolve_device
from src.metrics import regression_to_class


LABEL_NAMES = np.asarray(["Negative", "Neutral", "Positive"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict attachment 3 with an existing checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/attachment3_predictions.csv")
    parser.add_argument("--class-source", choices=["head", "regression"], default="head")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, checkpoint = load_trained_model(args.checkpoint, device)
    cfg = checkpoint["config"]
    dataset = MultimodalPickleDataset(
        args.data,
        args.split,
        corruption="natural",
        seed=int(cfg.get("seed", 1111)),
        strides=cfg.get("strides", {}),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 32)), shuffle=False)
    _, pred = evaluate(model, loader, device)
    if args.class_source == "regression":
        low, high = checkpoint.get("regression_class_thresholds", [-0.1, 0.1])
        classes = regression_to_class(pred["regression_pred"], low, high)
    else:
        classes = pred["classification_pred"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "intensity_pred", "polarity_pred", "polarity_index"])
        for sample_id, intensity, cls in zip(pred["id"], pred["regression_pred"], classes):
            writer.writerow([sample_id, f"{float(intensity):.6f}", LABEL_NAMES[int(cls)], int(cls)])
    print(f"Saved {len(classes)} predictions to {output.resolve()}")


if __name__ == "__main__":
    main()
