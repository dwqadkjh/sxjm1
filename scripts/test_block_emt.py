"""Evaluate a trained Block-aware EMT on the held-out aligned_50 test split.

Run directly in PyCharm, or from the project root:
    python -m scripts.test_block_emt

The script uses the checkpoint's own data configuration so that an aligned
checkpoint is never silently evaluated against unaligned_50.pkl.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MultimodalPickleDataset
from src.engine import evaluate, resolve_device
from src.missing import MissingSpec
from src.models import build_model


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "block_emt_aligned" / "best_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "block_emt_aligned" / "test"
CLASS_NAMES = ("Negative", "Neutral", "Positive")
MODALITIES = ("text", "audio", "vision")


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def checked_dataset(
    data_path: Path,
    split: str,
    checkpoint: dict,
    corruption: str,
    seed: int,
    spec: MissingSpec | None = None,
) -> MultimodalPickleDataset:
    dataset = MultimodalPickleDataset(
        data_path,
        split,
        corruption=corruption,
        fixed_spec=spec,
        seed=seed,
        strides=checkpoint["config"].get("strides", {}),
    )
    expected_dims = checkpoint["feature_dims"]
    expected_lengths = checkpoint["sequence_lengths"]
    if any(dataset.feature_dims[m] != expected_dims[m] for m in MODALITIES) or any(
        dataset.sequence_lengths[m] != expected_lengths[m] for m in MODALITIES
    ):
        raise ValueError(
            "测试数据与权重的特征维度或序列长度不一致。"
            "block_emt_aligned 必须使用对应的 aligned_50.pkl。\n"
            f"权重: dims={expected_dims}, lengths={expected_lengths}\n"
            f"数据: dims={dataset.feature_dims}, lengths={dataset.sequence_lengths}"
        )
    if dataset.regression is None or dataset.classification is None:
        raise ValueError(f"{split!r} 划分没有完整标签，无法计算测试指标")
    return dataset


def save_clean_predictions(path: Path, predictions: dict) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "intensity_true", "intensity_pred", "class_true", "class_pred", "class_true_name", "class_pred_name"]
        )
        for sample_id, y, pred, cls, pred_cls in zip(
            predictions["id"],
            predictions["regression_true"],
            predictions["regression_pred"],
            predictions["classification_true"],
            predictions["classification_pred"],
        ):
            cls, pred_cls = int(cls), int(pred_cls)
            writer.writerow(
                [str(sample_id), float(y), float(pred), cls, pred_cls, CLASS_NAMES[cls], CLASS_NAMES[pred_cls]]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the trained block_emt model")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data", default=None, help="Default: data_path stored in the checkpoint")
    parser.add_argument("--split", default="test", help="Default: held-out test split")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--patterns", nargs="+", choices=["T", "A", "V", "TA", "TV", "AV", "TAV"],
                        default=["T", "A", "V", "TA", "TV", "AV", "TAV"])
    parser.add_argument("--rates", nargs="+", type=float, default=[0.3])
    parser.add_argument("--positions", nargs="+", choices=["early", "middle", "late"], default=["middle"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Evaluate a fixed random subset of this many samples")
    parser.add_argument("--sample-seed", type=int, default=20260924,
                        help="Seed used only for reproducible test-set sampling")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()

    checkpoint_path = project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到模型权重: {checkpoint_path}")
    if any(rate <= 0 or rate > 1 for rate in args.rates):
        parser.error("--rates 必须位于 (0, 1]，例如 0.3")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size 必须为正整数")

    device = resolve_device(args.device)
    # Only load checkpoints produced by this project or another trusted source.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") not in {"block_emt", "br_emt_dlfr"}:
        raise ValueError(f"权重中的模型是 {checkpoint.get('model_name')!r}，不是 block_emt")
    cfg = checkpoint["config"]
    data_value = args.data or cfg.get("data_path")
    if not data_value:
        raise ValueError("权重中没有 data_path，请用 --data 指定 aligned_50.pkl")
    data_path = project_path(data_value)
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到测试数据: {data_path}；请用 --data 指定 aligned_50.pkl")
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1111))

    model = build_model(
        checkpoint["model_name"],
        checkpoint["feature_dims"],
        checkpoint["sequence_lengths"],
        cfg.get("model_params", {}),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    batch_size = int(cfg.get("batch_size", 32))

    print(f"Model: {checkpoint['model_name']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_path} | split={args.split} | device={device}")
    rows = []
    conditions = [("NONE", 0.0, "none")]
    conditions.extend(
        (pattern, rate, position)
        for pattern in args.patterns
        for rate in args.rates
        for position in args.positions
    )
    for pattern, rate, position in conditions:
        spec = None if pattern == "NONE" else MissingSpec(pattern=pattern, rate=rate, position=position)
        dataset = checked_dataset(
            data_path, args.split, checkpoint,
            corruption="none" if spec is None else "fixed",
            seed=seed, spec=spec,
        )
        evaluation_dataset = dataset
        if args.sample_size is not None and args.sample_size < len(dataset):
            rng = np.random.default_rng(args.sample_seed)
            indices = np.sort(rng.choice(len(dataset), size=args.sample_size, replace=False))
            evaluation_dataset = Subset(dataset, indices.tolist())
        loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics, predictions = evaluate(model, loader, device)
        row = {
            "model": checkpoint["model_name"],
            "split": args.split,
            "samples": len(evaluation_dataset),
            "pattern": pattern,
            "rate": rate,
            "position": position,
            "seed": seed,
            **{key: metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")},
        }
        rows.append(row)
        print(
            f"{pattern:>4} rate={rate:.2f} position={position:<6} "
            f"Acc={row['accuracy']:.4f} F1={row['macro_f1']:.4f} "
            f"MAE={row['mae']:.4f} Corr={row['pearson']:.4f}"
        )
        if pattern == "NONE":
            save_clean_predictions(output_dir / "predictions_complete.csv", predictions)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {metrics_path}")
    print(f"Saved: {output_dir / 'predictions_complete.csv'}")


if __name__ == "__main__":
    main()
