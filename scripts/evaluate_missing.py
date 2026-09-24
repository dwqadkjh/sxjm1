from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Support direct execution from PyCharm without configuring a working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MultimodalPickleDataset
from src.engine import evaluate, resolve_device
from src.missing import MissingSpec
from src.models import build_model


def find_default_checkpoint() -> Path:
    preferred = PROJECT_ROOT / "outputs" / "quick_emt_dlfr" / "best_model.pt"
    if preferred.is_file():
        return preferred
    checkpoints = sorted(
        (PROJECT_ROOT / "outputs").glob("**/best_model.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if checkpoints:
        return checkpoints[0]
    raise FileNotFoundError(
        f"No best_model.pt was found below {PROJECT_ROOT / 'outputs'}. Train a model first."
    )


def find_default_data() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "unaligned_50.pkl",
        PROJECT_ROOT.parent
        / "EQ"
        / "E题数据"
        / "E题数据"
        / "附件2-数据集特征文件"
        / "unaligned_50.pkl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Attachment 2 was not found. Checked:\n{checked}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-factor missing-segment robustness evaluation")
    parser.add_argument("--checkpoint", default=None, help="Defaults to quick_emt_dlfr/best_model.pt")
    parser.add_argument("--data", default=None, help="Defaults to attachment 2 unaligned_50.pkl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--patterns", nargs="+", default=["T", "A", "V", "TA", "TV", "AV", "TAV"])
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--positions", nargs="+", default=["early", "middle", "late"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[111, 1111, 11111])
    parser.add_argument("--output", default="outputs/missing_factorial_results.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else find_default_checkpoint()
    data_path = Path(args.data).expanduser().resolve() if args.data else find_default_data()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_path}")
    print(f"Output: {output}")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(
        checkpoint["model_name"],
        checkpoint["feature_dims"],
        checkpoint["sequence_lengths"],
        cfg.get("model_params", {}),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    rows = []
    conditions = [("NONE", 0.0, "none")]
    conditions += [
        (pattern, rate, position)
        for pattern in args.patterns
        for rate in args.rates
        for position in args.positions
    ]
    for pattern, rate, position in conditions:
        for seed in args.seeds:
            corruption = "none" if rate == 0 else "fixed"
            spec = None if rate == 0 else MissingSpec(pattern=pattern, rate=rate, position=position)
            dataset = MultimodalPickleDataset(
                data_path,
                args.split,
                corruption=corruption,
                fixed_spec=spec,
                seed=seed,
                strides=cfg.get("strides", {}),
            )
            loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 32)), shuffle=False)
            metrics, _ = evaluate(model, loader, device)
            row = {
                "model": checkpoint["model_name"],
                "pattern": pattern,
                "rate": rate,
                "position": position,
                "seed": seed,
                **metrics,
            }
            rows.append(row)
            print(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    keys = ["model", "pattern", "rate", "position", "seed", "accuracy", "macro_f1", "weighted_f1", "mae", "pearson", "loss"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
