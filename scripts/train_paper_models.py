"""Train the three literature baselines selected from Ye Mingyang's thesis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "misa": PROJECT_ROOT / "configs" / "paper_misa_quick.json",
    "tfr_net": PROJECT_ROOT / "configs" / "paper_tfr_quick.json",
    "lnln": PROJECT_ROOT / "configs" / "paper_lnln_quick.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MISA, TFR-Net, and LNLN adaptations")
    parser.add_argument(
        "--models", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS)
    )
    parser.add_argument("--data", default=None, help="Optional attachment-2 pickle path")
    parser.add_argument("--output-root", default="outputs/paper_three_models")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    for model_name in args.models:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config",
            str(CONFIGS[model_name]),
            "--model",
            model_name,
            "--output",
            str(output_root / model_name),
        ]
        if args.data:
            command.extend(["--data", args.data])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
