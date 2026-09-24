from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all three models with one configuration")
    parser.add_argument("--config", default="configs/quick.json")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-root", default="outputs/three_models")
    args = parser.parse_args()
    root = Path(args.output_root)
    for model in ("mult", "tfr_net", "emt_dlfr"):
        command = [
            sys.executable,
            "-m",
            "scripts.train",
            "--config",
            args.config,
            "--model",
            model,
            "--output",
            str(root / model),
        ]
        if args.data:
            command += ["--data", args.data]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

