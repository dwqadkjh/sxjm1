from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow this file to be launched directly from an IDE, regardless of the
# configured working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import summarize_pickle


def find_default_data() -> Path:
    """Locate attachment 2 automatically for one-click IDE execution."""
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
    raise FileNotFoundError(
        "No data file was supplied and unaligned_50.pkl was not found.\n"
        f"Checked:\n{checked}\n"
        "Pass it manually with: --data <path>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the supplied multimodal pickle")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to a dataset pickle. If omitted, attachment 2 is located automatically.",
    )
    args = parser.parse_args()
    data_path = Path(args.data).expanduser().resolve() if args.data else find_default_data()
    print(f"Data: {data_path}")
    print(json.dumps(summarize_pickle(data_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
