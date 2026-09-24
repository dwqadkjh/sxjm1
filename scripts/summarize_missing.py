from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate factorial evaluation across random seeds")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/missing_factorial_summary.csv")
    args = parser.parse_args()
    groups = defaultdict(list)
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["pattern"], row["rate"], row["position"])
            groups[key].append({m: float(row[m]) for m in METRICS})

    output_rows = []
    for key, values in groups.items():
        row = dict(zip(("model", "pattern", "rate", "position"), key))
        for metric in METRICS:
            array = np.asarray([item[metric] for item in values])
            row[f"{metric}_mean"] = float(array.mean())
            row[f"{metric}_std"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        output_rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(output_rows[0]) if output_rows else []
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()

