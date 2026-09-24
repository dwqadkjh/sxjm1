from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def make_split(rng: np.random.Generator, name: str, n: int, dims: tuple[int, int, int]) -> dict:
    dt, da, dv = dims
    text = rng.normal(size=(n, 12, dt)).astype(np.float32)
    audio = rng.normal(size=(n, 20, da)).astype(np.float32)
    vision = rng.normal(size=(n, 16, dv)).astype(np.float32)
    audio_lengths = rng.integers(14, 21, size=n)
    vision_lengths = rng.integers(10, 17, size=n)
    for i in range(n):
        audio[i, audio_lengths[i] :] = 0
        vision[i, vision_lengths[i] :] = 0
    latent = 0.8 * text[:, :, 0].mean(1) + 0.3 * audio[:, :, 0].mean(1) + 0.2 * vision[:, :, 0].mean(1)
    regression = np.clip(2.0 * latent + rng.normal(scale=0.15, size=n), -3, 3).astype(np.float32)
    classification = np.where(regression < -0.15, 0, np.where(regression <= 0.15, 1, 2)).astype(np.int64)
    return {
        "id": np.asarray([f"{name}_{i:04d}" for i in range(n)]),
        "raw_text": np.asarray([f"demo sentence {i}" for i in range(n)]),
        "text": text,
        "audio": audio,
        "vision": vision,
        "audio_lengths": audio_lengths,
        "vision_lengths": vision_lengths,
        "regression_labels": regression.reshape(-1, 1),
        "classification_labels": classification.reshape(-1, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small synthetic pickle for code smoke tests")
    parser.add_argument("--output", default="data/demo.pkl")
    parser.add_argument("--attachment3-output", default="data/demo_attachment3.pkl")
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    payload = {
        "train": make_split(rng, "train", 72, (32, 12, 8)),
        "valid": make_split(rng, "valid", 24, (32, 12, 8)),
        "test": make_split(rng, "test", 24, (32, 12, 8)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle)

    attachment = {"test": {k: np.array(v, copy=True) for k, v in payload["test"].items() if "labels" not in k}}
    for i in range(len(attachment["test"]["id"])):
        attachment["test"]["text"][i, 4:7] = 0
        attachment["test"]["audio"][i, 7:13] = 0
        if i % 2 == 0:
            attachment["test"]["vision"][i, 6:10] = 0
    attachment_output = Path(args.attachment3_output)
    attachment_output.parent.mkdir(parents=True, exist_ok=True)
    with attachment_output.open("wb") as handle:
        pickle.dump(attachment, handle)
    print(f"Saved demo data: {output.resolve()}")
    print(f"Saved demo attachment 3: {attachment_output.resolve()}")


if __name__ == "__main__":
    main()

