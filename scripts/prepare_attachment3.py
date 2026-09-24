from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def valid_lengths(features: np.ndarray, eps: float = 1e-12) -> list[int]:
    """Infer the valid span from the final nonzero row.

    Internal all-zero rows are retained as genuine local missing intervals.
    """

    lengths: list[int] = []
    for sample in features:
        nonzero = np.flatnonzero(np.linalg.norm(sample, axis=-1) > eps)
        lengths.append(int(nonzero[-1] + 1) if nonzero.size else 1)
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the 30 attachment-3 files and generate 50x768 BERT text features"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="data/attachment3_unaligned_50.pkl")
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--bert", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No pkl files found in {input_dir.resolve()}")

    raw_texts: list[str] = []
    audio_parts: list[np.ndarray] = []
    vision_parts: list[np.ndarray] = []
    ids: list[str] = []
    for file in files:
        with file.open("rb") as handle:
            payload = pickle.load(handle)
        split = payload["test"] if "test" in payload else payload
        raw = np.asarray(split["raw_text"]).reshape(-1)
        audio = np.asarray(split["audio"], dtype=np.float32)
        vision = np.asarray(split["vision"], dtype=np.float32)
        if len(raw) != len(audio) or len(raw) != len(vision):
            raise ValueError(f"Sample count mismatch in {file}")
        raw_texts.extend(str(x) for x in raw)
        audio_parts.append(audio)
        vision_parts.append(vision)
        ids.extend(f"{file.stem}_{i + 1:02d}" for i in range(len(raw)))

    audio = np.concatenate(audio_parts, axis=0)
    vision = np.concatenate(vision_parts, axis=0)
    audio[~np.isfinite(audio)] = 0.0
    vision[~np.isfinite(vision)] = 0.0

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.bert)
    model = AutoModel.from_pretrained(args.bert).to(device).eval()
    text_features, text_bert = [], []
    with torch.inference_mode():
        for start in range(0, len(raw_texts), args.batch_size):
            batch_text = raw_texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch_text,
                max_length=50,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            token_type_ids = encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"]))
            outputs = model(**{k: v.to(device) for k, v in encoded.items()})
            text_features.append(outputs.last_hidden_state.cpu().numpy().astype(np.float32))
            text_bert.append(
                torch.stack(
                    [encoded["input_ids"], encoded["attention_mask"], token_type_ids], dim=1
                ).cpu().numpy().astype(np.int64)
            )

    merged = {
        "test": {
            "id": ids,
            "raw_text": np.asarray(raw_texts),
            "text": np.concatenate(text_features, axis=0),
            "text_bert": np.concatenate(text_bert, axis=0),
            "audio": audio,
            "vision": vision,
            "audio_lengths": valid_lengths(audio),
            "vision_lengths": valid_lengths(vision),
        }
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Merged {len(files)} files and {len(raw_texts)} samples")
    print(f"text={merged['test']['text'].shape}, audio={audio.shape}, vision={vision.shape}")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
