from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MultimodalPickleDataset
from src.engine import load_trained_model, move_batch, resolve_device
from src.explain import (
    adaptive_alpha,
    build_attachment4_payload,
    explain_sample,
    mp4_duration,
    top_windows,
)
from src.models.common import MODALITIES


LABELS = ("Negative", "Neutral", "Positive")


def load_tokenizer(name: str, allow_download: bool):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, local_files_only=not allow_download)
    except Exception:
        return None


def text_evidence(
    scores: np.ndarray,
    text_bert: list,
    raw_text: str,
    tokenizer,
    count: int,
) -> list[dict]:
    token_ids = np.asarray(text_bert)[0] if np.asarray(text_bert).ndim == 2 else np.asarray([])
    words = raw_text.split()
    valid = [i for i in np.argsort(scores)[::-1] if scores[i] > 0]
    evidence = []
    for position in valid[:count]:
        token_id = int(token_ids[position]) if position < len(token_ids) else None
        if tokenizer is not None and token_id is not None:
            token = tokenizer.convert_ids_to_tokens(token_id)
        elif 0 < position <= len(words):
            token = words[position - 1]
        else:
            token = f"token_id:{token_id}" if token_id is not None else f"step:{position}"
        evidence.append(
            {"position": int(position), "token": str(token), "score": float(scores[position])}
        )
    return evidence


def build_evidence(detail: dict, metadata: dict, raw_text: str, tokenizer, top_k: int, window: int) -> dict:
    matrix = np.asarray(detail["contribution"]["time_modality"], dtype=np.float64)
    duration_error = None
    try:
        duration = mp4_duration(metadata["video_path"])
    except Exception as exc:
        duration, duration_error = None, str(exc)
    length = len(matrix)
    audio = []
    for start, end, score in top_windows(matrix[:, 1].tolist(), window, 3):
        audio.append(
            {
                "start_step": start,
                "end_step": end,
                "start_seconds": duration * start / length if duration is not None else None,
                "end_seconds": duration * end / length if duration is not None else None,
                "score": score,
            }
        )
    vision = []
    for position in np.argsort(matrix[:, 2])[::-1][:3]:
        vision.append(
            {
                "position": int(position),
                "seconds": duration * (position + 0.5) / length if duration is not None else None,
                "score": float(matrix[position, 2]),
                "video_path": metadata["video_path"],
            }
        )
    return {
        "raw_text": raw_text,
        "text": text_evidence(matrix[:, 0], metadata["text_bert"], raw_text, tokenizer, top_k),
        "audio": audio,
        "vision": vision,
        "video_duration_seconds": duration,
        "duration_error": duration_error,
    }


def svg_bars(values: dict[str, float]) -> str:
    colors = {"text": "#2563eb", "audio": "#16a34a", "vision": "#dc2626"}
    rows = []
    for index, name in enumerate(MODALITIES):
        value, y = float(values[name]), 10 + index * 30
        rows.append(
            f'<text x="0" y="{y + 14}" font-size="12">{name.title()}</text>'
            f'<rect x="58" y="{y}" width="{260 * value:.1f}" height="18" fill="{colors[name]}"/>'
            f'<text x="325" y="{y + 14}" font-size="12">{value:.1%}</text>'
        )
    return '<svg viewBox="0 0 390 100" role="img">' + "".join(rows) + "</svg>"


def svg_timeline(values: list[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    peak = max(float(array.max()), 1e-12)
    points = " ".join(
        f"{10 + i * 620 / max(len(array) - 1, 1):.1f},{105 - 90 * value / peak:.1f}"
        for i, value in enumerate(array)
    )
    return (
        '<svg viewBox="0 0 640 120" role="img">'
        '<line x1="10" y1="105" x2="630" y2="105" stroke="#94a3b8"/>'
        f'<polyline points="{points}" fill="none" stroke="#7c3aed" stroke-width="2"/>'
        '<text x="10" y="118" font-size="10">0</text><text x="615" y="118" font-size="10">49</text>'
        "</svg>"
    )


def fmt_seconds(value) -> str:
    return "unknown" if value is None else f"{float(value):.2f}s"


def render_report(samples: list[dict], aggregate: dict, checkpoint: Path, input_dir: Path) -> str:
    cards = []
    for item in samples:
        prediction = item["prediction"]
        contribution = item["contribution"]
        evidence = item["evidence"]
        text_items = ", ".join(
            f'{html.escape(entry["token"])} (t={entry["position"]})' for entry in evidence["text"]
        ) or "No positive text evidence"
        audio_items = ", ".join(
            f'{fmt_seconds(entry["start_seconds"])}–{fmt_seconds(entry["end_seconds"])}'
            for entry in evidence["audio"]
        )
        vision_items = ", ".join(fmt_seconds(entry["seconds"]) for entry in evidence["vision"])
        faithful = "PASS" if item["faithfulness"]["top_gt_bottom"] else "CHECK"
        stability = item["faithfulness"]["stability_spearman"]
        cards.append(
            f'<section><h2>Sample {html.escape(item["id"])}</h2>'
            f'<p><b>Prediction:</b> {LABELS[prediction["polarity_index"]]} | '
            f'Intensity {prediction["intensity"]:.4f} | α={item["alpha"]:.4f}</p>'
            '<p class="note">Reliability is the model gate; contribution is measured by occlusion.</p>'
            f'{svg_bars(contribution["modality_share"])}'
            f'<h3>Temporal contribution</h3>{svg_timeline(contribution["time_share"])}'
            f'<p><b>Text evidence:</b> {text_items}</p>'
            f'<p><b>Audio intervals:</b> {audio_items}</p>'
            f'<p><b>Vision timestamps:</b> {vision_items}</p>'
            f'<p><b>Faithfulness:</b> {faithful}; top={item["faithfulness"]["top_impact"]:.6f}, '
            f'bottom={item["faithfulness"]["bottom_impact"]:.6f}; '
            f'stability={"n/a" if stability is None else f"{stability:.4f}"}</p></section>'
        )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Problem 3 explanations</title>
<style>body{{font:14px/1.5 system-ui;margin:24px;max-width:1050px;color:#172033}}section{{border:1px solid #dbe3ee;border-radius:10px;padding:16px;margin:18px 0}}h1,h2,h3{{color:#123b64}}.note{{color:#64748b}}code{{word-break:break-all}}</style></head><body>
<h1>Attachment 4 Explainable Emotion Prediction</h1>
<p>Checkpoint: <code>{html.escape(str(checkpoint))}</code><br>Data: <code>{html.escape(str(input_dir))}</code></p>
<p>Samples: {aggregate['samples']} | Faithfulness pass rate: {aggregate['faithfulness_pass_rate']:.1%} | Mean stability: {aggregate['mean_stability_spearman']:.4f}</p>
{''.join(cards)}</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain aligned attachment-4 samples with Block-EMT")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="outputs/problem3_attachment4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--faithfulness-ratio", type=float, default=0.2)
    parser.add_argument("--stability-repeats", type=int, default=3)
    parser.add_argument("--noise-ratio", type=float, default=0.01)
    parser.add_argument("--occlusion-batch-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, checkpoint = load_trained_model(checkpoint_path, device)
    if checkpoint.get("model_name") not in {"block_emt", "br_emt_dlfr"}:
        raise ValueError("Problem-3 explanations currently require a Block-EMT checkpoint")
    payload, metadata = build_attachment4_payload(input_dir)
    cfg = checkpoint["config"]
    dataset = MultimodalPickleDataset(
        payload, "test", corruption="natural", seed=int(cfg.get("seed", 1111)),
        strides=cfg.get("strides", {}),
    )
    if dataset.feature_dims != checkpoint["feature_dims"] or dataset.sequence_lengths != checkpoint["sequence_lengths"]:
        raise ValueError(
            f"Attachment 4 and checkpoint shapes differ: data={dataset.feature_dims}/{dataset.sequence_lengths}, "
            f"checkpoint={checkpoint['feature_dims']}/{checkpoint['sequence_lengths']}"
        )
    alpha = adaptive_alpha(checkpoint.get("validation_metrics", {}))
    tokenizer = load_tokenizer(args.tokenizer, args.allow_tokenizer_download)
    samples = []
    for raw_batch in DataLoader(dataset, batch_size=1, shuffle=False):
        sample_id = str(raw_batch["id"][0])
        detail = explain_sample(
            model, move_batch(raw_batch, device), alpha,
            faithfulness_ratio=args.faithfulness_ratio,
            stability_repeats=args.stability_repeats,
            noise_ratio=args.noise_ratio,
            seed=int(cfg.get("seed", 1111)) + len(samples) * 1009,
            chunk_size=args.occlusion_batch_size,
        )
        detail["id"] = sample_id
        detail["evidence"] = build_evidence(
            detail, metadata[sample_id], str(raw_batch["raw_text"][0]), tokenizer,
            args.top_k, args.window,
        )
        detail["source_file"] = metadata[sample_id]["source_file"]
        samples.append(detail)
        print(f"Explained {sample_id} ({len(samples)}/{len(dataset)})")

    stability_values = [x["faithfulness"]["stability_spearman"] for x in samples]
    stability_values = [x for x in stability_values if x is not None]
    aggregate = {
        "samples": len(samples),
        "faithfulness_pass_rate": float(np.mean([x["faithfulness"]["top_gt_bottom"] for x in samples])),
        "mean_top_impact": float(np.mean([x["faithfulness"]["top_impact"] for x in samples])),
        "mean_bottom_impact": float(np.mean([x["faithfulness"]["bottom_impact"] for x in samples])),
        "mean_stability_spearman": float(np.mean(stability_values)) if stability_values else 0.0,
    }
    details = {
        "run": {
            "checkpoint": str(checkpoint_path),
            "input_dir": str(input_dir),
            "alpha": alpha,
            "validation_metrics": checkpoint.get("validation_metrics", {}),
            "definitions": {
                "reliability": "Block-EMT gate weights; not causal contribution.",
                "contribution": "Prediction sensitivity measured by feature occlusion.",
            },
        },
        "aggregate": aggregate,
        "samples": samples,
    }
    with (output_dir / "details.json").open("w", encoding="utf-8") as handle:
        json.dump(details, handle, ensure_ascii=False, indent=2)

    fields = [
        "id", "polarity", "polarity_index", "intensity", "alpha",
        "contribution_text", "contribution_audio", "contribution_vision",
        "reliability_text", "reliability_audio", "reliability_vision",
        "text_evidence", "audio_intervals", "vision_timestamps",
        "top_impact", "bottom_impact", "top_gt_bottom", "stability_spearman", "uninformative",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in samples:
            pred, rel, con, ev, faith = (
                item["prediction"], item["reliability"]["modality_mean"],
                item["contribution"], item["evidence"], item["faithfulness"],
            )
            writer.writerow({
                "id": item["id"], "polarity": LABELS[pred["polarity_index"]],
                "polarity_index": pred["polarity_index"], "intensity": pred["intensity"], "alpha": item["alpha"],
                **{f"contribution_{m}": con["modality_share"][m] for m in MODALITIES},
                **{f"reliability_{m}": rel[m] for m in MODALITIES},
                "text_evidence": "; ".join(x["token"] for x in ev["text"]),
                "audio_intervals": "; ".join(
                    f'{fmt_seconds(x["start_seconds"])}-{fmt_seconds(x["end_seconds"])}' for x in ev["audio"]
                ),
                "vision_timestamps": "; ".join(fmt_seconds(x["seconds"]) for x in ev["vision"]),
                "top_impact": faith["top_impact"], "bottom_impact": faith["bottom_impact"],
                "top_gt_bottom": faith["top_gt_bottom"], "stability_spearman": faith["stability_spearman"],
                "uninformative": con["uninformative"],
            })
    (output_dir / "report.html").write_text(
        render_report(samples, aggregate, checkpoint_path, input_dir), encoding="utf-8"
    )
    print(f"Saved {len(samples)} explanations to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
