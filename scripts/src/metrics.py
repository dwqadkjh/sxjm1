from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


def _f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return macro and support-weighted F1 without a scikit-learn dependency."""
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    labels = np.unique(np.concatenate((true, pred)))
    per_class: list[float] = []
    supports: list[int] = []
    for label in labels:
        tp = int(np.sum((true == label) & (pred == label)))
        fp = int(np.sum((true != label) & (pred == label)))
        fn = int(np.sum((true == label) & (pred != label)))
        denominator = 2 * tp + fp + fn
        per_class.append(2.0 * tp / denominator if denominator else 0.0)
        supports.append(int(np.sum(true == label)))
    if not per_class:
        return 0.0, 0.0
    macro = float(np.mean(per_class))
    total = sum(supports)
    weighted = float(np.average(per_class, weights=supports)) if total else 0.0
    return macro, weighted


def sentiment_metrics(
    regression_true: np.ndarray,
    regression_pred: np.ndarray,
    classification_true: np.ndarray,
    classification_pred: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(regression_true).reshape(-1)
    p = np.asarray(regression_pred).reshape(-1)
    c = np.asarray(classification_true).reshape(-1)
    cp = np.asarray(classification_pred).reshape(-1)
    macro_f1, weighted_f1 = _f1_scores(c, cp)
    corr = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else 0.0
    return {
        "accuracy": float(np.mean(c == cp)) if len(c) else 0.0,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mae": float(np.mean(np.abs(y - p))) if len(y) else 0.0,
        "pearson": corr,
    }


def selection_score(
    metrics: Dict[str, float], weights: Mapping[str, float] | None = None
) -> float:
    """Validation score; optional normalized weights include accuracy explicitly."""
    if weights is None:
        return metrics["macro_f1"] + metrics["pearson"] - metrics["mae"] / 3.0
    components = {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "pearson": (float(np.clip(metrics["pearson"], -1.0, 1.0)) + 1.0) / 2.0,
        "mae": 1.0 - float(np.clip(metrics["mae"], 0.0, 6.0)) / 6.0,
    }
    selected = {name: float(weights.get(name, 0.0)) for name in components}
    if any(value < 0.0 for value in selected.values()) or sum(selected.values()) <= 0.0:
        raise ValueError("selection metric weights must be non-negative with positive sum")
    return sum(selected[name] * components[name] for name in components) / sum(
        selected.values()
    )


def regression_to_class(regression_pred: np.ndarray, low: float, high: float) -> np.ndarray:
    p = np.asarray(regression_pred)
    return np.where(p < low, 0, np.where(p <= high, 1, 2)).astype(np.int64)


def tune_regression_thresholds(
    regression_pred: np.ndarray,
    classification_true: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float, float]:
    grid = np.asarray(grid if grid is not None else np.linspace(-0.8, 0.8, 33))
    best = (-0.1, 0.1, -1.0)
    for low in grid:
        for high in grid:
            if low >= high:
                continue
            pred = regression_to_class(regression_pred, float(low), float(high))
            score, _ = _f1_scores(classification_true, pred)
            if score > best[2]:
                best = (float(low), float(high), float(score))
    return best
