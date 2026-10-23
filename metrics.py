
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

GROUPS = ["Overall", "1-label", "2-label", "3-label", ">=4-label", ">=2-label"]
PAPER_METRICS = ["Aiming", "Coverage", "Accuracy", "AT", "AF"]


def binarize(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= float(threshold)).astype(bool)


def sample_metrics(pred: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    labels = labels.astype(bool)
    intersection = np.logical_and(pred, labels).sum(axis=1).astype(np.float64)
    pred_count = pred.sum(axis=1).astype(np.float64)
    true_count = labels.sum(axis=1).astype(np.float64)
    union = np.logical_or(pred, labels).sum(axis=1).astype(np.float64)
    exact = np.all(pred == labels, axis=1)
    return {
        "N": int(labels.shape[0]),
        "Aiming": float(np.where(intersection > 0, intersection / np.maximum(pred_count, 1.0), 0.0).mean()),
        "Coverage": float(np.where(intersection > 0, intersection / np.maximum(true_count, 1.0), 0.0).mean()),
        "Accuracy": float(np.where(intersection > 0, intersection / np.maximum(union, 1.0), 0.0).mean()),
        "AT": float(exact.mean()),
        "AF": float(((union - intersection) / labels.shape[1]).mean()),
        "missed": float(np.logical_and(~pred, labels).sum(axis=1).mean()),
    }


def group_masks(labels: np.ndarray) -> Dict[str, np.ndarray]:
    label_count = labels.sum(axis=1).astype(int)
    return {
        "Overall": np.ones(labels.shape[0], dtype=bool),
        "1-label": label_count == 1,
        "2-label": label_count == 2,
        "3-label": label_count == 3,
        ">=4-label": label_count >= 4,
        ">=2-label": label_count >= 2,
    }


def metrics_by_group(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, Dict[str, float]]:
    pred = binarize(scores, threshold)
    return {name: sample_metrics(pred[mask], labels[mask]) for name, mask in group_masks(labels).items()}


def threshold_candidates() -> Iterable[float]:
    return [0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75]


def _threshold_key(metrics: Dict[str, Dict[str, float]], reference: Dict[str, Dict[str, float]]) -> Tuple:
    feasible = (
        metrics["Overall"]["Accuracy"] >= reference["Overall"]["Accuracy"] - 0.02
        and metrics["Overall"]["AT"] >= reference["Overall"]["AT"] - 0.02
        and metrics["1-label"]["AT"] >= reference["1-label"]["AT"] - 0.03
        and metrics["Overall"]["AF"] <= reference["Overall"]["AF"] + 0.005
    )
    return (
        int(feasible),
        metrics[">=2-label"]["AT"],
        metrics[">=2-label"]["Accuracy"],
        metrics[">=2-label"]["Coverage"],
        metrics["2-label"]["AT"],
        metrics["Overall"]["AT"],
        metrics["Overall"]["Accuracy"],
        -metrics["Overall"]["AF"],
        -metrics["Overall"]["missed"],
    )


def select_threshold(val_scores: np.ndarray, val_labels: np.ndarray, fixed_threshold: float = 0.6) -> float:
    reference = metrics_by_group(val_scores, val_labels, fixed_threshold)
    best_threshold = fixed_threshold
    best_key = None
    for threshold in threshold_candidates():
        current = metrics_by_group(val_scores, val_labels, threshold)
        key = _threshold_key(current, reference)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def result_rows(scores: np.ndarray, labels: np.ndarray, threshold: float) -> List[Dict[str, float]]:
    grouped = metrics_by_group(scores, labels, threshold)
    return [{"Group": group, **{name: grouped[group][name] for name in PAPER_METRICS}} for group in GROUPS]


def write_results(path: str | Path, rows: List[Dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Group", *PAPER_METRICS])
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows: List[Dict[str, float]], threshold: float) -> None:
    print(f"\nValidation-selected threshold: {threshold:.3f}")
    print("Group       Aiming  Coverage Accuracy AT      AF")
    for row in rows:
        print(
            f"{row['Group']:<11} "
            f"{row['Aiming']:.4f}  {row['Coverage']:.4f}   {row['Accuracy']:.4f}   "
            f"{row['AT']:.4f}  {row['AF']:.4f}"
        )
