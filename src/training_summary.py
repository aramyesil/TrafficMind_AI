"""Read the saved training log. This does not recompute or invent metrics."""

from __future__ import annotations

import csv
from pathlib import Path


def best_validation_metrics(results_csv: Path) -> dict[str, float | int]:
    with results_csv.open(encoding="utf-8", newline="") as handle:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows in {results_csv}")
    metric = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row[metric]))
    return {
        "epoch": int(float(best["epoch"])),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best[metric]),
    }
