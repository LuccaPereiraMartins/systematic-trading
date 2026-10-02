"""Shared helpers for the small, repeatable benchmark runs."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.json"
SAMPLE = HERE / "benchmark_sample.json"
LABELS = ("routine", "review_worthy", "unclear")


def body_hash(record):
    return hashlib.sha256(record["body"].encode()).hexdigest()


def reference(record):
    annotation = record["human"] if record["human"]["label"] is not None else record["llm"]
    if annotation["label"] not in LABELS:
        raise ValueError("Every benchmark row needs a human or LLM label")
    return annotation["label"]


def load_sample():
    records = json.loads(DATASET.read_text(encoding="utf-8"))
    by_hash = {body_hash(record): record for record in records}
    hashes = json.loads(SAMPLE.read_text(encoding="utf-8"))["body_sha256s"]
    if len(hashes) != 50 or len(set(hashes)) != 50:
        raise ValueError("Benchmark sample must contain 50 unique filings")
    try:
        sample = [by_hash[key] for key in hashes]
    except KeyError as exc:
        raise ValueError(f"Sample filing missing from dataset: {exc}") from exc
    return records, sample, set(hashes)


def scores(rows):
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for row in rows:
        matrix[row["reference"]][row["prediction"]] += 1
    per_label = {}
    for label in LABELS:
        tp = matrix[label][label]
        predicted = sum(matrix[actual][label] for actual in LABELS)
        support = sum(matrix[label].values())
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "support": support,
        }
    return {
        "agreement": sum(matrix[label][label] for label in LABELS) / len(rows),
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(LABELS),
        "reference_distribution": {label: sum(row["reference"] == label for row in rows) for label in LABELS},
        "prediction_distribution": {label: sum(row["prediction"] == label for row in rows) for label in LABELS},
        "per_label": per_label,
        "confusion_matrix": matrix,
    }


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]
