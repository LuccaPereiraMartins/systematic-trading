"""Shared research metrics, calibration, discard policies and artifact fingerprints."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from prepare import family, sample
from schemas import LABELS, annotation, body_hash, load_split, save


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def targets(records):
    return np.array([LABELS.index(annotation(row)["label"]) for row in records])


def probabilities(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != len(LABELS) or not np.isfinite(logits).all():
        raise ValueError("Expected finite three-class logits")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    return softmax(logits / temperature, axis=1)


def temperature(logits, truth):
    """One scalar, selected by calibration-set multiclass negative log likelihood."""
    def objective(log_t):
        p = probabilities(logits, np.exp(log_t))
        return -np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-12)).mean()
    result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    if not result.success:
        raise ValueError("Temperature optimization failed")
    candidates = (0.0, float(result.x), -3.0, 3.0)
    chosen = min(candidates, key=objective)
    return float(np.exp(chosen))


def discard(probability, threshold):
    # A routine threshold cannot silently discard predicted unclear/review-worthy documents.
    p = np.asarray(probability)
    return (p.argmax(1) == 0) & (p[:, 0] >= threshold)


def policy_scores(truth, discarded):
    truth, discarded = np.asarray(truth), np.asarray(discarded, dtype=bool)
    dangerous = int(((truth == 1) & discarded).sum())
    worthwhile = int((truth == 1).sum())
    n_discard = int(discarded.sum())
    return {"discarded": n_discard, "workload_reduction": n_discard / len(truth),
            "review_worthy_support": worthwhile, "dangerous_misses": dangerous,
            "dangerous_miss_rate": dangerous / worthwhile if worthwhile else None,
            "discard_contamination": float(((truth != 0) & discarded).sum() / n_discard) if n_discard else None,
            "unclear_discarded": int(((truth == 2) & discarded).sum())}


def fit_policies(p, truth):
    policies = {}
    for rate in (0.01, 0.05):
        best = {"threshold": 1.000001, **policy_scores(truth, np.zeros(len(truth), dtype=bool))}
        if (truth == 1).any():
            for threshold in sorted(set(p[:, 0]), reverse=True):
                score = policy_scores(truth, discard(p, threshold))
                if score["dangerous_miss_rate"] <= rate and score["discarded"] > best["discarded"]:
                    best = {"threshold": float(threshold), **score}
        policies[str(rate)] = {**best, "target_miss_rate": rate,
                               "criterion": "Empirical calibration miss rate; no population guarantee"}
    return policies


def prediction_rows(records, logits, decision=None):
    decision = decision or {"temperature": 1.0, "policies": {}}
    p = probabilities(logits, decision["temperature"])
    rows = []
    for record, vector in zip(records, p, strict=True):
        row = {"body_sha256": body_hash(record), "group_id": record.get("group_id", body_hash(record)),
               "date": record["date"], "source": record.get("source", "sec"), "family": family(record),
               "document_type": record.get("document_type", "8-K"), "characters": len(record["body"]),
               "reference": annotation(record)["label"],
               "reference_source": "human" if record["human"]["label"] is not None else "teacher",
               "teacher_uncertainty": record["llm"].get("uncertainty"),
               "prediction": LABELS[vector.argmax()], "probabilities": vector.tolist(),
               "uncertainty": float(1 - vector.max()),
               "discard": {key: bool(vector.argmax() == 0 and vector[0] >= value["threshold"])
                           for key, value in decision["policies"].items()}}
        rows.append(row)
    return rows


def metrics(rows):
    if not rows:
        return {"count": 0}
    truth = np.array([LABELS.index(r["reference"]) for r in rows])
    predicted = np.array([LABELS.index(r["prediction"]) for r in rows])
    p = np.array([r["probabilities"] for r in rows])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1, 2], zero_division=0,
    )
    confidence = p.max(1)
    bins, ece = [], 0.0
    for index in range(10):
        mask = (confidence >= index / 10) & ((confidence < (index + 1) / 10) if index < 9 else (confidence <= 1))
        accuracy = float((predicted[mask] == truth[mask]).mean()) if mask.any() else None
        mean_confidence = float(confidence[mask].mean()) if mask.any() else None
        if mask.any():
            ece += float(mask.mean()) * abs(accuracy - mean_confidence)
        bins.append({"lower": index / 10, "upper": (index + 1) / 10, "count": int(mask.sum()),
                     "accuracy": accuracy, "confidence": mean_confidence})
    default_policy = policy_scores(truth, predicted == 0)
    return {"count": len(rows), "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro")),
            "accuracy": float(accuracy_score(truth, predicted)),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(LABELS)},
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
            "default_discard": default_policy,
            "brier": float(((p - np.eye(3)[truth]) ** 2).sum(1).mean()),
            "log_loss": float(-np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-12)).mean()),
            "ece_10_bins": ece, "calibration_bins": bins,
            "policies": {key: policy_scores(truth, [r["discard"][key] for r in rows])
                         for key in rows[0].get("discard", {})}}


def breakdown(rows):
    return {"overall": metrics(rows),
            "by_family": {key: metrics([r for r in rows if r["family"] == key])
                          for key in sorted({r["family"] for r in rows})},
            "by_source": {key: metrics([r for r in rows if r["source"] == key])
                          for key in sorted({r["source"] for r in rows})},
            "by_reference_source": {key: metrics([r for r in rows if r["reference_source"] == key])
                                    for key in sorted({r["reference_source"] for r in rows})}}


def calibrate(records, logits, output):
    truth = targets(records)
    t = temperature(logits, truth)
    p = probabilities(logits, t)
    decision = {"temperature": t, "method": "Single temperature on separate calibration groups",
                "policies": fit_policies(p, truth), "calibration_documents": len(records),
                "calibration_classes": dict(Counter(LABELS[i] for i in truth)),
                "calibration_sha256": hashlib.sha256(json.dumps([
                    (body_hash(r), annotation(r)["label"]) for r in records]).encode()).hexdigest()}
    rows = prediction_rows(records, logits, decision)
    save(decision, output / "decision.json")
    save({"metrics": breakdown(rows), "predictions": rows, "evaluation": "Calibration fitting data"},
         output / "calibration.json")
    return decision


def training_subset(splits, number):
    rows = load_split(splits, "train")
    frozen = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    if frozen.get("subsets_sha256") and fingerprint(splits / "subsets.json") != frozen["subsets_sha256"]:
        raise ValueError("Frozen training subsets changed")
    manifest = json.loads((splits / "subsets.json").read_text(encoding="utf-8"))
    number = len(rows) if number is None else number
    identities = manifest["subsets"].get(str(number))
    if identities is None:
        raise ValueError(f"No frozen training subset of {number} documents")
    by_hash = {body_hash(r): r for r in rows}
    return [by_hash[key] for key in identities]


def speed(predict, records, synchronize=lambda: None):
    """Warmed full inference; persist the exact sample, batch definitions and measurements."""
    records = sample(records, 128)
    texts = [r["body"] for r in records]
    predict(texts[:1])
    synchronize()
    latencies = []
    for text in texts:
        synchronize()
        start = time.perf_counter()
        predict([text])
        synchronize()
        latencies.append(time.perf_counter() - start)
    throughput = {}
    for batch in (1, 8):
        synchronize()
        start = time.perf_counter()
        for offset in range(0, len(texts), batch):
            predict(texts[offset:offset + batch])
        synchronize()
        elapsed = time.perf_counter() - start
        throughput[str(batch)] = {"documents_per_second": len(texts) / elapsed, "seconds": elapsed}
    return {"documents": len(records), "sample": [body_hash(r) for r in records],
            "latency_seconds": {"p50": float(np.quantile(latencies, .5)), "p95": float(np.quantile(latencies, .95)),
                                "mean": float(np.mean(latencies)), "measurements": latencies},
            "batch_throughput": throughput,
            "measurement": "Warm inference incl. text processing; excludes loading; shared local machine"}
