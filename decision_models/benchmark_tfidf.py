"""Benchmark TF-IDF plus logistic regression on the fixed 50-filing sample."""

import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from benchmark_common import HERE, body_hash, load_sample, percentile, reference, save, scores


_, sample, sample_hashes = load_sample()
records = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
train = [record for record in records if body_hash(record) not in sample_hashes]
started = time.perf_counter()
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000),
                      LogisticRegression(max_iter=1000, class_weight="balanced"))
model.fit([record["body"] for record in train], [reference(record) for record in train])
fit_seconds = time.perf_counter() - started

rows = []
latencies = []
for record in sample:
    before = time.perf_counter()
    prediction = model.predict([record["body"]])[0]
    latencies.append(time.perf_counter() - before)
    rows.append({
        "date": record["date"],
        "body_sha256": body_hash(record),
        "reference": reference(record),
        "reference_source": "human" if record["human"]["label"] is not None else "llm",
        "prediction": prediction,
    })

result = {
    "model": "TF-IDF word uni/bi-grams + balanced logistic regression",
    "training_examples": len(train),
    "test_examples": len(sample),
    "comparison": "Agreement with human labels when present, otherwise current LLM labels; not ground-truth accuracy.",
    "fit_seconds": fit_seconds,
    "elapsed_seconds": time.perf_counter() - started,
    "latency_seconds": {"mean": sum(latencies) / len(latencies), "p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95)},
    "estimated_cost_per_run_usd": 0,
    "scores": scores(rows),
    "predictions": rows,
}
output = HERE / "benchmark_results" / "tfidf_logistic_regression.json"
save(output, result)
print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))
print(f"Saved predictions to {output}")
