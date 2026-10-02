"""Benchmark Laya on the fixed 50-filing sample."""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import laya
import torch

from benchmark_common import HERE, LABELS, body_hash, load_sample, percentile, reference, save, scores


MODEL = "convaiinnovations/laya"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
QUESTIONS = {
    "triage": {
        "type": "choice",
        "instructions": "Classify this 8-K for whether an investment analyst should spend time reviewing it.",
        "criteria": {
            "routine": "Ordinary update with no apparent development requiring closer review.",
            "review_worthy": "A potentially significant development that merits closer review.",
            "unclear": "Insufficient or conflicting information to decide.",
        },
    }
}


def benchmark(device, output):
    _, sample, _ = load_sample()
    warnings.filterwarnings("ignore", message=r"laya:.*invalid temperatures")
    load_started = time.perf_counter()
    agent = laya.load(MODEL, device=device, revision=MODEL_REVISION)
    load_seconds = time.perf_counter() - load_started

    rows = []
    started = time.perf_counter()
    for index, record in enumerate(sample, 1):
        if device == "cuda":
            torch.cuda.synchronize()
        before = time.perf_counter()
        result = agent.predict_long(record["body"], QUESTIONS)["answers"]["triage"]
        if device == "cuda":
            torch.cuda.synchronize()
        prediction = result["choice"]
        if prediction not in LABELS:
            raise ValueError(f"Unexpected Laya label: {prediction}")
        rows.append({
            "date": record["date"],
            "body_sha256": body_hash(record),
            "reference": reference(record),
            "reference_source": "human" if record["human"]["label"] is not None else "llm",
            "prediction": prediction,
            "latency_seconds": time.perf_counter() - before,
            "probabilities": result["probabilities"],
            "confidence": result["confidence"],
            "answer_confidence": result["answer_confidence"],
            "windows": result.get("window", {}).get("count", 1),
        })
        if index % 10 == 0 or index == len(sample):
            save(output, {"model": MODEL, "device": device, "completed": index, "total": len(sample), "predictions": rows})
            print(f"Laya {device}: {index}/{len(sample)}", flush=True)

    elapsed = time.perf_counter() - started
    latencies = [row["latency_seconds"] for row in rows]
    result = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "device": device,
        "comparison": "Agreement with human labels when present, otherwise current LLM labels; not ground-truth accuracy.",
        "confidence_note": "The checkpoint warns that some confidence values are uncalibrated.",
        "model_load_seconds": load_seconds,
        "elapsed_seconds": elapsed,
        "latency_seconds": {"mean": sum(latencies) / len(latencies), "p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95)},
        "estimated_cost_per_run_usd": 0,
        "scores": scores(rows),
        "predictions": rows,
    }
    save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or HERE / "benchmark_results" / f"laya-{args.device}.json"
    benchmark(args.device, output)
