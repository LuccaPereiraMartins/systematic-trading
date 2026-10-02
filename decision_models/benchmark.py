"""Compare Laya's 8-K triage labels with the dataset's current annotations."""

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import laya
import torch


MODEL = "convaiinnovations/laya"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
LABELS = ("routine", "review_worthy", "unclear")
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


def save(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def scores(rows):
    matrix = {label: {pred: 0 for pred in LABELS} for label in LABELS}
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
        "label_distribution": {label: sum(row["prediction"] == label for row in rows) for label in LABELS},
        "per_label": per_label,
        "confusion_matrix": matrix,
    }


def benchmark(dataset, output, device):
    records = json.loads(dataset.read_text(encoding="utf-8"))
    references = [record["human"]["label"] or record["llm"]["label"] for record in records]
    if any(label not in LABELS for label in references):
        raise ValueError("Every record needs a human or LLM label before benchmarking")

    warnings.filterwarnings("ignore", message=r"laya:.*invalid temperatures")
    agent = laya.load(MODEL, device=device, revision=MODEL_REVISION)
    rows = []
    started = time.perf_counter()
    for index, (record, reference) in enumerate(zip(records, references), 1):
        result = agent.predict_long(record["body"], QUESTIONS)["answers"]["triage"]
        prediction = result["choice"]
        if prediction not in LABELS:
            raise ValueError(f"Unexpected Laya label at record {index}: {prediction}")
        rows.append({
            "date": record["date"],
            "body_sha256": hashlib.sha256(record["body"].encode()).hexdigest(),
            "reference": reference,
            "reference_source": "human" if record["human"]["label"] is not None else "llm",
            "prediction": prediction,
            "probabilities": result["probabilities"],
            "confidence": result["confidence"],
            "answer_confidence": result["answer_confidence"],
            "windows": result.get("window", {}).get("count", 1),
        })
        if index % 25 == 0 or index == len(records):
            summary = scores(rows)
            save(output, {
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "device": str(agent.device),
                "comparison": "Agreement with human labels when present, otherwise current LLM labels; not ground-truth accuracy.",
                "confidence_note": "The checkpoint warned that some confidence values are uncalibrated.",
                "examples_completed": index,
                "examples_total": len(records),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
                "scores_so_far": summary,
                "predictions": rows,
            })
            print(f"{index}/{len(records)} | agreement {summary['agreement']:.3f} | macro F1 {summary['macro_f1']:.3f}", flush=True)

    final = scores(rows)
    save(output, {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "device": str(agent.device),
        "comparison": "Agreement with human labels when present, otherwise current LLM labels; not ground-truth accuracy.",
        "confidence_note": "The checkpoint warned that some confidence values are uncalibrated.",
        "examples_completed": len(records),
        "examples_total": len(records),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "scores": final,
        "predictions": rows,
    })
    print(json.dumps(final, indent=2))
    print(f"Saved {len(rows)} predictions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("dataset.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("laya_benchmark.json"))
    args = parser.parse_args()
    benchmark(args.dataset, args.output, args.device)
