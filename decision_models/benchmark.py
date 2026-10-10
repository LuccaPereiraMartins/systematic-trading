"""Evaluate base Laya, OpenAI models or fitted approaches on the frozen test split."""

import argparse
import asyncio
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

from schemas import (
    LABELS,
    LAYA_MODEL,
    LAYA_REVISION,
    LAYA_QUESTIONS,
    SPLITS,
    annotation,
    body_hash,
    load_split,
    save as save_json,
)


HERE = Path(__file__).resolve().parent


class SavedModel:
    """One inference interface for a saved neural head or a fitted linear baseline."""

    def __init__(self, directory, device="cuda"):
        import numpy as np
        import torch

        torch.set_num_threads(4)
        self.directory = directory
        self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if self.config.get("family", "").startswith("research_"):
            raise ValueError("Research runs require the study evaluator and final selection freeze")
        if self.config["labels"] != list(Benchmark.labels):
            raise ValueError("Saved model label order differs from benchmark")
        decision = directory / "decision.json"
        self.decision = (
            json.loads(decision.read_text(encoding="utf-8")) if decision.exists() else {"offsets": [0, 0, 0]}
        )
        if (
            decision.exists()
            and self.decision["validation_sha256"]
            != hashlib.sha256((directory / "validation.json").read_bytes()).hexdigest()
        ):
            raise ValueError("Validation predictions changed after decision tuning")
        self.device = device
        self.encoder = None
        self.linear = self.config.get("family") in ("linear", "baseline")
        self.blend = self.config.get("family") == "blend"
        self.artifact = directory / ("config.json" if self.blend else "model.joblib" if self.linear else "best.pt")
        if self.blend:
            self.model = [SavedModel(HERE / member["directory"], device) for member in self.config["members"]]
            for member, saved in zip(self.model, self.config["members"]):
                if hashlib.sha256(member.artifact.read_bytes()).hexdigest() != saved["artifact_sha256"]:
                    raise ValueError("Blend member changed after validation selection")
            if all(member.device == "cpu" for member in self.model):
                self.device = "cpu"
        elif self.linear:
            if self.config["kind"] not in ("finbert", "bge"):
                self.device = "cpu"
            import joblib

            self.model = joblib.load(directory / "model.joblib")
            self.order = [list(self.model.classes_).index(label) for label in Benchmark.labels]
            if self.config["kind"] in ("finbert", "bge"):
                from encoders import EncoderModel

                self.encoder = EncoderModel(device, config=self.config).eval()
        else:
            from encoders import load_model

            self.model = load_model(device, checkpoint=directory / "best.pt").eval()
        self.offsets = np.array(self.decision["offsets"])
        if (
            "artifact_sha256" in self.decision
            and self.decision["artifact_sha256"] != hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        ):
            raise ValueError("Model changed after decision tuning")

    def probabilities(self, body):
        import numpy as np
        import torch

        if self.blend:
            return np.average(
                [member.probabilities(body) for member in self.model], axis=0, weights=self.config["weights"]
            )
        with torch.no_grad():
            if not self.linear:
                logits, _ = self.model(self.model.windows(body))
                return logits.softmax(-1).cpu().numpy()
            if self.encoder:
                inputs = self.encoder.embed(body).cpu().numpy()[None]
            else:
                inputs = [body]
            return self.model.predict_proba(inputs)[0, self.order]


class Benchmark:
    labels = LABELS
    models = {"luna": "gpt-6-luna", "sol": "gpt-6.1-sol"}
    rates = {
        "gpt-6-luna": {"input": 0.05, "cached_input": 0.005, "output": 0.25},
        "gpt-6.1-sol": {"input": 1.00, "cached_input": 0.05, "output": 5.00},
    }
    concurrency = 4
    service_tier = "flex"
    reasoning_effort = "low"
    laya_model = LAYA_MODEL
    laya_revision = LAYA_REVISION
    questions = LAYA_QUESTIONS

    def __init__(self, device=None, splits=SPLITS):
        self.device = device
        self.split_manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
        if "prepared" in self.split_manifest:
            raise ValueError("Fresh research test inference requires the study's final selection freeze")
        self.sample = load_split(splits, "test")

    body_hash = staticmethod(body_hash)

    def reference(self, record):
        label = annotation(record)["label"]
        if label not in self.labels:
            raise ValueError("Every benchmark row needs a human or LLM label")
        return label

    @classmethod
    def scores(cls, rows):
        matrix = {actual: {predicted: 0 for predicted in cls.labels} for actual in cls.labels}
        for row in rows:
            matrix[row["reference"]][row["prediction"]] += 1
        per_label = {}
        for label in cls.labels:
            tp = matrix[label][label]
            predicted = sum(matrix[actual][label] for actual in cls.labels)
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
            "agreement": sum(matrix[label][label] for label in cls.labels) / len(rows),
            "macro_f1": sum(item["f1"] for item in per_label.values()) / len(cls.labels),
            "reference_distribution": {label: sum(matrix[label].values()) for label in cls.labels},
            "prediction_distribution": {
                label: sum(matrix[actual][label] for actual in cls.labels) for label in cls.labels
            },
            "per_label": per_label,
            "confusion_matrix": matrix,
        }

    def start(self, filename, **metadata):
        if self.split_manifest:
            filename += "-heldout"
        self.output = HERE / "benchmark_results" / f"{filename}.json"
        self.rows = []
        self.started = time.perf_counter()
        self.result = {
            **metadata,
            "split_manifest": self.split_manifest,
            "comparison": "Agreement with human labels when present, otherwise current LLM labels; not ground-truth accuracy.",
            "estimated_cost_per_run_usd": 0.0,
        }

    def record(self, record, prediction, latency, **details):
        if prediction not in self.labels:
            raise ValueError(f"Unexpected prediction: {prediction}")
        self.rows.append(
            {
                "date": record["date"],
                "body_sha256": self.body_hash(record),
                "reference": self.reference(record),
                "reference_source": "human" if record["human"]["label"] is not None else "llm",
                "prediction": prediction,
                "latency_seconds": latency,
                **details,
            }
        )
        # No await between recording and saving: successful concurrent responses stay saved.
        self.save()
        if len(self.rows) % 50 == 0 or len(self.rows) == len(self.sample):
            print(f"{self.result['model']}: {len(self.rows)}/{len(self.sample)}", flush=True)

    def save(self):
        self.result.update(
            completed=len(self.rows),
            total=len(self.sample),
            elapsed_seconds=time.perf_counter() - self.started,
            scores=self.scores(self.rows),
            predictions=self.rows,
        )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.result, self.output)

    def finish(self):
        latencies = sorted(row["latency_seconds"] for row in self.rows)
        self.result["latency_seconds"] = {
            "mean": sum(latencies) / len(latencies),
            "p50": latencies[round((len(latencies) - 1) * 0.50)],
            "p95": latencies[round((len(latencies) - 1) * 0.95)],
        }
        self.save()
        print(json.dumps({key: value for key, value in self.result.items() if key != "predictions"}, indent=2))
        print(f"Saved predictions to {self.output}")
        return self.result

    def run_laya(self):
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        import laya
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        warnings.filterwarnings("ignore", message=r"laya:.*invalid temperatures")
        before = time.perf_counter()
        agent = laya.load(self.laya_model, device=device, revision=self.laya_revision)
        self.start(
            f"laya-{device}",
            model=self.laya_model,
            model_revision=self.laya_revision,
            device=device,
            model_load_seconds=time.perf_counter() - before,
            confidence_note="The checkpoint warns that some confidence values are uncalibrated.",
        )
        agent.predict_long(self.sample[0]["body"], self.questions)  # Warm up before timing.
        for record in self.sample:
            if device == "cuda":
                torch.cuda.synchronize()
            before = time.perf_counter()
            answer = agent.predict_long(record["body"], self.questions)["answers"]["triage"]
            if device == "cuda":
                torch.cuda.synchronize()
            self.record(
                record,
                answer["choice"],
                time.perf_counter() - before,
                probabilities=answer["probabilities"],
                confidence=answer["confidence"],
                answer_confidence=answer["answer_confidence"],
                windows=answer.get("window", {}).get("count", 1),
            )
        return self.finish()

    async def run_openai(self, model):
        raise PermissionError(
            "Paid benchmarking is not currently authorised. Only Luna Flex corpus labeling may spend credits."
        )

    def run_saved(self, directory):
        import numpy as np
        import torch

        model = SavedModel(directory, self.device or "cuda")
        if self.split_manifest != model.config["split_manifest"]:
            raise ValueError("Saved model and benchmark use different splits")
        name = directory.name
        self.start(
            name,
            model=name,
            training_config=model.config,
            decision=model.decision,
            device=model.device,
            selected_epoch=getattr(model.model, "selected_epoch", None),
            artifact_sha256=hashlib.sha256(model.artifact.read_bytes()).hexdigest(),
            confidence_note="Decision offsets optimize validation macro F1; probabilities are uncalibrated.",
        )
        model.probabilities(self.sample[0]["body"])
        for row in self.sample:
            if model.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            probabilities = model.probabilities(row["body"])
            prediction = self.labels[(np.log(np.maximum(probabilities, 1e-12)) + model.offsets).argmax()]
            if model.device == "cuda":
                torch.cuda.synchronize()
            self.record(
                row,
                prediction,
                time.perf_counter() - started,
                probabilities=dict(zip(self.labels, map(float, probabilities))),
            )
        return self.finish()

    def run(self, models=("laya",)):
        """Run approaches sequentially; OpenAI uses four concurrent requests per model."""
        results = {}
        for name in models:
            if name in self.models:
                results[name] = asyncio.run(self.run_openai(self.models[name]))
            elif name == "laya":
                results[name] = self.run_laya()
            else:
                raise ValueError(f"Unknown benchmark approach: {name}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("laya", "luna", "sol"), default=("laya",))
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Laya device; defaults to CUDA when available")
    parser.add_argument("--splits", type=Path, default=SPLITS, help="Frozen JSONL split directory")
    parser.add_argument("--saved", type=Path, nargs="+", help="Evaluate these validation-selected run directories")
    args = parser.parse_args()
    benchmark = Benchmark(device=args.device, splits=args.splits)
    if args.saved:
        for directory in args.saved:
            benchmark.run_saved(directory)
    else:
        benchmark.run(args.models)
