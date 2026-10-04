"""Compare Laya, OpenAI models and TF-IDF on the same 50 filings."""

import argparse
import asyncio
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

from label import RUBRIC, annotation
from schemas import LabelOutput

HERE = Path(__file__).resolve().parent


class Benchmark:
    labels = ("routine", "review_worthy", "unclear")
    models = {"luna": "gpt-6-luna", "sol": "gpt-6.1-sol"}
    rates = {
        "gpt-6-luna": {"input": 0.05, "cached_input": 0.005, "output": 0.25},
        "gpt-6.1-sol": {"input": 1.00, "cached_input": 0.05, "output": 5.00},
    }
    baselines = (
        "majority", "length_raw", "length_item", "keyword_prior", "keyword_learned", "bge_lr", "finbert_lr", "finbert_ft"
    )
    concurrency = 4
    service_tier = "flex"
    reasoning_effort = "low"
    laya_model = "convaiinnovations/laya"
    laya_revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    questions = {
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

    def __init__(self, device=None):
        self.device = device
        self.records = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
        hashes = json.loads((HERE / "benchmark_sample.json").read_text(encoding="utf-8"))["body_sha256s"]
        if len(hashes) != 50 or len(set(hashes)) != 50:
            raise ValueError("Benchmark sample must contain 50 unique filings")
        by_hash = {self.body_hash(record): record for record in self.records}
        try:
            self.sample = [by_hash[key] for key in hashes]
        except KeyError as exc:
            raise ValueError(f"Sample filing missing from dataset: {exc}") from exc
        sample_hashes = set(hashes)
        self.train = [record for record in self.records if self.body_hash(record) not in sample_hashes]

    @staticmethod
    def body_hash(record):
        return hashlib.sha256(record["body"].encode()).hexdigest()

    def reference(self, record):
        label = annotation(record)["label"]
        if label not in self.labels:
            raise ValueError("Every benchmark row needs a human or LLM label")
        return label

    def scores(self, rows):
        matrix = {actual: {predicted: 0 for predicted in self.labels} for actual in self.labels}
        for row in rows:
            matrix[row["reference"]][row["prediction"]] += 1
        per_label = {}
        for label in self.labels:
            tp = matrix[label][label]
            predicted = sum(matrix[actual][label] for actual in self.labels)
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
            "agreement": sum(matrix[label][label] for label in self.labels) / len(rows),
            "macro_f1": sum(item["f1"] for item in per_label.values()) / len(self.labels),
            "reference_distribution": {label: sum(matrix[label].values()) for label in self.labels},
            "prediction_distribution": {
                label: sum(matrix[actual][label] for actual in self.labels) for label in self.labels
            },
            "per_label": per_label,
            "confusion_matrix": matrix,
        }

    def start(self, filename, **metadata):
        self.output = HERE / "benchmark_results" / f"{filename}.json"
        self.rows = []
        self.started = time.perf_counter()
        self.result = {
            **metadata,
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
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output)

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
        from dotenv import load_dotenv
        from openai import AsyncOpenAI

        load_dotenv(HERE.parent / ".env")
        self.start(
            model,
            model=model,
            service_tier=self.service_tier,
            reasoning_effort=self.reasoning_effort,
            pricing_usd_per_million_tokens=self.rates[model],
            cost_note="Estimate for this 50-filing run from API token usage and short-context Flex rates; verify against account billing.",
            tokens={"input": 0, "cached_input": 0, "output": 0},
        )
        semaphore = asyncio.Semaphore(self.concurrency)
        client = AsyncOpenAI(timeout=900.0, max_retries=8)

        async def predict(record):
            async with semaphore:
                before = time.perf_counter()
                response = await client.responses.parse(
                    model=model,
                    service_tier=self.service_tier,
                    reasoning={"effort": self.reasoning_effort},
                    instructions=RUBRIC,
                    input=record["body"],
                    text_format=LabelOutput,
                    max_output_tokens=256,
                )
                latency = time.perf_counter() - before
                if response.status != "completed" or any(
                    item.type == "refusal" for output in response.output for item in getattr(output, "content", [])
                ):
                    raise ValueError(f"{model} did not return a label: {response.status}")
                answer = response.output_parsed
                if answer is None or answer.label not in self.labels:
                    raise ValueError(f"{model} returned an invalid structured label")
                usage = response.usage
                cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
                tokens = {"input": usage.input_tokens, "cached_input": cached, "output": usage.output_tokens}
                for key, value in tokens.items():
                    self.result["tokens"][key] += value
                rates = self.rates[model]
                self.result["estimated_cost_per_run_usd"] += (
                    (tokens["input"] - cached) * rates["input"]
                    + cached * rates["cached_input"]
                    + tokens["output"] * rates["output"]
                ) / 1_000_000
                self.record(record, answer.label, latency, uncertainty=answer.uncertainty)

        tasks = [asyncio.create_task(predict(record)) for record in self.sample]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            await client.close()
        return self.finish()

    def run_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.start(
            "tfidf_logistic_regression",
            model="TF-IDF word uni/bi-grams + balanced logistic regression",
            training_examples=len(self.train),
            test_examples=len(self.sample),
        )
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        before = time.perf_counter()
        model.fit([record["body"] for record in self.train], [self.reference(record) for record in self.train])
        self.result["fit_seconds"] = time.perf_counter() - before
        for record in self.sample:
            before = time.perf_counter()
            prediction = model.predict([record["body"]])[0]
            self.record(record, prediction, time.perf_counter() - before)
        return self.finish()

    def run_baseline(self, name):
        """Fit a baselines.py approach on the 450 non-sample filings and predict the 50 pilot filings one by one."""
        from baselines import APPROACHES, make

        self.start(name, model=APPROACHES[name][0], training_examples=len(self.train), test_examples=len(self.sample))
        model = make(name)
        before = time.perf_counter()
        model.fit([record["body"] for record in self.train], [self.reference(record) for record in self.train])
        self.result["fit_seconds"] = time.perf_counter() - before
        for record in self.sample:
            before = time.perf_counter()
            prediction = model.predict([record["body"]])[0]
            self.record(record, prediction, time.perf_counter() - before)
        return self.finish()

    def run(self, models=("laya", "luna", "sol", "tfidf")):
        """Run approaches sequentially; OpenAI uses four concurrent requests per model."""
        results = {}
        for name in models:
            if name in self.models:
                results[name] = asyncio.run(self.run_openai(self.models[name]))
            elif name == "laya":
                results[name] = self.run_laya()
            elif name == "tfidf":
                results[name] = self.run_tfidf()
            elif name in self.baselines:
                results[name] = self.run_baseline(name)
            else:
                raise ValueError(f"Unknown benchmark approach: {name}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("laya", "luna", "sol", "tfidf", *Benchmark.baselines),
        default=("laya", "luna", "sol", "tfidf"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Laya device; defaults to CUDA when available")
    args = parser.parse_args()
    Benchmark(device=args.device).run(args.models)
