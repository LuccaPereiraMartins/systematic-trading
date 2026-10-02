"""Run one OpenAI model over the fixed 50-filing sample."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from benchmark_common import HERE, LABELS, body_hash, load_sample, percentile, reference, save, scores
from label import RUBRIC
from schemas import LabelOutput


RATES_PER_MILLION = {
    "gpt-6-luna": {"input": 0.05, "cached_input": 0.005, "output": 0.25},
    "gpt-6.1-sol": {"input": 1.00, "cached_input": 0.05, "output": 5.00},
}
CONCURRENCY = 4


def cost(model, usage):
    cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
    ordinary = usage.input_tokens - cached
    rates = RATES_PER_MILLION[model]
    return (ordinary * rates["input"] + cached * rates["cached_input"] + usage.output_tokens * rates["output"]) / 1_000_000


async def benchmark(model, output):
    load_dotenv(HERE.parent / ".env")
    _, sample, _ = load_sample()
    client = AsyncOpenAI(timeout=900.0, max_retries=8)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    rows = []
    tokens = {"input": 0, "cached_input": 0, "output": 0}
    estimated_cost = 0.0
    started = time.perf_counter()

    async def run(record):
        nonlocal estimated_cost
        async with semaphore:
            before = time.perf_counter()
            response = await client.responses.parse(
                model=model,
                service_tier="flex",
                reasoning={"effort": "low"},
                instructions=RUBRIC,
                input=record["body"],
                text_format=LabelOutput,
                max_output_tokens=256,
            )
            latency = time.perf_counter() - before
            if response.status != "completed" or any(
                item.type == "refusal" for output_item in response.output
                for item in getattr(output_item, "content", [])
            ):
                raise ValueError(f"{model} did not return a label: {response.status}")
            if response.output_parsed is None or response.output_parsed.label not in LABELS:
                raise ValueError(f"{model} returned an invalid structured label")
            usage = response.usage
            cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
            tokens["input"] += usage.input_tokens
            tokens["cached_input"] += cached
            tokens["output"] += usage.output_tokens
            estimated_cost += cost(model, usage)
            rows.append({
                "date": record["date"],
                "body_sha256": body_hash(record),
                "reference": reference(record),
                "reference_source": "human" if record["human"]["label"] is not None else "llm",
                "prediction": response.output_parsed.label,
                "uncertainty": response.output_parsed.uncertainty,
                "latency_seconds": round(latency, 4),
            })
            summary = scores(rows)
            save(output, {
                "model": model,
                "service_tier": "flex",
                "reasoning_effort": "low",
                "comparison": "Agreement with human labels when present, otherwise current LLM labels; not ground-truth accuracy.",
                "completed": len(rows),
                "total": len(sample),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "estimated_cost_per_run_usd": estimated_cost,
                "tokens": tokens,
                "scores": summary,
                "predictions": rows,
            })
            print(f"{model}: {len(rows)}/{len(sample)} | agreement {summary['agreement']:.3f} | cost ${estimated_cost:.4f}", flush=True)

    tasks = [asyncio.create_task(run(record)) for record in sample]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        await client.close()

    elapsed = time.perf_counter() - started
    latencies = [row["latency_seconds"] for row in rows]
    result = {
        "model": model,
        "service_tier": "flex",
        "reasoning_effort": "low",
        "comparison": "Agreement with human labels when present, otherwise current LLM labels; not ground-truth accuracy.",
        "elapsed_seconds": round(elapsed, 2),
        "latency_seconds": {"mean": sum(latencies) / len(latencies), "p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95)},
        "estimated_cost_per_run_usd": estimated_cost,
        "cost_note": "Estimate for this 50-filing run from API token usage and short-context Flex rates; verify against account billing.",
        "pricing_usd_per_million_tokens": RATES_PER_MILLION[model],
        "tokens": tokens,
        "scores": scores(rows),
        "predictions": rows,
    }
    save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(RATES_PER_MILLION))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or HERE / "benchmark_results" / f"{args.model}.json"
    asyncio.run(benchmark(args.model, output))
