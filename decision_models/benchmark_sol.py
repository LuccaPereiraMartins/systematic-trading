"""Benchmark GPT-6.1 Sol on the fixed 50-filing sample."""

import asyncio
from benchmark_openai import HERE, benchmark


if __name__ == "__main__":
    asyncio.run(benchmark("gpt-6.1-sol", HERE / "benchmark_results" / "gpt-6.1-sol.json"))
