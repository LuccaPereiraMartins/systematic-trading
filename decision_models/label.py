"""Label unlabeled financial texts in dataset.json."""

import argparse
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from schemas import DATASET, FilingRecord, LabelOutput, annotation, save


MODEL = "gpt-6-luna"
CONCURRENCY = 4
CHECKPOINT_EVERY = 250
TOKENS_PER_MINUTE = 180_000
RUBRIC = """Classify whether this financial text warrants an investment analyst's closer review.
routine: ordinary updates with no apparent development requiring closer review.
review_worthy: a potentially significant development that merits closer review.
unclear: insufficient or conflicting information to decide.
Uncertainty is your uncertainty about this label: 0.0 means certain, 1.0 means very uncertain.
Use only increments of 0.1. Judge the supplied text alone. Return only label and uncertainty."""


async def label_document(client, body):
    response = await client.responses.parse(
        model=MODEL,
        service_tier="flex",
        reasoning={"effort": "low"},
        instructions=RUBRIC,
        input=body,
        text_format=LabelOutput,
        max_output_tokens=1024,
    )
    if response.status != "completed" or any(
        item.type == "refusal" for output in response.output for item in getattr(output, "content", [])
    ):
        raise ValueError(f"Model did not return a label: {response.status}")
    if response.output_parsed is None:
        raise ValueError("Model response did not match the label schema")
    return response.output_parsed.model_dump()


async def label_dataset(dataset=DATASET):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    records = [
        FilingRecord.model_validate(record).model_dump() for record in json.loads(dataset.read_text(encoding="utf-8"))
    ]
    pending = [i for i, record in enumerate(records) if annotation(record)["label"] is None]
    if not pending:
        print("No unlabeled records")
        return records
    client = AsyncOpenAI(timeout=900.0, max_retries=2)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    pace_lock = asyncio.Lock()
    next_request = 0.0

    async def label_one(i):
        nonlocal next_request
        async with semaphore:
            body = records[i]["body"]
            while True:
                # Approximate tokens from text length and leave headroom under the API TPM limit.
                async with pace_lock:
                    loop = asyncio.get_running_loop()
                    delay = max(0, next_request - loop.time())
                    if delay:
                        await asyncio.sleep(delay)
                    next_request = loop.time() + (len(body) / 3 + 512) / (TOKENS_PER_MINUTE / 60)
                try:
                    records[i]["llm"] = await label_document(client, body)
                    return
                except RateLimitError as exc:
                    message = str(exc)
                    if "flex_unavailable" not in message and "rate_limit_exceeded" not in message:
                        raise
                    delay = 60 if "flex_unavailable" in message else 10
                    print(f"API capacity limited; retrying filing {i} in {delay}s", flush=True)
                    await asyncio.sleep(delay)

    tasks = [asyncio.create_task(label_one(i)) for i in pending]
    completed = 0
    try:
        for future in asyncio.as_completed(tasks):
            await future
            completed += 1
            if completed % CHECKPOINT_EVERY == 0 or completed == len(pending):
                await asyncio.to_thread(save, records, dataset)
                print(f"Labeled {completed}/{len(pending)} remaining")
    except BaseException as exc:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(save, records, dataset)
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise RuntimeError(f"Labeling stopped after {completed} completed; progress saved: {exc}") from exc
    finally:
        await client.close()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    asyncio.run(label_dataset(args.dataset))
