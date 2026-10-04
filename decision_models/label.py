"""Label unlabeled financial texts in dataset.json."""

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from schemas import FilingRecord, LabelOutput

DATASET = Path(__file__).with_name("dataset.json")
MODEL = "gpt-6-luna"
CONCURRENCY = 4
CHECKPOINT_EVERY = 10
RUBRIC = """Classify whether this financial text warrants an investment analyst's closer review.
routine: ordinary updates with no apparent development requiring closer review.
review_worthy: a potentially significant development that merits closer review.
unclear: insufficient or conflicting information to decide.
Uncertainty is your uncertainty about this label: 0.0 means certain, 1.0 means very uncertain.
Use only increments of 0.1. Judge the supplied text alone. Return only label and uncertainty."""


def annotation(record):
    return record["human"] if record["human"]["label"] is not None else record["llm"]


def save(records):
    temporary = DATASET.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(DATASET)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(1)


async def label_document(client, body):
    response = await client.responses.parse(
        model=MODEL,
        service_tier="flex",
        reasoning={"effort": "low"},
        instructions=RUBRIC,
        input=body,
        text_format=LabelOutput,
        max_output_tokens=256,
    )
    if response.status != "completed" or any(
        item.type == "refusal"
        for output in response.output
        for item in getattr(output, "content", [])
    ):
        raise ValueError(f"Model did not return a label: {response.status}")
    if response.output_parsed is None:
        raise ValueError("Model response did not match the label schema")
    return response.output_parsed.model_dump()


async def label_dataset():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    records = [
        FilingRecord.model_validate(record).model_dump()
        for record in json.loads(DATASET.read_text(encoding="utf-8"))
    ]
    pending = [i for i, record in enumerate(records) if annotation(record)["label"] is None]
    if not pending:
        print("No unlabeled records")
        return records
    client = AsyncOpenAI(timeout=900.0, max_retries=8)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def label_one(i):
        async with semaphore:
            body = records[i]["body"]
            records[i]["llm"] = await label_document(client, body)

    tasks = [asyncio.create_task(label_one(i)) for i in pending]
    completed = 0
    try:
        for future in asyncio.as_completed(tasks):
            await future
            completed += 1
            if completed % CHECKPOINT_EVERY == 0 or completed == len(pending):
                await asyncio.to_thread(save, records)
                print(f"Labeled {completed}/{len(pending)} remaining")
    except BaseException as exc:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(save, records)
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise RuntimeError(
            f"Labeling stopped after {completed} completed; progress saved: {exc}"
        ) from exc
    finally:
        await client.close()
    return records


if __name__ == "__main__":
    asyncio.run(label_dataset())
