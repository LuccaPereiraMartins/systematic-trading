"""Collect primary 8-K text into dataset.json."""

import argparse
import asyncio
import json
import os
import re
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


DATASET = Path(__file__).with_name("dataset.json")
RATE = 10


def key(record):
    return record["date"], re.sub(r"\s+", " ", record["body"]).strip().casefold()


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


def latest_filings(number, start, end):
    os.environ["EDGAR_RATE_LIMIT_PER_SEC"] = str(RATE)
    from edgar import get_filings, set_identity

    agent = os.getenv("SEC_USER_AGENT")
    if not agent:
        raise ValueError("Set SEC_USER_AGENT in the root .env file")
    set_identity(agent)
    filing_date = f"{start}:{end}" if start else None
    filings = get_filings(form="8-K", amendments=False, filing_date=filing_date) if filing_date else get_filings(form="8-K", amendments=False)
    if number is None:
        return list(filings)
    latest = filings.latest(number)
    return [latest] if number == 1 else list(latest)


async def collect_text(filings, records):
    limit = asyncio.Semaphore(RATE)
    seen = {key(record) for record in records}

    async def fetch(filing):
        async with limit:
            try:
                body = await asyncio.to_thread(filing.text)
                return filing, body.strip(), None
            except Exception as exc:
                return filing, "", exc

    tasks = [asyncio.create_task(fetch(filing)) for filing in filings]
    added = failed = 0
    for task in asyncio.as_completed(tasks):
        filing, body, error = await task
        if error or len(body) < 100:
            failed += 1
            print(f"Filing failed: {filing.accession_number}: {error or 'no readable text'}")
            continue
        record = {"date": str(filing.filing_date), "body": body,
                  "llm": {"label": None, "uncertainty": None},
                  "human": {"label": None, "uncertainty": None}}
        if key(record) not in seen:
            records.append(record)
            seen.add(key(record))
            added += 1
            if added % 25 == 0:
                records.sort(key=lambda item: (item["date"], key(item)[1]), reverse=True)
                save(records)
    records.sort(key=lambda item: (item["date"], key(item)[1]), reverse=True)
    save(records)
    print(f"Added {added}, failed {failed}; dataset has {len(records)} records")
    return records


def collect_8ks(number=None, start=None, end=None):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if (start is None) != (end is None):
        raise ValueError("Supply both start and end dates")
    start = date.fromisoformat(start) if isinstance(start, str) else start
    end = date.fromisoformat(end) if isinstance(end, str) else end
    if start and start > end:
        raise ValueError("start must be on or before end")
    if number is not None and number < 1:
        raise ValueError("number must be positive")
    if number is None and start is None:
        number = 50
    records = json.loads(DATASET.read_text(encoding="utf-8")) if DATASET.exists() else []
    filings = latest_filings(number, start, end)
    return asyncio.run(collect_text(filings, records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int)
    parser.add_argument("--start", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive YYYY-MM-DD")
    collect_8ks(**vars(parser.parse_args()))
