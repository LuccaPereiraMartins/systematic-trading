"""Collect primary 8-K text into dataset.json."""

import argparse
import asyncio
import hashlib
import json
import os
import re
from datetime import date
from itertools import zip_longest
from pathlib import Path

from dotenv import load_dotenv
from schemas import DATASET, save


RATE = 10


def key(record):
    return record["date"], re.sub(r"\s+", " ", record["body"]).strip().casefold()


def latest_filings(number, start, end, sample=False):
    os.environ["EDGAR_RATE_LIMIT_PER_SEC"] = str(RATE)
    from edgar import get_filings, set_identity

    agent = os.getenv("SEC_USER_AGENT")
    if not agent:
        raise ValueError("Set SEC_USER_AGENT in the root .env file")
    set_identity(agent)
    filing_date = f"{start}:{end}" if start else None
    filings = (
        get_filings(form="8-K", amendments=False, filing_date=filing_date)
        if filing_date
        else get_filings(form="8-K", amendments=False)
    )
    if sample:
        months = {}
        for filing in filings:
            months.setdefault(str(filing.filing_date)[:7], []).append(filing)
        # Hash ordering gives a repeatable selection without favoring companies or dates.
        groups = [
            sorted(months[month], key=lambda filing: hashlib.sha256(filing.accession_number.encode()).hexdigest())
            for month in sorted(months)
        ]
        return [filing for row in zip_longest(*groups) for filing in row if filing is not None]
    if number is None:
        return list(filings)
    latest = filings.latest(number)
    return [latest] if number == 1 else list(latest)


async def collect_text(filings, records, output=DATASET):
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
        record = {
            "date": str(filing.filing_date),
            "body": body,
            "llm": {"label": None, "uncertainty": None},
            "human": {"label": None, "uncertainty": None},
        }
        if key(record) not in seen:
            records.append(record)
            seen.add(key(record))
            added += 1
            if added % 100 == 0:
                records.sort(key=lambda item: (item["date"], key(item)[1]), reverse=True)
                save(records, output)
                print(f"Collected {len(records)} records", flush=True)
    records.sort(key=lambda item: (item["date"], key(item)[1]), reverse=True)
    save(records, output)
    print(f"Added {added}, failed {failed}; dataset has {len(records)} records")
    return records


def collect_8ks(number=None, start=None, end=None, sample=False, output=DATASET):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if (start is None) != (end is None):
        raise ValueError("Supply both start and end dates")
    start = date.fromisoformat(start) if isinstance(start, str) else start
    end = date.fromisoformat(end) if isinstance(end, str) else end
    if start and start > end:
        raise ValueError("start must be on or before end")
    if number is not None and number < 1:
        raise ValueError("number must be positive")
    if sample and (start is None or number is None):
        raise ValueError("Monthly sampling requires a date range and number")
    if number is None and start is None:
        number = 50
    output = Path(output)
    records = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    if sample and len(records) >= number:
        print(f"Already have {len(records)} records in {output}")
        return records
    filings = latest_filings(number, start, end, sample)
    if not sample:
        return asyncio.run(collect_text(filings, records, output))
    # Continue through the monthly selection to replace failed or duplicate bodies.
    for offset in range(0, len(filings), 1000):
        remaining = number - len(records)
        if remaining <= 0:
            break
        batch = filings[offset : offset + min(1000, remaining)]
        asyncio.run(collect_text(batch, records, output))
    if len(records) < number:
        raise RuntimeError(f"Only collected {len(records)}/{number}; rerun to retry")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int)
    parser.add_argument("--start", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive YYYY-MM-DD")
    parser.add_argument(
        "--sample", action="store_true", help="Sample evenly across months; number is the target dataset size"
    )
    parser.add_argument("--output", type=Path, default=DATASET)
    collect_8ks(**vars(parser.parse_args()))
