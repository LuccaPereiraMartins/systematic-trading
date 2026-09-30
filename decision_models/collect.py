"""Collect main 8-K documents from SEC quarterly indexes."""

import argparse
import json
import os
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATASET = Path(__file__).with_name("dataset.json")
SEC = "https://www.sec.gov"


def save(records):
    temporary = DATASET.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(DATASET)


def key(record):
    return record["date"], re.sub(r"\s+", " ", record["body"]).strip().casefold()


def quarters_between(start, end):
    year, quarter = end.year, (end.month - 1) // 3 + 1
    while (year, quarter) >= (start.year, (start.month - 1) // 3 + 1):
        yield year, quarter
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4


def fetch(session, url):
    time.sleep(0.25)  # SEC asks automated clients to stay below 10 requests/s.
    response = session.get(url, timeout=45)
    response.raise_for_status()
    return response


def main_document(session, filename):
    index_url = f"{SEC}/Archives/{filename.removesuffix('.txt')}-index.html"
    page = BeautifulSoup(fetch(session, index_url).text, "html.parser")
    for row in page.select("table.tableFile tr"):
        cells = row.find_all("td")
        if len(cells) >= 4 and cells[3].get_text(strip=True) == "8-K":
            link = cells[2].find("a", href=True)
            if link:
                url = urljoin(index_url, link["href"])
                if urlparse(url).path == "/ix":
                    url = urljoin(SEC, parse_qs(urlparse(url).query)["doc"][0])
                return url
    raise ValueError(f"No main 8-K document in {index_url}")


def filing_text(session, filename):
    url = main_document(session, filename)
    response = fetch(session, url)
    if url.lower().endswith((".htm", ".html")):
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup.find_all(True):
            if tag.attrs is None:
                continue
            if tag.name in {"script", "style", "noscript", "head", "ix:header"} or tag.has_attr("hidden") or "display:none" in tag.get("style", "").replace(" ", "").lower():
                tag.decompose()
        for tag in soup.find_all("br"):
            tag.replace_with("\n")
        for tag in soup.find_all(["p", "div", "tr", "h1", "h2", "h3"]):
            tag.insert_after("\n")
        body = soup.get_text(" ")
    else:
        body = response.text
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" *\n *", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) < 100:
        raise ValueError(f"Main document has too little readable text: {url}")
    return body


def collect_8ks(number=None, start=None, end=None):
    load_dotenv(Path(__file__).with_name(".env"))
    agent = os.getenv("SEC_USER_AGENT")
    if not agent:
        raise ValueError("Set SEC_USER_AGENT in .env to your app name and contact email")
    if (start is None) != (end is None):
        raise ValueError("Supply both --start and --end, or neither")
    start = date.fromisoformat(start) if isinstance(start, str) else start
    end = date.fromisoformat(end) if isinstance(end, str) else end
    if start and start > end:
        raise ValueError("--start must be on or before --end")
    if number is None and start is None:
        number = 50
    if number is not None and number < 1:
        raise ValueError("--number must be positive")

    records = json.loads(DATASET.read_text(encoding="utf-8")) if DATASET.exists() else []
    seen = {key(record) for record in records}
    session = requests.Session()
    session.headers["User-Agent"] = agent
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    today = date.today()
    quarters = quarters_between(start or date(today.year - 2, 1, 1), end or today)
    filings = []
    for year, quarter in quarters:
        url = f"{SEC}/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
        try:
            index = fetch(session, url).text
        except requests.RequestException as exc:
            print(f"Index failed: {url}: {exc}")
            continue
        for line in index.splitlines():
            parts = line.split("|", 4)
            if len(parts) == 5 and parts[2] == "8-K":
                filed = date.fromisoformat(parts[3])
                if (start is None or start <= filed <= end):
                    filings.append((filed.isoformat(), parts[4]))
        if start is None and len(filings) >= number:
            break
    filings.sort(key=lambda item: (-date.fromisoformat(item[0]).toordinal(), item[1]))

    selected = added = failed = 0
    for filed, filename in filings:
        if number is not None and selected >= number:
            break
        try:
            record = {"date": filed, "body": filing_text(session, filename),
                      "llm": {"label": None, "uncertainty": None},
                      "human": {"label": None, "uncertainty": None}}
            identity = key(record)
            if identity not in seen:
                records.append(record)
                seen.add(identity)
                save(records)
                added += 1
            selected += 1
        except (requests.RequestException, ValueError) as exc:
            failed += 1
            print(f"Filing failed: {filename}: {exc}")
    print(f"Selected {selected}, added {added}, failed {failed}; dataset has {len(records)} records")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int)
    parser.add_argument("--start", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive YYYY-MM-DD")
    collect_8ks(**vars(parser.parse_args()))
