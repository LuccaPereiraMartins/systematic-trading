"""Collect traceable public documents into separate, resumable JSONL corpora."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urljoin, urlparse
from urllib.parse import urlencode

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from filelock import FileLock
import requests

from schemas import HERE, body_hash, read_records, save


FAMILIES = ("8k", "releases", "6k", "fed", "ecb", "news", "govuk")
EXTRACTION_VERSION = "html-text-v1"
RELEASE_QUERY = '"press release" OR "news release" OR "GLOBE NEWSWIRE" OR "PRNewswire"'
RIGHTS = {
    "govuk": ("Crown copyright; Open Government Licence v3.0 except stated third-party/personal content",
              "https://www.gov.uk/help/reuse-govuk-content"),
    "sec": ("SEC public filing reuse", "https://www.sec.gov/files/about/webmaster-faq.htm"),
    "fed": ("Public domain unless otherwise indicated", "https://www.federalreserve.gov/disclaimer.htm"),
    "ecb": ("ECB institutional information: attribution and accuracy conditions",
            "https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html"),
    "voa": ("VOA-original text only: public domain; agency content excluded", "https://www.voanews.com/p/5338.html"),
    "wikinews": ("CC BY 4.0 since 2024-12-16; CC BY 2.5 earlier",
                 "https://en.wikinews.org/wiki/Wikinews:Copyright"),
}


class HTTP:
    """Cache original bytes and response provenance; one shared request stream."""
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(directory / "responses.sqlite", timeout=30)
        self.cache.execute("CREATE TABLE IF NOT EXISTS responses (url TEXT PRIMARY KEY, body BLOB, metadata TEXT)")
        self.cache.commit()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = os.environ["SEC_USER_AGENT"]
        self.last = 0.0

    def get(self, url):
        cached = self.cache.execute("SELECT body,metadata FROM responses WHERE url=?", (url,)).fetchone()
        if cached:
            return cached[0], json.loads(cached[1])
        key = hashlib.sha256(url.encode()).hexdigest()
        raw, meta = self.directory / f"{key}.bin", self.directory / f"{key}.json"
        if raw.exists() and meta.exists():
            return raw.read_bytes(), json.loads(meta.read_text(encoding="utf-8"))
        for attempt in range(4):
            time.sleep(max(0, self.last + (0.26 if urlparse(url).netloc.endswith("sec.gov") else 0.4) - time.monotonic()))
            self.last = time.monotonic()
            try:
                response = self.session.get(url, timeout=45)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
                continue
            if response.status_code not in (429, 500, 502, 503, 504):
                response.raise_for_status()
                break
            time.sleep(2 ** attempt)
        else:
            response.raise_for_status()
        metadata = {"url": url, "final_url": response.url, "status": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "raw_sha256": hashlib.sha256(response.content).hexdigest()}
        self.cache.execute("INSERT OR IGNORE INTO responses VALUES (?,?,?)", (url, response.content, json.dumps(metadata)))
        self.cache.commit()
        return response.content, metadata

    def json(self, url):
        return json.loads(self.get(url)[0])


def balanced(items):
    """Round-robin months, with stable hash ordering within each month."""
    months = defaultdict(list)
    for row in items:
        months[row["date"][:7]].append(row)
    groups = [sorted(months[m], key=lambda r: hashlib.sha256(r["document_id"].encode()).hexdigest())
              for m in sorted(months, key=lambda month: hashlib.sha256(month.encode()).hexdigest())]
    return [r for group in zip_longest(*groups) for r in group if r is not None]


def govuk_candidates(http, start, end):
    """HM Treasury news/press releases; unsupported search API is discovery only."""
    rows, offset = [], 0
    while True:
        parameters = [("filter_organisations", "hm-treasury"), ("filter_format", "press_release"),
                      ("filter_format", "news_story"), ("fields", "title,link,public_timestamp,format"),
                      ("count", 1000), ("start", offset), ("order", "-public_timestamp")]
        url = "https://www.gov.uk/api/search.json?" + urlencode(parameters)
        payload = http.json(url)
        for item in payload["results"]:
            day = item.get("public_timestamp", "")[:10]
            if start <= day <= end:
                rows.append({"document_id": "govuk:" + item["link"], "event_id": "govuk:" + item["link"],
                             "source": "govuk", "document_type": "news", "date": day,
                             "url": "https://www.gov.uk" + item["link"], "title": item["title"],
                             "content_api_url": "https://www.gov.uk/api/content" + item["link"],
                             "discovery_url": url, "publisher": "HM Treasury"})
        offset += len(payload["results"])
        if offset >= payload["total"]:
            return rows
        if not payload["results"]:
            raise ValueError("GOV.UK discovery stopped before its reported total")


def govuk_document(http, candidate):
    raw, metadata = http.get(candidate["content_api_url"])
    content = json.loads(raw)
    if content["locale"] != "en" or content["document_type"] not in ("news_story", "press_release"):
        raise ValueError("Not an English official news article")
    published, updated = content["first_published_at"], content["public_updated_at"]
    day = datetime.fromisoformat(published).astimezone(timezone.utc).date().isoformat()
    changed = datetime.fromisoformat(updated).astimezone(timezone.utc).date().isoformat()
    if changed > day:
        raise ValueError("Public revision after original publication day; historical body unavailable")
    page = BeautifulSoup(content["details"]["body"], "html.parser")
    for node in page.select("img, script, style"):
        node.decompose()
    body = content["title"] + "\n" + page.get_text("\n", strip=True)
    if len(body) < 100 or re.search(r"copyright|third[- ]party content|all rights reserved", body, re.I):
        raise ValueError("Insufficient text or special-rights notice")
    return {**candidate, "date": day, "body": body, "llm": {"label": None, "uncertainty": None},
            "human": {"label": None, "uncertainty": None}, "date_precision": "day",
            "provenance": {**metadata, "content_id": content["content_id"], "published_at": published,
                           "public_updated_at": updated, "extraction_version": "govuk-json-text-v1",
                           "rights": RIGHTS["govuk"][0], "rights_url": RIGHTS["govuk"][1],
                           "attribution": "Contains public sector information licensed under the Open Government Licence v3.0; HM Treasury",
                           "modification": "Title and article text extracted; HTML, images and attachments excluded"}}


def sec_candidates(family, start, end):
    from edgar import get_filings, set_identity
    set_identity(os.environ["SEC_USER_AGENT"])
    form = "6-K" if family == "6k" else "8-K"
    filings = get_filings(year=list(range(int(start[:4]), int(end[:4]) + 1)), quarter=[1, 2, 3, 4],
                          form=form, amendments=False, filing_date=f"{start}:{end}")
    return [{"document_id": f"sec:{f.accession_number}", "event_id": f"sec:{f.accession_number}",
             "date": str(f.filing_date), "source": "sec", "document_type": form,
             "issuer": str(f.cik), "company": f.company, "accession": f.accession_number}
            for f in filings]


def release_candidates(http, start, end):
    """Discover release exhibits by their text, avoiding attachment-description template bias."""
    # EFTS treats grouping parentheses as required literal terms.
    query = RELEASE_QUERY
    periods, items = [], {}
    first, final = date.fromisoformat(start), date.fromisoformat(end)
    while first <= final:
        next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        last = min(final, next_month - timedelta(days=1))
        periods.append((first, last))
        first = last + timedelta(days=1)
    def search(first, last):
        parameters = {"q": query, "dateRange": "custom", "startdt": str(first), "enddt": str(last),
                      "forms": "8-K", "from": 0, "size": 100}
        root = "https://efts.sec.gov/LATEST/search-index?"
        result = http.json(root + urlencode(parameters))
        if result.get("timed_out") or result.get("_shards", {}).get("failed", 0):
            raise ValueError("SEC discovery returned incomplete search results")
        total = result["hits"]["total"]
        if total["relation"] != "eq" or total["value"] > 10_000:
            return False
        for offset in range(0, total["value"], 100):
            parameters["from"] = offset
            discovery = root + urlencode(parameters)
            page = result if offset == 0 else http.json(discovery)
            if (page.get("timed_out") or page.get("_shards", {}).get("failed", 0)
                    or len(page["hits"]["hits"]) != min(100, total["value"] - offset)):
                raise ValueError("SEC discovery returned an incomplete results page")
            for hit in page["hits"]["hits"]:
                row = hit["_source"]
                if not row.get("file_type", "").startswith("EX-99") or "8-K" not in row.get("root_forms", []):
                    continue
                accession, filename = hit["_id"].split(":", 1)
                issuer = str(int(row["ciks"][0]))
                url = f"https://www.sec.gov/Archives/edgar/data/{issuer}/{accession.replace('-', '')}/{filename}"
                items[url] = {"document_id": url, "event_id": f"sec:{accession}", "url": url,
                              "date": row["file_date"], "source": "sec", "document_type": "corporate_release",
                              "issuer": issuer, "accession": accession, "discovery_url": discovery,
                              "discovery_query": query, "attachment_description": row.get("file_description", "")}
        return True

    for first, last in periods:
        try:
            complete = search(first, last)
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code not in (500, 502, 503, 504):
                raise
            complete = False
        if not complete:
            if first == last:
                raise ValueError("SEC search failed or reached its cap for one day; discovery remains incomplete")
            midpoint = first + (last - first) // 2
            periods.extend(((first, midpoint), (midpoint + timedelta(days=1), last)))
            print(f"Retrying SEC release discovery in smaller ranges: {first}..{last}", flush=True)
        else:
            print(f"Release discovery: {first}..{last}; {len(items)} unique exhibits", flush=True)
    return list(items.values())


def fed_candidates(http, start, end):
    items = {}
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for kind, path in (("press_release", f"pressreleases/{year}-press.htm"),
                           ("speech", f"speech/{year}-speeches.htm")):
            url = f"https://www.federalreserve.gov/newsevents/{path}"
            page = BeautifulSoup(http.get(url)[0], "html.parser")
            for a in page.select("a[href]"):
                link = urljoin(url, a["href"])
                match = re.search(r"/(?:pressreleases|speech)/[a-z]+(\d{8})[a-z]?\.htm$", link)
                if not match:
                    continue
                date = datetime.strptime(match[1], "%Y%m%d").date().isoformat()
                if start <= date <= end:
                    items[link] = {"document_id": link, "url": link, "date": date,
                                   "source": "fed", "document_type": kind}
    return list(items.values())


def ecb_candidates(http, start, end):
    # This is the same versioned archive consumed by the ECB's public foedb website widget.
    root = "https://www.ecb.europa.eu/foedb/dbs/foedb/publications.en/"
    version = http.json(root + "versions.json")[0]
    base = root + version["version"] + "/" + version["hash"] + "/"
    metadata = http.json(base + "metadata.json")
    items = []
    width = len(metadata["header"])
    for chunk in range((metadata["total_records"] + metadata["chunk_size"] - 1) // metadata["chunk_size"]):
        group = chunk // metadata["chunk_group_size"]
        data = http.json(base + f"data/{group}/chunk_{chunk}.json")
        for offset in range(0, len(data), width):
            record = dict(zip(metadata["header"], data[offset:offset + width]))
            date = datetime.fromtimestamp(record["pub_timestamp"], timezone.utc).date().isoformat()
            if not start <= date <= end or record.get("Authors") or record.get("boardmember"):
                continue
            for path in record.get("documentTypes") or []:
                if not re.match(r"/press/(?:pr|govcdec/mopo|accounts)/", path) or not path.endswith(".en.html"):
                    continue
                url = urljoin("https://www.ecb.europa.eu", path)
                items.append({"document_id": url, "url": url, "date": date, "source": "ecb",
                              "document_type": "institutional_release", "archive_version": version})
    return items


def wikinews_candidates(http, start, end):
    root = "https://en.wikinews.org/w/api.php?format=json&action=query&list=categorymembers"
    categories, visited, items = [("Category:Economy_and_business", 0)], set(), {}
    for category, depth in categories:
        if category in visited:
            continue
        visited.add(category)
        continuation = ""
        while True:
            query = root + "&cmtitle=" + requests.utils.quote(category) + "&cmnamespace=0%7C14&cmlimit=500" + continuation
            result = http.json(query)
            for row in result["query"]["categorymembers"]:
                if row["ns"] == 14:
                    if depth < 2:
                        categories.append((row["title"], depth + 1))
                    continue
                url = "https://en.wikinews.org/wiki/" + requests.utils.quote(row["title"].replace(" ", "_"))
                items.setdefault(row["pageid"], {"document_id": f"wikinews:{row['pageid']}", "url": url,
                                 "source": "wikinews", "document_type": "news", "date": "",
                                 "discovery_category": category})
            if "continue" not in result:
                break
            continuation = "&cmcontinue=" + requests.utils.quote(result["continue"]["cmcontinue"])
    return list(items.values())


def news_candidates(http, start, end):
    items, failures = [], []
    # GDELT gives URLs/metadata, not a license to the underlying publisher text.
    query = ('https://api.gdeltproject.org/api/v2/doc/doc?query=domainis:voanews.com%20'
             '(economy%20OR%20business%20OR%20inflation%20OR%20earnings)&mode=artlist&format=json'
             '&maxrecords=250&sort=datedesc&timespan=3m')
    try:
        for row in http.json(query).get("articles", []):
            date = datetime.strptime(row["seendate"][:8], "%Y%m%d").date().isoformat()
            if start <= date <= end and urlparse(row["url"]).netloc in ("www.voanews.com", "voanews.com"):
                items.append({"document_id": row["url"], "url": row["url"], "date": date,
                              "source": "voa", "document_type": "news", "discovery_date": date})
    except (requests.RequestException, ValueError, KeyError) as exc:
        failures.append({"provider": "gdelt", "error": str(exc)})
    try:
        items.extend(wikinews_candidates(http, start, end))
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise RuntimeError("Required Wikinews discovery failed; cache retained for resume") from exc
    # VOA's accessible archive is also useful when recent GDELT coverage is sparse.
    for page in range(1, 11):
        url = f"https://www.voanews.com/z/599?p={page}"
        try:
            html = BeautifulSoup(http.get(url)[0], "html.parser")
        except requests.RequestException as exc:
            failures.append({"provider": "voa", "error": str(exc)})
            break
        for a in html.select('a[href*="/a/"]'):
            title = a.get_text(" ", strip=True)
            if not re.search(r"\b(econom|business|bank|trade|tariff|inflation|stock|market|earnings|financial|company|companies)\w*", title, re.I):
                continue
            link = urljoin(url, a["href"])
            items.append({"document_id": link, "url": link, "date": "", "source": "voa", "document_type": "news"})
    return list({r["document_id"]: r for r in items}.values()), failures


def extract(http, candidate):
    source = candidate["source"]
    raw, metadata = http.get(candidate["url"])
    if "pdf" in metadata["content_type"].lower() or raw.startswith(b"%PDF"):
        raise ValueError("Unsupported PDF; original bytes retained")
    page = BeautifulSoup(raw, "html.parser")
    for element in page.select("script,style,noscript,nav,footer,header"):
        element.decompose()
    selectors = {"fed": "#article", "ecb": "main", "voa": ".wsw, .body-container",
                 "wikinews": ".mw-parser-output", "sec": "body"}
    content = page.select_one(selectors[source])
    if content is None:
        raise ValueError(f"Missing article container for {source}")
    if source == "voa":
        author = page.select_one(".authors, .author, .byline")
        author_text = author.get_text(" ", strip=True) if author else ""
        if re.search(r"\b(Reuters|Associated Press|Agence France|AFP|AP)\b", content.get_text(" ") + author_text):
            raise ValueError("Agency-owned or mixed-rights content excluded")
        if not re.search(r"\bVOA\b|Voice of America", author_text, re.I):
            raise ValueError("VOA-exclusive authorship not established")
    if source in ("voa", "wikinews"):
        stamp = page.select_one('#publishDate[title], meta[property="article:published_time"], meta[name="date"], time[datetime]')
        date = (stamp.get("content") or stamp.get("datetime") or stamp.get("title"))[:10] if stamp else ""
        if not date and source == "wikinews":
            text = content.get_text(" ", strip=True)
            match = re.search(r"\b(\d{1,2} [A-Z][a-z]+ \d{4})\b", text)
            if match:
                date = datetime.strptime(match[1], "%d %B %Y").date().isoformat()
        if not date:
            raise ValueError("Publication date missing; discovery time is not publication time")
        candidate = {**candidate, "date": date}
    for element in content.select(".related, .related-items, .mw-editsection, .noprint, .sources, .infobox"):
        element.decompose()
    if source == "wikinews":
        for element in content.select("table, aside"):
            if re.search(r"Related articles|Sister links", element.get_text(" ")):
                element.decompose()
    body = content.get_text("\n", strip=True)
    if len(body) < 100:
        raise ValueError("No readable document text")
    if candidate["document_type"] == "corporate_release" and re.search(
        r"management.{0,5}s?\s+discussion\s+and\s+analysis|(?:unaudited\s+)?consolidated\s+financial\s+statements",
        body[:300], re.I,
    ):
        raise ValueError("Financial-statement or MD&A exhibit is not a corporate release")
    rights = RIGHTS[source][0]
    if source == "wikinews":
        rights = "CC BY 4.0" if candidate["date"] >= "2024-12-16" else "CC BY 2.5"
    return {**candidate, "body": body, "llm": {"label": None, "uncertainty": None},
            "human": {"label": None, "uncertainty": None}, "date_precision": "day",
            "provenance": {**metadata, "extraction_version": EXTRACTION_VERSION,
                           "rights": rights, "rights_url": RIGHTS[source][1],
                           "attribution": source, "modification": "HTML navigation removed; plain-text extraction"}}


def resume_records(output):
    if output.exists():
        raw = output.read_bytes()
        end = raw.rfind(b"\n") + 1
        tail = raw[end:]
        if tail:
            try:
                json.loads(tail)
            except (json.JSONDecodeError, UnicodeDecodeError):
                archive = output.with_name(output.name + f".interrupted-{time.time_ns()}")
                archive.write_bytes(tail)
                with output.open("r+b") as stream:
                    stream.truncate(end)
                print(f"Retained {len(tail)} interrupted append bytes in {archive.name}", flush=True)
            else:
                with output.open("ab") as stream:
                    stream.write(b"\n")
    return read_records(output)


def sec_documents(candidate, family, http=None):
    if family != "releases" and http is not None:
        from edgar.attachments import Attachments, parse_homepage_html
        directory = f"https://www.sec.gov/Archives/edgar/data/{int(candidate['issuer'])}/{candidate['accession'].replace('-', '')}/"
        index_url = directory + candidate["accession"] + "-index.html"
        try:
            raw, metadata = http.get(index_url)
        except requests.HTTPError as exc:
            if exc.response.status_code not in (403, 404):
                raise
        else:
            document = Attachments.load(parse_homepage_html(raw)).primary_html_document
            if document and document.document_type == candidate["document_type"] and document.extension != ".paper":
                return [{**candidate, "url": document.url,
                         "primary_discovery": {"url": index_url, "raw_sha256": metadata["raw_sha256"]}}]
    from edgar import Filing
    filing = Filing(int(candidate["issuer"]), candidate["company"], candidate["document_type"],
                    candidate["date"], candidate["accession"])
    if family != "releases":
        return [{**candidate, "url": filing.document.url}]
    rows = []
    for attachment in filing.attachments:
        kind = str(getattr(attachment, "document_type", ""))
        description = str(getattr(attachment, "description", ""))
        if kind.startswith("EX-99") and re.search(r"release|earnings|results|announcement", description, re.I):
            rows.append({**candidate, "document_id": attachment.url, "url": attachment.url,
                         "document_type": "corporate_release"})
    return rows


def collect(family, number, start, end, output):
    from langid.langid import LanguageIdentifier, model
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    load_dotenv(HERE.parent / ".env")
    if not os.getenv("SEC_USER_AGENT"):
        raise ValueError("Set SEC_USER_AGENT; also used as the contact for public-source requests")
    if family in ("8k", "releases", "6k"):
        # SDK metadata plus direct text requests remain below SEC's combined ten-request limit.
        os.environ["EDGAR_RATE_LIMIT_PER_SEC"] = "4"
    if not 0 < number <= 100_000 or start > end:
        raise ValueError("Invalid collection target or dates")
    datetime.strptime(start, "%Y-%m-%d")
    datetime.strptime(end, "%Y-%m-%d")
    output = Path(output)
    if output.suffix != ".jsonl":
        raise ValueError("Source collection requires a JSONL output")
    output.parent.mkdir(parents=True, exist_ok=True)
    http = HTTP(output.parent / "raw")
    existing = resume_records(output)
    state_path, index_path = output.with_suffix(".state.json"), output.with_suffix(".index.json")
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"processed": {}, "errors": []}
    settings = {"family": family, "start": start, "end": end,
                "extraction": "govuk-json-text-v1" if family == "govuk" else EXTRACTION_VERSION}
    if family == "releases":
        settings["discovery_query"] = RELEASE_QUERY
    if state.get("settings", settings) != settings:
        raise ValueError("Collection settings changed; use a separate output")
    state["settings"] = settings
    if len(existing) >= number:
        print(f"Already collected {len(existing)} {family} documents", flush=True)
        return
    if index_path.exists():
        candidates = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        print(f"Indexing {family}: {start}..{end}", flush=True)
        if family == "releases":
            candidates = release_candidates(http, start, end)
        elif family in ("8k", "6k"):
            candidates = sec_candidates(family, start, end)
        elif family == "fed":
            candidates = fed_candidates(http, start, end)
        elif family == "ecb":
            candidates = ecb_candidates(http, start, end)
        elif family == "govuk":
            candidates = govuk_candidates(http, start, end)
        else:
            candidates, errors = news_candidates(http, start, end)
            state["errors"].extend(errors)
        candidates = balanced([r for r in candidates if r["date"]]) + [r for r in candidates if not r["date"]]
        save(candidates, index_path)
    candidates = balanced([r for r in candidates if r["date"]]) + [r for r in candidates if not r["date"]]
    state["indexed_candidates"] = len(candidates)
    candidate_ids = {row["document_id"] for row in candidates}
    for row in existing:
        if row["document_id"] in candidate_ids:
            state["processed"].setdefault(row["document_id"], "accepted")
    seen = {r.get("document_id") for r in existing}
    hashes = {body_hash(r) for r in existing}
    added = len(existing)
    counts = Counter(state["processed"].values())
    with output.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=1) as worker:
        for candidate in candidates:
            key = candidate["document_id"]
            if key in state["processed"]:
                continue
            if added >= number:
                break
            try:
                documents = sec_documents(candidate, family, http) if candidate["source"] == "sec" and not candidate.get("url") else [candidate]
                accepted = 0
                for document in documents:
                    if document["document_id"] in seen or added >= number:
                        continue
                    row = govuk_document(http, document) if document["source"] == "govuk" else extract(http, document)
                    if not start <= row["date"] <= end:
                        continue
                    language, probability = worker.submit(identifier.classify, row["body"]).result()
                    if language != "en" or probability < 0.9:
                        raise ValueError("Non-English or uncertain language")
                    digest = body_hash(row)
                    if digest in hashes:
                        continue
                    row["provenance"]["language"] = {"label": language, "probability": probability}
                    row["provenance"]["body_sha256"] = digest
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    hashes.add(digest)
                    seen.add(row["document_id"])
                    added += 1
                    accepted += 1
                outcome = "accepted" if accepted else "no_eligible_document"
                state["processed"][key] = outcome
                counts[outcome] += 1
            except ValueError as exc:
                state["processed"][key] = "excluded"
                counts["excluded"] += 1
                state["errors"].append({"id": key, "kind": "excluded", "error": str(exc)})
            except Exception as exc:
                state["errors"].append({"id": key, "kind": "retryable", "error": str(exc)})
                counts["retryable"] += 1
            if sum(counts.values()) % 25 == 0:
                state.update(collected=added, outcomes=dict(counts))
                save(state, state_path)
                print(f"{family}: {added}/{number} documents; {dict(counts)}", flush=True)
    state.update(collected=added, outcomes=dict(counts), exhausted=added < number)
    save(state, state_path)
    print(f"Finished {family}: {added}/{number}; indexed {len(candidates)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--number", type=int, default=50)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    if args["family"] in ("8k", "releases", "6k"):
        lock = HERE / "data/research/sec-collection.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        print("Waiting for the shared SEC collection slot", flush=True)
        with FileLock(lock):
            collect(**args)
    else:
        collect(**args)
