"""Recover and verify old filing events from retained SDK submissions, without network requests."""

import argparse
from contextlib import closing
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sqlite3

from edgar import Filing
from edgar._filings import HTMLParser, ParserConfig
from edgar.attachments import Attachments, parse_homepage_html
from edgar.sgml import FilingSGML
import httpx
from httpxthrottlecache.filecache.transport import FileCache

from schemas import HERE, body_hash, file_hash as fingerprint, read_records, save


def reconstruct(raw_gzip, supporting_response=None, method="primary_parser"):
    raw = gzip.decompress(raw_gzip)
    text, mode = raw.decode("utf-8"), "complete submission"
    try:
        sgml = FilingSGML.from_text(text)
    except ValueError as exc:
        if "Truncated SGML content" not in str(exc) or "</DOCUMENT>" not in text:
            raise
        # Retain the original bytes; only complete documents can establish an exact old-primary match.
        complete = text[:text.rfind("</DOCUMENT>") + len("</DOCUMENT>")] + "\n</SEC-DOCUMENT>\n"
        sgml = FilingSGML.from_text(complete)
        mode = "complete documents before truncated trailing attachment"
    if sgml.form != "8-K":
        raise ValueError("Legacy event evidence must be a primary 8-K")
    def response(url):
        request = httpx.Request("GET", url)
        if (request.url.host != "www.sec.gov" or not any(accession in request.url.path for accession in
                (sgml.accession_number, sgml.accession_number.replace("-", "")))
                or supporting_response is None):
            raise ValueError(f"No retained offline evidence for {request.url}")
        content, metadata = supporting_response(str(request.url))
        return httpx.Response(200, headers=json.loads(metadata).get("headers", {}), content=content, request=request)

    if method == "primary_parser":
        primary = sgml.get_document_by_name(sgml.primary_documents[0].document)
        content = primary.content
    elif method == "cached_primary_parser":
        cik = re.search(r"CENTRAL INDEX KEY:\s*(\d+)", text)
        if cik is None:
            raise ValueError("Retained submission has no CIK for its index URL")
        filing = Filing(int(cik.group(1)), "Offline", sgml.form, str(sgml.filing_date), sgml.accession_number)
        primary = Attachments.load(parse_homepage_html(response(filing.homepage_url).content)).primary_html_document
        if primary is None or primary.document_type != "8-K":
            raise ValueError("Retained index has no primary 8-K")
        content = response(primary.url).text
    elif method == "raw_text":
        # Match the pinned SDK's download_text_between_tags fallback using the original cached submission.
        captured, lines = False, []
        for line in text.splitlines():
            if line.startswith("<TEXT>"):
                captured = True
            elif line.startswith("</TEXT>"):
                break
            elif captured and line:
                lines.append(line)
        if not lines:
            raise ValueError("Primary text is unavailable in the retained submission")
        body = "\n".join(lines).strip()
    else:
        raise ValueError("Unknown old-primary reconstruction method")
    if method != "raw_text":
        body = HTMLParser(ParserConfig(form="8-K")).parse(content).text(
            table_max_col_width=500, include_images=False).strip()
    return "sec:" + sgml.accession_number, str(sgml.filing_date), body_hash({"body": body}), hashlib.sha256(raw).hexdigest(), mode


def recover(cache, legacy, output):
    rows = read_records(legacy)
    previous = {body_hash(row) for row in rows}
    start, end = min(row["date"] for row in rows).replace("-", ""), max(row["date"] for row in rows).replace("-", "")
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "events.json"
    if existing.exists():
        registry = json.loads(existing.read_text(encoding="utf-8"))
        if registry["legacy_sha256"] != fingerprint(legacy):
            raise ValueError("Recovery uses another legacy archive; preserve it and use another output")
        if registry["status"] == "complete":
            load(existing, legacy)
            return
    result = {"legacy_sha256": fingerprint(legacy), "legacy_unique_bodies": len(previous),
              "sdk_version": version("edgartools"), "events": {}, "errors": [], "status": "scanning"}
    with closing(sqlite3.connect(output / "submissions.sqlite")) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS submissions (event_id TEXT PRIMARY KEY, raw_gzip BLOB, cache_metadata TEXT, sha256 TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS supporting_responses (event_id TEXT, url TEXT, body BLOB, metadata TEXT, PRIMARY KEY(event_id,url))")
        transport_cache = FileCache(cache.parent)
        for path in sorted(cache.iterdir()):
            match = re.search(r"-(\d{10}-\d{2}-\d{6})\.txt-[a-f0-9]+$", path.name)
            if not match:
                continue
            try:
                with gzip.open(path, "rb") as stream:
                    header = stream.read(65536).decode("utf-8", errors="replace")
                form = re.search(r"CONFORMED SUBMISSION TYPE:\s*([^\n]+)", header)
                day = re.search(r"FILED AS OF DATE:\s*(\d{8})", header)
                if not form or form.group(1).strip() != "8-K" or not day or not start <= day.group(1) <= end:
                    continue
                raw = path.read_bytes()
                supporting = {}
                def response(url):
                    parsed = httpx.URL(url)
                    cached = transport_cache.to_path(parsed.host, parsed.path, parsed.query.decode())
                    value = cached.read_bytes(), Path(str(cached) + ".meta").read_text(encoding="utf-8")
                    supporting[url] = value
                    return value
                for method in ("primary_parser", "raw_text", "cached_primary_parser"):
                    supporting.clear()
                    try:
                        event, date, digest, submission_sha, mode = reconstruct(raw, response, method)
                    except FileNotFoundError:
                        continue
                    if digest in previous:
                        break
                else:
                    continue
                if event != "sec:" + match.group(1):
                    raise ValueError("Cache filename and submission accession differ")
                if digest not in previous:
                    continue
                meta_path = Path(str(path) + ".meta")
                metadata = meta_path.read_text(encoding="utf-8") if meta_path.exists() else "{}"
                raw_sha = hashlib.sha256(raw).hexdigest()
                result["events"][event] = {"legacy_body_sha256": digest, "date": date, "cache_file": path.name,
                                           "raw_gzip_sha256": raw_sha, "submission_sha256": submission_sha,
                                           "cache_metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest(),
                                           "reconstruction": mode, "method": method}
                db.execute("INSERT OR REPLACE INTO submissions VALUES (?,?,?,?)", (event, raw, metadata, raw_sha))
                for url, (content, response_meta) in supporting.items():
                    db.execute("INSERT OR REPLACE INTO supporting_responses VALUES (?,?,?,?)", (event, url, content, response_meta))
                result["events"][event]["supporting_responses"] = {
                    url: {"body_sha256": hashlib.sha256(value[0]).hexdigest(), "metadata_sha256": hashlib.sha256(value[1].encode()).hexdigest()}
                    for url, value in supporting.items()}
                if len(result["events"]) % 250 == 0:
                    db.commit()
                    save(result, output / "events.json")
                    print(f"Recovered {len(result['events'])} old filing events", flush=True)
            except Exception as exc:
                result["errors"].append({"cache_file": path.name, "error": str(exc)})
    if fingerprint(legacy) != result["legacy_sha256"]:
        raise ValueError("Legacy archive changed during recovery")
    found = {entry["legacy_body_sha256"] for entry in result["events"].values()}
    result.update(matched_bodies=len(found), missing_body_sha256=sorted(previous - found),
                  submissions_sha256=fingerprint(output / "submissions.sqlite"),
                  status="complete" if found == previous else "partial")
    save(result, output / "events.json")
    if result["status"] != "complete":
        raise ValueError(f"Incomplete old filing identity: {len(previous - found)} bodies remain; preserve the recovery evidence")


def load(path, legacy):
    registry = json.loads(path.read_text(encoding="utf-8"))
    previous = {body_hash(row) for row in read_records(legacy)}
    database = path.parent / "submissions.sqlite"
    if (registry["status"] != "complete" or registry["legacy_sha256"] != fingerprint(legacy)
            or registry["sdk_version"] != version("edgartools")
            or registry["submissions_sha256"] != fingerprint(database)
            or {entry["legacy_body_sha256"] for entry in registry["events"].values()} != previous):
        raise ValueError("Old filing registry is incomplete, changed or uses another archive/SDK")
    events = {}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for event, info in registry["events"].items():
            value = db.execute("SELECT raw_gzip,cache_metadata,sha256 FROM submissions WHERE event_id=?", (event,)).fetchone()
            if (not value or not hashlib.sha256(value[0]).hexdigest() == info["raw_gzip_sha256"] == value[2]
                    or hashlib.sha256(value[1].encode()).hexdigest() != info["cache_metadata_sha256"]):
                raise ValueError(f"Old filing retained evidence differs: {event}")
            def response(url):
                expected = info.get("supporting_responses", {}).get(url)
                if expected is None:
                    raise FileNotFoundError(f"No retained supporting response for {url}")
                stored = db.execute("SELECT body,metadata FROM supporting_responses WHERE event_id=? AND url=?", (event, url)).fetchone()
                if (not stored or hashlib.sha256(stored[0]).hexdigest() != expected["body_sha256"]
                        or hashlib.sha256(stored[1].encode()).hexdigest() != expected["metadata_sha256"]):
                    raise ValueError(f"Old filing supporting evidence differs: {url}")
                return stored
            restored_event, date, digest, raw_sha, mode = reconstruct(value[0], response, info.get("method", "primary_parser"))
            if (restored_event, date, digest, raw_sha, mode) != (event, info["date"], info["legacy_body_sha256"], info["submission_sha256"],
                                                              info.get("reconstruction", "complete submission")):
                raise ValueError(f"Old filing reconstruction differs: {event}")
            events[event] = digest
            if len(events) % 1000 == 0:
                print(f"Verified {len(events)} old filing events", flush=True)
    return events, {"sdk_version": registry["sdk_version"], "legacy_bodies": len(previous), "verified_events": len(events)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".edgar/_tcache/www.sec.gov")
    parser.add_argument("--legacy", type=Path, default=HERE / "data/dataset.json")
    parser.add_argument("--output", type=Path, default=HERE / "data/research/legacy-events")
    args = parser.parse_args()
    recover(args.cache, args.legacy, args.output)
