"""Wait for collection, validate/label/freeze the corpus, then fit and test the serial study."""

import argparse
from contextlib import ExitStack, closing
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from filelock import FileLock
import psutil

from label import Budget, LEDGER, MAX_OUTPUT, MODEL, RATES, RUBRIC_HASH, reservation
from prepare import family
from schemas import HERE, annotation, body_hash, file_hash, read_records, save


FAMILIES = ("8k", "releases", "6k", "fed", "ecb", "news", "govuk")


def wait_collectors(pids):
    for pid in pids:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            continue  # The completed corpus still needs verification below.
        if pid == psutil.Process().pid or not any(Path(arg).name == "collect_sources.py" for arg in process.cmdline()):
            raise ValueError(f"PID {pid} is not a source collector")
        print(f"Waiting for collector PID {pid}", flush=True)
        while process.is_running():
            try:
                code = process.wait(timeout=30)
            except psutil.TimeoutExpired:
                continue
            if code not in (0, None):
                raise RuntimeError(f"Collector PID {pid} exited with {code}; preserve logs and resolve collection")


def audit(corpora, legacy):
    result, checked, total = {}, set(), len(read_records(legacy))
    if not total:
        raise ValueError("The legacy corpus is missing or empty")
    caches = (corpora[0].parent / "raw", corpora[0].parent.parent / "pilot/raw")
    with ExitStack() as stack:
        databases = [stack.enter_context(closing(sqlite3.connect((directory / "responses.sqlite").resolve().as_uri() + "?mode=ro", uri=True)))
                     for directory in caches if (directory / "responses.sqlite").exists()]
        for path in corpora:
            state = json.loads(path.with_suffix(".state.json").read_text(encoding="utf-8"))
            if "exhausted" not in state:
                raise ValueError(f"Collector has not finished {path.name}")
            settings = state["settings"]
            if (settings["family"], settings["start"], settings["end"]) != (path.stem, "2019-01-01", "2026-09-30"):
                raise ValueError(f"Collection family/dates differ: {path.name}")
            rows = read_records(path)
            if len(rows) != state["collected"] or not rows:
                raise ValueError(f"Collector count differs or corpus is empty: {path.name}")
            total += len(rows)
            identities, bodies = set(), set()
            expected_family = "news" if path.stem == "govuk" else path.stem
            providers = {"voa", "wikinews"} if path.stem == "news" else {
                "sec" if path.stem in ("8k", "6k", "releases") else path.stem}
            for row in rows:
                if (date.fromisoformat(row["date"]).isoformat() != row["date"] or family(row) != expected_family
                        or row["source"] not in providers):
                    raise ValueError(f"Noncanonical date or wrong source family: {path.name}")
                provenance = row["provenance"]
                if not "2019-01-01" <= row["date"] <= "2026-09-30" or not row["body"].strip():
                    raise ValueError(f"Invalid source date/body: {path.name}")
                if any(not row.get(key) for key in ("document_id", "url", "source")) or any(not provenance.get(key) for key in
                        ("raw_sha256", "body_sha256", "retrieved_utc", "extraction_version", "rights", "rights_url", "attribution", "modification")):
                    raise ValueError(f"Required provenance missing: {path.name}")
                stamp = datetime.fromisoformat(provenance["retrieved_utc"])
                digest = body_hash(row)
                if (stamp.tzinfo is None or stamp.astimezone(timezone.utc).date().isoformat() < row["date"]
                        or digest != provenance["body_sha256"]):
                    raise ValueError(f"Source date/body hash differs: {row['document_id']}")
                if row["document_id"] in identities or digest in bodies:
                    raise ValueError(f"Repeated document identity/body within {path.name}")
                identities.add(row["document_id"])
                bodies.add(digest)
                language = provenance["language"]
                if language["label"] != "en" or not .9 <= language["probability"] <= 1:
                    raise ValueError(f"Language filter differs: {row['document_id']}")
                key = provenance["url"], provenance["raw_sha256"]
                if key not in checked:
                    responses = [db.execute("SELECT body,metadata FROM responses WHERE url=?", (key[0],)).fetchone() for db in databases]
                    digest = hashlib.sha256(key[0].encode()).hexdigest()
                    for directory in caches:
                        raw, metadata = directory / f"{digest}.bin", directory / f"{digest}.json"
                        if raw.exists() and metadata.exists():
                            responses.append((raw.read_bytes(), metadata.read_text(encoding="utf-8")))
                    if not any(value and hashlib.sha256(value[0]).hexdigest() == key[1] == json.loads(value[1])["raw_sha256"]
                               for value in responses):
                        raise ValueError(f"Retained response missing or hash differs: {row['document_id']}")
                    checked.add(key)
            result[path.name] = {"documents": len(rows), "sha256": file_hash(path),
                                 "collector_outcomes": state["outcomes"]}
            print(f"Verified {path.name}: {len(rows)} documents", flush=True)
    if total > 100_000:
        raise ValueError("Raw input exceeds the approved 100,000-document ceiling")
    return {"raw_documents_including_legacy": total, "sources": result}


def quote(splits):
    result = {"model": MODEL, "service_tier": "flex", "rubric_sha256": RUBRIC_HASH,
              "rates_usd_per_million": RATES, "output_tokens_max": MAX_OUTPUT, "queues": {},
              "note": "Offline token approximation plus 128 request tokens; byte-bound reservations settle to actual usage"}
    budget = Budget(LEDGER)
    try:
        result["project_charged_or_reserved_usd"] = budget.total()
        for name in ("validation", "test", "train"):
            rows = [row for row in read_records(splits / f"{name}-input.jsonl") if annotation(row)["label"] is None]
            tokens, maximum, estimate = 0, 0.0, 0.0
            for row in rows:
                count, bound = reservation(row["body"])
                count += 128
                long = count > 272_000
                tokens += count
                maximum += bound
                estimate += (count * RATES["input"] * (2 if long else 1) + MAX_OUTPUT * RATES["output"] * (1.5 if long else 1)) / 1e6
            result["queues"][name] = {"unlabeled": len(rows), "approximate_input_tokens": tokens,
                                       "approximate_usd_output_max": estimate, "byte_bound_reservations_usd": maximum}
    finally:
        budget.db.close()
    return result


def run(args):
    corpora = [args.corpus / f"{name}.jsonl" for name in FAMILIES]
    output = args.study / "workflow"
    output.mkdir(parents=True, exist_ok=True)
    def source_hashes():
        paths = [*sorted(HERE.glob("*.py")), HERE.parent / "pyproject.toml", HERE.parent / "uv.lock"]
        return {str(path.relative_to(HERE.parent)): file_hash(path) for path in paths}

    sources = source_hashes()
    plan = {"corpora": list(map(str, corpora)), "legacy": str(args.legacy), "legacy_events": str(args.legacy_events), "splits": str(args.splits),
            "study": str(args.study), "source_sha256": sources, "label_ceiling_usd": 3.0, "ledger": str(LEDGER)}
    plan_path = output / "plan.json"
    def unchanged():
        if source_hashes() != sources:
            raise ValueError("Source changed while waiting/running; preserve the workflow before resuming")

    def command(stage, script, arguments):
        unchanged()
        print(f"START {stage}", flush=True)
        with (output / f"{stage}.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(HERE / script), *map(str, arguments)], cwd=HERE,
                                       stdout=log, stderr=subprocess.STDOUT)
            save({"status": "running", "stage": stage, "child_pid": process.pid}, output / "state.json")
            code = process.wait()
        if code:
            raise RuntimeError(f"{stage} exited with {code}; inspect {output / f'{stage}.log'}")

    with FileLock(str(output / "pipeline.lock"), timeout=0):
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Workflow paths/source changed; preserve this workflow and use a new study output")
        save(plan, plan_path)
        try:
            save({"status": "waiting_for_collection", "collector_pids": args.wait_pid}, output / "state.json")
            wait_collectors(args.wait_pid)
            unchanged()
            save(audit(corpora, args.legacy), output / "corpus-audit.json")
            if not (args.splits / "prepared.json").exists():
                command("prepare", "prepare.py", ["--corpora", *corpora, "--legacy", args.legacy,
                                                  "--legacy-events", args.legacy_events, "--output", args.splits])
            prepared = json.loads((args.splits / "prepared.json").read_text(encoding="utf-8"))
            event_audit = prepared.get("legacy_event_check") or {}
            legacy_bodies = len({body_hash(row) for row in read_records(args.legacy)})
            if (set(map(Path, prepared["inputs"])) != set([*corpora, args.legacy, args.legacy_events, args.legacy_events.parent / "submissions.sqlite"])
                    or prepared["rubric_sha256"] != RUBRIC_HASH or event_audit.get("legacy_bodies") != legacy_bodies
                    or event_audit.get("verified_events", 0) < legacy_bodies
                    or any(file_hash(path) != digest
                           for path, digest in prepared["inputs"].items())):
                raise ValueError("Prepared corpus inputs/rubric changed")
            if not (args.splits / "manifest.json").exists():
                unchanged()
                save(quote(args.splits), output / "label-quote.json")
                for name in ("validation", "test", "train"):
                    path = args.splits / f"{name}-input.jsonl"
                    command(f"label-{name}", "label.py", ["--dataset", path, "--ledger", LEDGER, "--budget-usd", 3])
                    missing = [row for row in read_records(path) if annotation(row)["label"] is None]
                    if any(row.get("label_error") for row in missing):
                        raise ValueError(f"Resolve failed {name} requests before freezing; reservations remain in the shared ledger")
                    if name != "train" and missing:
                        raise ValueError(f"Complete selected {name} labels before freezing; shared $3 cap remains in force")
                command("data-freeze", "prepare.py", ["--output", args.splits, "--freeze"])
            review = output / "review"
            if not (review / "mapping.json").exists():
                command("review-pack", "review.py", ["--splits", args.splits, "--output", review])
            common = ["--splits", args.splits, "--output", args.study]
            if not (args.study / "freeze.json").exists():
                command("fit", "study.py", [*common, "--stage", "fit"])
                command("model-freeze", "study.py", [*common, "--stage", "freeze"])
            command("test", "study.py", [*common, "--stage", "test"])
            save({"status": "ready_for_report", "note": "Model study complete; report/PDF and qualified human audit remain"}, output / "state.json")
            print("Study ready for report generation and inspection", flush=True)
        except Exception as exc:
            save({"status": "needs_attention", "error": str(exc)}, output / "state.json")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "data/research/corpus")
    parser.add_argument("--legacy", type=Path, default=HERE / "data/dataset.json")
    parser.add_argument("--legacy-events", type=Path, default=HERE / "data/research/legacy-events/events.json")
    parser.add_argument("--splits", type=Path, default=HERE / "data/research/benchmark")
    parser.add_argument("--study", type=Path, default=HERE / "training_runs/research-study")
    parser.add_argument("--wait-pid", type=int, nargs="*", default=[])
    args = parser.parse_args()
    for name in ("corpus", "legacy", "legacy_events", "splits", "study"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)
