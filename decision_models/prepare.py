"""Prepare a fresh temporal benchmark, then freeze labeled splits and nested training subsets."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from datetime import date
import hashlib
from importlib.metadata import version
from itertools import zip_longest
import json
from pathlib import Path
import re
import sqlite3

from datasketch import MinHash, MinHashLSH

from schemas import HERE, annotation, body_hash, read_records, save, write_records


PERMUTATIONS = 128
NEAR_THRESHOLD = 0.90
SEARCH_THRESHOLD = 0.80
CUTOFFS = ("2026-03-31", "2026-06-30", "2026-09-30")


def family(row):
    kind = row.get("document_type", "8-K")
    if row.get("source", "sec") == "sec":
        return {"8-K": "8k", "6-K": "6k", "corporate_release": "releases"}[kind]
    return "news" if kind == "news" else row["source"]


def shingle_set(body):
    tokens = re.findall(r"\w+", body.casefold())
    return {" ".join(tokens[i:i + 5]).encode() for i in range(max(1, len(tokens) - 4))}


def legacy_links(rows, corpora, legacy_hashes):
    """Recognise prior 8-Ks in the SDK's text format using retained HTML, without changing model inputs."""
    from bs4 import UnicodeDammit
    from edgar._filings import HTMLParser, ParserConfig

    caches = sorted({directory for path in corpora for directory in (path.parent / "raw", path.parent.parent / "pilot/raw")})
    aliases = defaultdict(set)
    audit = {"method": "Exact legacy body SHA from offline edgartools primary-HTML reconstruction",
             "sdk_version": version("edgartools"), "attempted": 0, "matched_documents": 0, "matches": []}
    with ExitStack() as stack:
        databases = [stack.enter_context(closing(sqlite3.connect((directory / "responses.sqlite").resolve().as_uri() + "?mode=ro", uri=True)))
                     for directory in caches if (directory / "responses.sqlite").exists()]
        for row in rows:
            provenance = row.get("provenance", {})
            if family(row) != "8k" or not row.get("document_id") or not provenance.get("raw_sha256"):
                continue  # Legacy records and minimal fixtures have no retained response to reconstruct.
            url, expected = provenance["url"], provenance["raw_sha256"]
            responses = [db.execute("SELECT body,metadata FROM responses WHERE url=?", (url,)).fetchone() for db in databases]
            key = hashlib.sha256(url.encode()).hexdigest()
            for directory in caches:
                raw, metadata = directory / f"{key}.bin", directory / f"{key}.json"
                if raw.exists() and metadata.exists():
                    responses.append((raw.read_bytes(), metadata.read_text(encoding="utf-8")))
            raw = next((value[0] for value in responses if value and
                        hashlib.sha256(value[0]).hexdigest() == expected == json.loads(value[1])["raw_sha256"]), None)
            if raw is None:
                raise ValueError(f"Missing or changed retained HTML for legacy reconstruction: {row['document_id']}")
            document = HTMLParser(ParserConfig(form="8-K")).parse(UnicodeDammit(raw).unicode_markup)
            restored = document.text(table_max_col_width=500, include_images=False).strip()
            previous = body_hash({"body": restored})
            audit["attempted"] += 1
            if previous in legacy_hashes:
                digest = body_hash(row)
                aliases[digest].add(previous)
                audit["matched_documents"] += 1
                audit["matches"].append({"document_id": row["document_id"], "body_sha256": digest,
                                         "legacy_body_sha256": previous, "raw_sha256": expected})
            if audit["attempted"] % 1000 == 0:
                print(f"Legacy extraction check: {audit['attempted']} primary filings", flush=True)
    return aliases, audit


def cluster(rows, legacy_hashes, aliases=None):
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)

    hashes, events = {}, {}
    index = MinHashLSH(threshold=SEARCH_THRESHOLD, num_perm=PERMUTATIONS)
    for i, row in enumerate(rows):
        digest = body_hash(row)
        if digest in hashes:
            join(i, hashes[digest])
        else:
            hashes[digest] = i
            shingles = shingle_set(row["body"])
            signature = MinHash(num_perm=PERMUTATIONS, seed=42)
            ordered = sorted(shingles)
            for start in range(0, len(ordered), 4096):
                signature.update_batch(ordered[start:start + 4096])
            for other in index.query(signature):
                candidate = shingle_set(rows[other]["body"])
                if len(shingles & candidate) / len(shingles | candidate) >= NEAR_THRESHOLD:
                    join(i, other)
            index.insert(i, signature)
        event = row.get("event_id")
        if event:
            if event in events:
                join(i, events[event])
            else:
                events[event] = i
        if (i + 1) % 1000 == 0:
            print(f"Duplicate audit: {i + 1}/{len(rows)}", flush=True)
    for digest, previous in (aliases or {}).items():
        for legacy in previous:
            join(hashes[digest], hashes[legacy])
    members = defaultdict(list)
    for i in range(len(rows)):
        members[root(i)].append(i)
    groups = {}
    for indices in members.values():
        group = hashlib.sha256("|".join(sorted(body_hash(rows[i]) for i in indices)).encode()).hexdigest()
        previous = any(body_hash(rows[i]) in legacy_hashes for i in indices)
        for i in indices:
            groups[i] = (group, previous)
    return groups


def partition(date):
    if date <= CUTOFFS[0]:
        return "train"
    if date <= CUTOFFS[1]:
        return "validation"
    if date <= CUTOFFS[2]:
        return "test"
    return "outside"


def sample(rows, number=None, by_label=False):
    buckets = defaultdict(list)
    for row in rows:
        key = (family(row), annotation(row)["label"] if by_label else row["date"][:7])
        buckets[key].append(row)
    groups = [sorted(buckets[key], key=body_hash) for key in sorted(buckets, key=str)]
    ordered = [row for group in zip_longest(*groups) for row in group if row is not None]
    return ordered[:number] if number is not None else ordered


def cap_primary(rows, by_label=False):
    others = [row for row in rows if family(row) != "8k"]
    primary = [row for row in rows if family(row) == "8k"]
    # At most two primary filings per three other documents gives a 40% share.
    if len(primary) * 3 <= len(others) * 2:
        return rows
    primary = sample(primary, len(others) * 2 // 3, by_label=by_label)
    return sample(others + primary, by_label=by_label)


def prepare(corpora, output, legacy=HERE / "data/dataset.json", validation=1200, test=1800, train=25000):
    from label import RUBRIC_HASH
    if (output / "manifest.json").exists():
        raise ValueError("Benchmark is frozen; use another output directory")
    if (output / "prepared.json").exists():
        raise ValueError("Preparation already exists; label its inputs and use --freeze, or a new directory")
    legacy_rows = read_records(legacy)
    rows, census = [], []
    for path in [*corpora, legacy]:
        source_rows = legacy_rows if path == legacy else read_records(path)
        census.append({"input": str(path), "rows": len(source_rows), "legacy": path == legacy,
                       "families": dict(Counter(family(row) for row in source_rows)),
                       "providers": dict(Counter(row.get("source", "sec") for row in source_rows)),
                       "start": min((row["date"] for row in source_rows), default=None),
                       "end": max((row["date"] for row in source_rows), default=None)})
        rows.extend(source_rows)
    # Stable input order also makes the candidate index and union groups deterministic.
    rows.sort(key=lambda row: (row["date"], body_hash(row), not bool(row.get("document_id")), row.get("document_id") or ""))
    for row in rows:
        if date.fromisoformat(row["date"]).isoformat() != row["date"]:
            raise ValueError("Every row needs an ISO publication date")
    legacy_hashes = {body_hash(row) for row in legacy_rows}
    aliases, legacy_audit = legacy_links(rows, corpora, legacy_hashes)
    groups = cluster(rows, legacy_hashes, aliases)
    group_partitions = defaultdict(set)
    for i, row in enumerate(rows):
        group_partitions[groups[i][0]].add(partition(row["date"]))
    selected, excluded, exclusions_by_family, seen = defaultdict(list), Counter(), defaultdict(Counter), set()
    for i, row in enumerate(rows):
        name, digest = partition(row["date"]), body_hash(row)
        group, previous = groups[i]
        reason = None
        if len(group_partitions[group]) > 1:
            reason = "cross_boundary_group"
        elif name == "outside":
            reason = "outside_dates"
        elif name != "train" and previous:
            reason = "previously_experimented_group"
        elif digest in seen:
            reason = "exact_duplicate"
        if reason:
            excluded[reason] += 1
            exclusions_by_family[family(row)][reason] += 1
        else:
            seen.add(digest)
            if row["llm"]["label"] is not None and row["llm"].get("rubric_sha256") != RUBRIC_HASH:
                row = {**row, "previous_llm_annotation": row["llm"], "llm": {"label": None, "uncertainty": None}}
            selected[name].append({**row, "group_id": group, "previously_experimented": previous})
    maximums = {"train": train, "validation": validation, "test": test}
    details = {}
    for name in maximums:
        chosen = cap_primary(sample(selected[name], maximums[name]))
        write_records(chosen, output / f"{name}-input.jsonl")
        details[name] = {"available": len(selected[name]), "chosen": len(chosen),
                         "families": dict(Counter(family(r) for r in chosen)),
                         "labeled": sum(annotation(r)["label"] is not None for r in chosen)}
    save({"cutoffs": CUTOFFS, "near_duplicate_threshold": NEAR_THRESHOLD, "candidate_threshold": SEARCH_THRESHOLD,
          "rubric_sha256": RUBRIC_HASH,
          "num_perm": PERMUTATIONS, "primary_8k_share_cap": 0.4,
          "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*corpora, legacy]},
          "legacy_extraction_check": legacy_audit,
          "census": census, "excluded": dict(excluded),
          "exclusions_by_family": {name: dict(counts) for name, counts in exclusions_by_family.items()},
          "partitions": details,
          "method": "Temporal; event and verified near-duplicate groups; legacy excluded from validation/test"},
         output / "prepared.json")
    print(json.dumps(details), flush=True)


def freeze(output):
    prepared = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
    groups, coverage = {}, {}
    for name in ("train", "validation", "test"):
        inputs = read_records(output / f"{name}-input.jsonl")
        labeled = [row for row in inputs if annotation(row)["label"] is not None]
        missing = [row for row in inputs if annotation(row)["label"] is None]
        coverage[name] = {"queued": len(inputs), "labeled": len(labeled), "unlabeled": len(missing),
                          "unlabeled_by_family": dict(Counter(family(row) for row in missing)),
                          "unlabeled_errors": dict(Counter(row.get("label_error", {}).get("type", "Not labeled") for row in missing))}
        if name != "train" and missing:
            raise ValueError(f"Complete all selected {name} labels before freezing; {len(missing)} remain")
        if any(row["human"]["label"] is None and prepared.get("rubric_sha256") and
               row["llm"].get("rubric_sha256") != prepared["rubric_sha256"] for row in labeled):
            raise ValueError("Teacher labels must use the prepared benchmark's rubric")
        # Missing training labels can change the source mix; retain the cap after labeling.
        retained = cap_primary(labeled, by_label=name == "train")
        coverage[name].update(retained=len(retained), excluded_8k_share=len(labeled) - len(retained))
        labeled = retained
        teacher = [row["llm"] for row in labeled if row["llm"]["label"] is not None and
                   row["llm"].get("rubric_sha256") == prepared.get("rubric_sha256")]
        known_costs = [label["estimated_cost_usd"] for label in teacher if label.get("estimated_cost_usd") is not None]
        coverage[name]["successful_teacher_labels"] = len(teacher)
        coverage[name]["teacher_cost_usd"] = sum(known_costs)
        coverage[name]["teacher_cost_unavailable"] = len(teacher) - len(known_costs)
        if not labeled:
            raise ValueError(f"No labeled {name} documents")
        groups[name] = sorted(labeled, key=lambda r: (r["date"], body_hash(r)))
    group_sets = {name: {r["group_id"] for r in rows} for name, rows in groups.items()}
    if any(group_sets[a] & group_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Related groups cross partitions")
    # Partition validation by whole group, without allowing calibration data to enter model selection.
    group_keys = sorted(group_sets["validation"])
    selection_keys = set(group_keys[:max(1, int(len(group_keys) * 0.7))])
    selection = [r for r in groups["validation"] if r["group_id"] in selection_keys]
    calibration = [r for r in groups["validation"] if r["group_id"] not in selection_keys]
    if not calibration:
        raise ValueError("Need at least two independent validation groups")
    groups["selection"], groups["calibration"] = selection, calibration
    manifest, files = {"method": prepared["method"], "prepared": prepared, "label_coverage": coverage, "splits": {}}, {}
    for name, rows in groups.items():
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        files[f"{name}.jsonl"] = content
        manifest["splits"][name] = {"count": len(rows), "start": rows[0]["date"], "end": rows[-1]["date"],
                                     "sha256": hashlib.sha256(content).hexdigest(),
                                     "families": dict(Counter(family(r) for r in rows)),
                                     "labels": dict(Counter(annotation(r)["label"] for r in rows))}
    ordered = sample(groups["train"], by_label=True)
    subsets = {}
    for number in sorted(set([n for n in (250, 1000, 4000, 16000) if n <= len(ordered)] + [len(ordered)])):
        subsets[str(number)] = [body_hash(r) for r in ordered[:number]]
    files["subsets.json"] = (json.dumps({"seed": 42, "order": "Round-robin family/label; stable body-hash ordering",
                                       "subsets": subsets}, indent=2) + "\n").encode()
    manifest["subsets_sha256"] = hashlib.sha256(files["subsets.json"]).hexdigest()
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for name, content in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Frozen benchmark differs: {path}")
    for name, content in files.items():
        (output / name).write_bytes(content)
    print(json.dumps(manifest["splits"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, default=HERE / "data/dataset.json")
    parser.add_argument("--validation", type=int, default=1200)
    parser.add_argument("--test", type=int, default=1800)
    parser.add_argument("--train", type=int, default=25000)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.output)
    elif args.corpora:
        prepare(args.corpora, args.output, args.legacy, args.validation, args.test, args.train)
    else:
        parser.error("Supply --corpora or --freeze")
