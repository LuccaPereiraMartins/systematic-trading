"""Prepare a fresh temporal benchmark, then freeze labeled splits and nested training subsets."""

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re

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


def cluster(rows, legacy_hashes):
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


def prepare(corpora, output, legacy=HERE / "data/dataset.json", validation=1200, test=1800, train=25000):
    if (output / "manifest.json").exists():
        raise ValueError("Benchmark is frozen; use another output directory")
    if (output / "prepared.json").exists():
        raise ValueError("Preparation already exists; label its inputs and use --freeze, or a new directory")
    legacy_rows = read_records(legacy)
    rows = [row for path in corpora for row in read_records(path)] + legacy_rows
    # Stable input order also makes the candidate index and union groups deterministic.
    rows.sort(key=lambda row: (row["date"], body_hash(row), row.get("document_id") or ""))
    for row in rows:
        if date.fromisoformat(row["date"]).isoformat() != row["date"]:
            raise ValueError("Every row needs an ISO publication date")
    groups = cluster(rows, {body_hash(row) for row in legacy_rows})
    group_partitions = defaultdict(set)
    for i, row in enumerate(rows):
        group_partitions[groups[i][0]].add(partition(row["date"]))
    selected, excluded, seen = defaultdict(list), Counter(), set()
    for i, row in enumerate(rows):
        name, digest = partition(row["date"]), body_hash(row)
        group, previous = groups[i]
        if len(group_partitions[group]) > 1:
            excluded["cross_boundary_group"] += 1
        elif name == "outside":
            excluded["outside_dates"] += 1
        elif name != "train" and previous:
            excluded["previously_experimented_group"] += 1
        elif digest in seen:
            excluded["exact_duplicate"] += 1
        else:
            seen.add(digest)
            selected[name].append({**row, "group_id": group, "previously_experimented": previous})
    maximums = {"train": train, "validation": validation, "test": test}
    details = {}
    for name in maximums:
        chosen = sample(selected[name], maximums[name])
        # Primary 8-Ks cannot consume a mixed corpus merely because they are easy to acquire.
        if name == "train":
            others = [r for r in chosen if family(r) != "8k"]
            primary = [r for r in chosen if family(r) == "8k"][:int(len(others) * 2 / 3)]
            chosen = sample(others + primary)
        write_records(chosen, output / f"{name}-input.jsonl")
        details[name] = {"available": len(selected[name]), "chosen": len(chosen),
                         "families": dict(Counter(family(r) for r in chosen)),
                         "labeled": sum(annotation(r)["label"] is not None for r in chosen)}
    save({"cutoffs": CUTOFFS, "near_duplicate_threshold": NEAR_THRESHOLD, "candidate_threshold": SEARCH_THRESHOLD,
          "num_perm": PERMUTATIONS,
          "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*corpora, legacy]},
          "excluded": dict(excluded), "partitions": details,
          "method": "Temporal; event and verified near-duplicate groups; legacy excluded from validation/test"},
         output / "prepared.json")
    print(json.dumps(details), flush=True)


def freeze(output):
    prepared = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
    groups = {}
    for name in ("train", "validation", "test"):
        inputs = read_records(output / f"{name}-input.jsonl")
        labeled = [row for row in inputs if annotation(row)["label"] is not None]
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
    manifest, files = {"method": prepared["method"], "prepared": prepared, "splits": {}}, {}
    for name, rows in groups.items():
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        files[f"{name}.jsonl"] = content
        manifest["splits"][name] = {"count": len(rows), "start": rows[0]["date"], "end": rows[-1]["date"],
                                     "sha256": hashlib.sha256(content).hexdigest(),
                                     "families": dict(Counter(family(r) for r in rows)),
                                     "labels": dict(Counter(annotation(r)["label"] for r in rows))}
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for name, content in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Frozen benchmark differs: {path}")
    for name, content in files.items():
        (output / name).write_bytes(content)
    ordered = sample(groups["train"], by_label=True)
    subsets = {}
    for number in sorted(set([n for n in (250, 1000, 4000, 16000) if n <= len(ordered)] + [len(ordered)])):
        subsets[str(number)] = [body_hash(r) for r in ordered[:number]]
    save({"seed": 42, "order": "Round-robin family/label; stable body-hash ordering", "subsets": subsets},
         output / "subsets.json")
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
