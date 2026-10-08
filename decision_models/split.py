"""Freeze stratified train/validation/test JSONL files, keeping prior pilot rows in train."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from schemas import FilingRecord
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent


def split_dataset(dataset=HERE / "data/dataset.json", output=HERE / "data/splits", pilot=HERE / "data/splits/pilot.json"):
    source = dataset.read_bytes()
    records = [FilingRecord.model_validate(row).model_dump(exclude_unset=True) for row in json.loads(source)]

    def body_hash(row):
        return hashlib.sha256(row["body"].encode()).hexdigest()

    for row in records:
        if date.fromisoformat(row["date"]).isoformat() != row["date"]:
            raise ValueError(f"Expected an ISO date: {row['date']}")
    if len(records) != 10_000 or len({body_hash(row) for row in records}) != len(records):
        raise ValueError("Expected 10,000 records with unique bodies")
    if any((row["human"]["label"] or row["llm"]["label"]) is None for row in records):
        raise ValueError("Every record needs a reference label")
    pilot_source = pilot.read_bytes()
    seen_hashes = {body_hash(row) for row in json.loads(pilot_source)}
    seen = [row for row in records if body_hash(row) in seen_hashes]
    unseen = sorted((row for row in records if body_hash(row) not in seen_hashes), key=body_hash)

    def hold_out(rows):
        labels = [row["human"]["label"] or row["llm"]["label"] for row in rows]
        return train_test_split(rows, test_size=1000, random_state=42, stratify=labels)

    remaining, test = hold_out(unseen)
    train, validation = hold_out(remaining)
    groups = {"train": train + seen, "validation": validation, "test": test}
    assert len(groups["train"]) == 8000
    manifest = {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "pilot_sha256": hashlib.sha256(pilot_source).hexdigest(),
        "method": "Label-stratified with seed 42; hash-sorted input; prior pilot bodies reserved for train",
        "pilot_rows_reserved": len(seen),
        "splits": {},
    }
    files = {}
    for name, rows in groups.items():
        rows.sort(key=lambda row: (row["date"], body_hash(row)))
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        files[f"{name}.jsonl"] = content
        manifest["splits"][name] = {
            "count": len(rows),
            "start": rows[0]["date"],
            "end": rows[-1]["date"],
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    # Refuse to silently replace an established evaluation split.
    for name, content in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Frozen split differs: {path}; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output / name).write_bytes(content)
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE / "data/dataset.json")
    parser.add_argument("--output", type=Path, default=HERE / "data/splits")
    parser.add_argument("--pilot", type=Path, default=HERE / "data/splits/pilot.json")
    args = parser.parse_args()
    split_dataset(args.dataset, args.output, args.pilot)
