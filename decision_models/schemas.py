"""Shared dataset and model-output schemas."""

from typing import Literal
import hashlib
import json

from pydantic import BaseModel


LabelName = Literal["routine", "review_worthy", "unclear"]
Uncertainty = Literal[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class Annotation(BaseModel):
    label: LabelName | None = None
    uncertainty: Uncertainty | None = None


class FilingRecord(BaseModel):
    date: str
    body: str
    llm: Annotation
    human: Annotation


class LabelOutput(BaseModel):
    label: LabelName
    uncertainty: Uncertainty


def load_split(directory, name):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    content = (directory / f"{name}.jsonl").read_bytes()
    expected = manifest["splits"][name]
    if hashlib.sha256(content).hexdigest() != expected["sha256"]:
        raise ValueError(f"Split checksum mismatch: {name}")
    rows = [json.loads(line) for line in content.decode().splitlines()]
    if len(rows) != expected["count"]:
        raise ValueError(f"Split count mismatch: {name}")
    return rows
