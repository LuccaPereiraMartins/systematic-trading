"""Shared data schemas, labels and Laya task settings."""

from typing import Literal, get_args
import hashlib
import json
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict


LabelName = Literal["routine", "review_worthy", "unclear"]
LABELS = get_args(LabelName)

# One pinned model and task definition for base inference and supervised adaptation.
LAYA_MODEL = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
LAYA_QUESTIONS = {
    "triage": {
        "type": "choice",
        "instructions": "Classify this financial document for whether an investment analyst should spend time reviewing it.",
        "criteria": {
            "routine": "Ordinary update with no apparent development requiring closer review.",
            "review_worthy": "A potentially significant development that merits closer review.",
            "unclear": "Insufficient or conflicting information to decide.",
        },
    }
}

Uncertainty = Literal[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class Annotation(BaseModel):
    model_config = ConfigDict(extra="allow")
    label: LabelName | None = None
    uncertainty: Uncertainty | None = None


class FilingRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    date: str
    body: str
    llm: Annotation
    human: Annotation
    document_id: str | None = None
    url: str | None = None
    source: str = "sec"
    document_type: str = "8-K"
    event_id: str | None = None
    issuer: str | None = None
    date_precision: Literal["day", "second"] = "day"
    provenance: dict = {}


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


HERE = Path(__file__).resolve().parent
DATASET = HERE / "data/dataset.json"
SPLITS = HERE / "data/splits"
API_MODEL = HERE / "data/api_model.joblib"


def annotation(record):
    return record["human"] if record["human"]["label"] is not None else record["llm"]


def body_hash(record):
    return hashlib.sha256(record["body"].encode()).hexdigest()


def read_records(path):
    """Read the legacy JSON array or a resumable JSONL corpus without dropping metadata."""
    path = Path(path)
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    rows = json.loads(content) if path.suffix == ".json" else [json.loads(line) for line in content.splitlines() if line]
    return [FilingRecord.model_validate(row).model_dump(exclude_unset=True) for row in rows]


def write_records(records, path):
    path = Path(path)
    if path.suffix == ".json":
        return save(records, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    replace_file(temporary, path)


def replace_file(temporary, destination):
    """Atomic replacement, with brief retries for OneDrive file locks."""
    for attempt in range(5):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(1)


def save(value, path=DATASET):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace_file(temporary, path)
