"""Label unlabeled financial texts in dataset.json."""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


DATASET = Path(__file__).with_name("dataset.json")
MODEL = "gpt-6-luna"
RUBRIC = """Classify whether this financial text warrants an investment analyst's closer review.
routine: ordinary updates with no apparent development requiring closer review.
review_worthy: a potentially significant development that merits closer review.
unclear: insufficient or conflicting information to decide.
Uncertainty is your uncertainty about this label: 0.0 means certain, 1.0 means very uncertain.
Use only increments of 0.1. Judge the supplied text alone. Return only label and uncertainty."""
LABELS = {"routine", "review_worthy", "unclear"}
UNCERTAINTIES = [round(i / 10, 1) for i in range(11)]


def annotation(record):
    return record["human"] if record["human"]["label"] is not None else record["llm"]


def save(records):
    temporary = DATASET.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(DATASET)


def valid(value):
    return (value["label"] in LABELS and value["uncertainty"] in UNCERTAINTIES
            and type(value["uncertainty"]) in (int, float))


def label_document(client, body):
    response = client.responses.create(
        model=MODEL,
        service_tier="flex",
        reasoning={"effort": "low"},
        instructions=RUBRIC,
        input=body,
        text={"format": {"type": "json_schema", "name": "filing_label", "strict": True,
                         "schema": {"type": "object", "additionalProperties": False,
                                    "properties": {"label": {"type": "string", "enum": sorted(LABELS)},
                                                   "uncertainty": {"type": "number", "enum": UNCERTAINTIES}},
                                    "required": ["label", "uncertainty"]}}},
    )
    if response.status != "completed" or any(
        item.type == "refusal" for output in response.output for item in getattr(output, "content", [])
    ):
        raise ValueError(f"Model did not return a label: {response.status}")
    result = json.loads(response.output_text)
    if set(result) != {"label", "uncertainty"} or not valid(result):
        raise ValueError(f"Invalid model label: {result}")
    return result


def label_dataset():
    load_dotenv(Path(__file__).with_name(".env"))
    records = json.loads(DATASET.read_text(encoding="utf-8"))
    pending = [i for i, record in enumerate(records) if annotation(record)["label"] is None]
    if not pending:
        print("No unlabeled records")
        return records
    client = OpenAI(timeout=900.0, max_retries=3)
    for i in pending:
        try:
            result = label_document(client, records[i]["body"])
        except Exception as exc:
            raise RuntimeError(f"Stopped at record {i + 1}; previous labels are saved: {exc}") from exc
        records[i]["llm"] = result
        save(records)
        print(f"Labeled {i + 1}/{len(records)}: {result['label']}, uncertainty {result['uncertainty']:.1f}")
    return records


if __name__ == "__main__":
    label_dataset()
