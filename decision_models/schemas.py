"""Shared dataset and model-output schemas."""

from typing import Literal

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
