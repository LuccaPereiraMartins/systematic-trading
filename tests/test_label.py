import pytest

pytest.importorskip("openai")

from label import annotation


def test_human_label_overrides_llm():
    record = {"human": {"label": "routine"}, "llm": {"label": "review_worthy"}}
    assert annotation(record)["label"] == "routine"


def test_falls_back_to_llm_label():
    record = {"human": {"label": None}, "llm": {"label": "unclear"}}
    assert annotation(record)["label"] == "unclear"
