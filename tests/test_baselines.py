import pytest

pytest.importorskip("sklearn")

import numpy as np
from baselines import LengthFeature, LengthThreshold, LexiconCounts, make, registrant, substantive_text

FILING = """FORM 8-K
Acme Corp.
(Exact name of registrant as specified in its charter)
cover page boilerplate

Item 1.01 Entry into a Material Definitive Agreement.
The company signed a merger agreement.

SIGNATURES
By: /s/ Someone
"""


def test_substantive_text_runs_from_item_heading_to_signatures():
    text = substantive_text(FILING)
    assert text.startswith("Item 1.01")
    assert "merger agreement" in text
    assert "cover page" not in text and "Someone" not in text


def test_substantive_text_without_heading_keeps_whole_body():
    assert substantive_text("no headings here") == "no headings here"


def test_registrant_reads_name_above_marker_and_falls_back_to_hash():
    assert registrant(FILING) == "acme corp"
    assert registrant("unstructured text") == registrant("unstructured text")
    assert registrant("unstructured text") != registrant("other text")


def test_length_feature_measures_requested_span():
    raw = LengthFeature("raw").transform([FILING])[0, 0]
    substantive = LengthFeature("substantive").transform([FILING])[0, 0]
    assert raw > substantive


def test_lexicon_counts_distinct_terms_not_occurrences():
    text = "Item 1.01 merger merger merger and a bankruptcy. Regulation FD."
    material, routine = LexiconCounts().transform([text])[0]
    assert material == pytest.approx(np.log1p(2))
    assert routine == pytest.approx(np.log1p(1))


def test_length_threshold_cuts_between_classes():
    X = np.log1p([[10], [20], [30], [1000], [2000], [3000]])
    y = ["routine"] * 3 + ["review_worthy"] * 3
    model = LengthThreshold().fit(X, y)
    assert list(model.predict(np.log1p([[15], [2500]]))) == ["routine", "review_worthy"]


@pytest.mark.parametrize(
    "name", ["majority", "length_raw", "length_item", "keyword_prior", "keyword_learned", "tfidf_balanced"]
)
def test_cheap_baselines_fit_and_predict(name):
    routine = [f"Item 7.01 Regulation FD furnished investor presentation {i}" for i in range(12)]
    material = [f"Item 1.01 merger agreement default bankruptcy covenant credit agreement {i} " * 5 for i in range(12)]
    texts, labels = routine + material, ["routine"] * 12 + ["review_worthy"] * 12
    model = make(name).fit(texts, labels)
    assert set(model.predict(texts)) <= {"routine", "review_worthy"}


def test_unknown_baseline_rejected():
    with pytest.raises(ValueError):
        make("nope")
