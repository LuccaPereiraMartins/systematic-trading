"""Majority, length, keyword, TF-IDF and capped/cleaned encoder baselines.

Factories provide sklearn estimators. tune.py selects parameters on our frozen validation
split; evaluate.py runs the common held-out test. Historical trials are in experiments.md.
"""

import hashlib
import re
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ITEM_HEADING = re.compile(r"^[ \t]*Item\s+\d\.\d\d\b", re.M | re.I)
SIGNATURES = re.compile(r"^[ \t]*SIGNATURES?[ \t]*$", re.M)
REGISTRANT_MARKER = re.compile(r"\(\s*Exact name of registrant", re.I)


def substantive_text(body):
    """The text between the first Item heading and the signature block (whole body if no heading)."""
    heading = ITEM_HEADING.search(body)
    start = heading.start() if heading else 0
    end = SIGNATURES.search(body, start)
    return body[start : end.start() if end else len(body)]


def substantive_texts(texts):
    return [substantive_text(text) for text in texts]


def registrant(body):
    """Best-effort company name, used to keep a company's filings in one CV fold. Falls back to a body hash."""
    marker = REGISTRANT_MARKER.search(body)
    if marker:
        lines = [line.strip() for line in body[: marker.start()].splitlines()]
        lines = [line for line in lines if re.search(r"[A-Za-z]", line)]
        if lines:
            return re.sub(r"[^a-z0-9]+", " ", lines[-1].casefold()).strip()
    return hashlib.sha256(body.encode()).hexdigest()


MATERIAL_TERMS = (
    "merger",
    "acquisition",
    "acquire",
    "business combination",
    "definitive agreement",
    "tender offer",
    "spin-off",
    "divest",
    "sale of substantially all",
    "joint venture",
    "license agreement",
    "collaboration",
    "credit agreement",
    "indenture",
    "senior notes",
    "convertible",
    "private placement",
    "public offering",
    "underwriting agreement",
    "default",
    "acceleration",
    "bankruptcy",
    "chapter 11",
    "going concern",
    "covenant",
    "delist",
    "deficiency",
    "forbearance",
    "waiver",
    "resign",
    "terminat",
    "dismiss",
    "removal",
    "departure",
    "interim chief",
    "restat",
    "non-reliance",
    "material weakness",
    "impairment",
    "investigation",
    "subpoena",
    "lawsuit",
    "litigation",
    "settlement",
    "complaint",
    "guidance",
    "preliminary results",
    "net loss",
    "share repurchase",
    "stock split",
    "reverse stock split",
    "fda",
    "clinical trial",
    "complete response",
    "recall",
    "cyber",
    "data breach",
)
ROUTINE_TERMS = (
    "regulation fd",
    "investor presentation",
    "furnished",
    "annual meeting",
    "submission of matters",
    "results of the vote",
    "bylaws",
    "regular quarterly",
    "distribution",
    "press release",
    "conference call",
    "webcast",
)


class LengthFeature(BaseEstimator, TransformerMixin):
    """log(1 + characters), of the raw body or its substantive part."""

    def __init__(self, measure="substantive"):
        self.measure = measure

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = [substantive_text(text) if self.measure == "substantive" else text for text in X]
        return np.log1p([[len(text)] for text in texts])


class LengthThreshold(ClassifierMixin, BaseEstimator):
    """One cut on a single feature: below it ``short_label``, at or above it ``long_label``."""

    def __init__(self, short_label="routine", long_label="review_worthy"):
        self.short_label = short_label
        self.long_label = long_label

    def fit(self, X, y):
        from sklearn.metrics import f1_score

        values, y = np.ravel(X), np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        best = (-1.0, np.inf)
        for cut in np.unique(np.percentile(values, np.arange(5, 96))):
            predicted = np.where(values < cut, self.short_label, self.long_label)
            best = max(best, (f1_score(y, predicted, average="macro", labels=self.classes_), cut), key=lambda b: b[0])
        self.threshold_ = best[1]
        return self

    def predict(self, X):
        return np.where(np.ravel(X) < self.threshold_, self.short_label, self.long_label)

    def predict_proba(self, X):
        return (self.predict(X)[:, None] == self.classes_[None, :]).astype(float)


class LexiconCounts(BaseEstimator, TransformerMixin):
    """Distinct material-term and routine-term hits (prefix match on word starts)."""

    def __init__(self, material=MATERIAL_TERMS, routine=ROUTINE_TERMS):
        self.material = material
        self.routine = routine

    @staticmethod
    def pattern(terms):
        return re.compile(r"\b(?:" + "|".join(re.escape(term) for term in terms) + ")", re.I)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        material, routine = self.pattern(self.material), self.pattern(self.routine)
        rows = []
        for text in X:
            text = substantive_text(text)
            rows.append(
                [len({m.lower() for m in material.findall(text)}), len({m.lower() for m in routine.findall(text)})]
            )
        return np.log1p(rows)


def inner_search(estimator, grid):
    return GridSearchCV(
        estimator, grid, scoring="f1_macro", cv=StratifiedKFold(3, shuffle=True, random_state=0), n_jobs=1
    )


def best_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def load_encoder(model_name, device):
    from transformers import AutoModel, AutoTokenizer

    from encoders import MODELS

    revision = next(pin for name, pin in MODELS.values() if name == model_name)
    torch = __import__("torch")
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.model_max_length = 10**6  # we window by hand; silence the length warning
    return tokenizer, AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()


_EMBEDDINGS = {}


class Embedder(BaseEstimator, TransformerMixin):
    """Frozen encoder: windows the substantive text, pools each window, and averages the windows.

    ``cache=True`` memoises embeddings in-process so cross-validation encodes each filing once; leave it off when
    timing inference.
    """

    def __init__(
        self, model_name="BAAI/bge-small-en-v1.5", pooling="cls", max_chunks=4, window=510, cache=False, device=None
    ):
        self.model_name = model_name
        self.pooling = pooling
        self.max_chunks = max_chunks
        self.window = window
        self.cache = cache
        self.device = device

    def fit(self, X, y=None):
        return self

    def embed(self, text):
        import torch

        device = self.device or best_device()
        tokenizer, model = load_encoder(self.model_name, device)
        ids = tokenizer(substantive_text(text), add_special_tokens=False)["input_ids"]
        windows = [ids[i : i + self.window] for i in range(0, max(len(ids), 1), self.window)][: self.max_chunks]
        rows = [[tokenizer.cls_token_id, *window, tokenizer.sep_token_id] for window in windows]
        width = max(len(row) for row in rows)
        batch = {
            "input_ids": torch.tensor([row + [tokenizer.pad_token_id] * (width - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        }
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        vector = pooled.mean(0)
        return torch.nn.functional.normalize(vector, dim=0).cpu().numpy()

    def transform(self, X):
        vectors = []
        for text in X:
            key = (
                self.model_name,
                self.pooling,
                self.window,
                self.max_chunks,
                self.device or best_device(),
                hashlib.sha256(text.encode()).hexdigest(),
            )
            if self.cache and key in _EMBEDDINGS:
                vectors.append(_EMBEDDINGS[key])
                continue
            vector = self.embed(text)
            if self.cache:
                _EMBEDDINGS[key] = vector
            vectors.append(vector)
        return np.vstack(vectors)


def embedding_pipeline(model_name, pooling, cache):
    return make_pipeline(
        Embedder(model_name, pooling, cache=cache),
        StandardScaler(),
        inner_search(LogisticRegression(max_iter=2000, class_weight="balanced"), {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
    )


# name -> (description, factory(cache) -> estimator)
APPROACHES = {
    "majority": ("Always the most common training label", lambda cache: DummyClassifier(strategy="most_frequent")),
    "length_raw": (
        "One length cut on the raw body: short routine, long review_worthy",
        lambda cache: make_pipeline(LengthFeature("raw"), LengthThreshold()),
    ),
    "length_item": (
        "One length cut on the substantive (Item to signature) text",
        lambda cache: make_pipeline(LengthFeature("substantive"), LengthThreshold()),
    ),
    "keyword_prior": (
        "Hand-written material/routine lexicon -> 2 distinct-hit counts -> logistic regression",
        lambda cache: make_pipeline(LexiconCounts(), LogisticRegression(class_weight="balanced")),
    ),
    "keyword_learned": (
        "Top-k chi2 uni/bi-grams (k by inner CV) -> logistic regression",
        lambda cache: make_pipeline(
            FunctionTransformer(substantive_texts),
            CountVectorizer(binary=True, ngram_range=(1, 2), min_df=3, stop_words="english"),
            inner_search(
                make_pipeline(SelectKBest(chi2), LogisticRegression(max_iter=1000, class_weight="balanced")),
                {"selectkbest__k": [10, 25, 50, 100, 250]},
            ),
        ),
    ),
    "tfidf_balanced": (
        "TF-IDF uni/bi-grams on the full body + balanced logistic regression (as in the original benchmark)",
        lambda cache: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    ),
    "bge_capped": (
        "Frozen bge-small-en-v1.5 window embeddings + logistic regression",
        lambda cache: embedding_pipeline("BAAI/bge-small-en-v1.5", "cls", cache),
    ),
    "finbert_capped": (
        "Frozen ProsusAI/finbert window embeddings + logistic regression",
        lambda cache: embedding_pipeline("ProsusAI/finbert", "mean", cache),
    ),
}


FITTED_MODELS = (*APPROACHES, "tfidf", "char", "finbert", "bge")


def make(name, cache=False):
    if name not in APPROACHES:
        raise ValueError(f"Unknown baseline: {name}")
    return APPROACHES[name][1](cache)
