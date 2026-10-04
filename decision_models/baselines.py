"""Baselines for 8-K triage, from trivial to fine-tuned encoders.

Every approach is an sklearn-style estimator: ``fit(texts, labels)`` / ``predict(texts)``, built by ``make(name)``.

How the simple baselines were chosen
------------------------------------
Substantive text. 8-Ks open with a cover page and end with a signature block, and most of the middle is boilerplate
that says nothing about the event. ``substantive_text`` keeps what lies between the first "Item X.XX" heading and
"SIGNATURES" (falling back to the whole body when there is no heading, 13/500 filings). Every text-based baseline
uses it, as does the length baseline when ``measure="substantive"``.

Length. On the 500 labelled filings the review-worthy share rises monotonically with length, from about 0.55 in the
shortest octile to about 0.9 in the longest, and every ``unclear`` filing is short. A monotone relationship needs one
threshold, so the model is a single cut on log length: shorter is ``routine``, longer is ``review_worthy``, and the
cut is the training-set value (searched over the 5th to 95th percentiles) that maximises macro F1. A first attempt,
a Gini stump with balanced class weights, spent its one split isolating the rare short ``unclear`` class and
scored below the majority baseline, so the direction is fixed by the observation above instead of left to the tree.
Raw and substantive length are both offered; cross-validation, not intuition, says which is better.

Keywords. Two variants, deliberately different in where the words come from:
* ``keyword_prior``: a hand-written lexicon of terms that signal a material 8-K event (M&A, financing, distress,
  officer departures, restatements, litigation, regulatory outcomes) and a short one for routine housekeeping
  (Reg FD furnishings, annual meeting votes, routine distributions). It was written from domain knowledge, before
  any label-word statistics were computed. The only fitted parameters are the 3 weights of a logistic regression on
  the two distinct-term counts (distinct terms, not occurrences, so long filings are not rewarded twice).
* ``keyword_learned``: binary uni/bi-gram presence, the top-k terms by chi-squared against the label, then
  logistic regression. k is chosen by 3-fold inner cross-validation (macro F1) from a small grid, never on the
  evaluation fold. Comparing it to the prior lexicon shows how much a data-driven vocabulary adds.

None of the cheap baselines can tell why a filing is review-worthy; they set the floor a learned or LLM model must
clear to justify its cost.
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
    "merger", "acquisition", "acquire", "business combination", "definitive agreement", "tender offer",
    "spin-off", "divest", "sale of substantially all", "joint venture", "license agreement", "collaboration",
    "credit agreement", "indenture", "senior notes", "convertible", "private placement", "public offering",
    "underwriting agreement", "default", "acceleration", "bankruptcy", "chapter 11", "going concern", "covenant",
    "delist", "deficiency", "forbearance", "waiver", "resign", "terminat", "dismiss", "removal", "departure",
    "interim chief", "restat", "non-reliance", "material weakness", "impairment", "investigation", "subpoena",
    "lawsuit", "litigation", "settlement", "complaint", "guidance", "preliminary results", "net loss",
    "share repurchase", "stock split", "reverse stock split", "fda", "clinical trial", "complete response",
    "recall", "cyber", "data breach",
)
ROUTINE_TERMS = (
    "regulation fd", "investor presentation", "furnished", "annual meeting", "submission of matters",
    "results of the vote", "bylaws", "regular quarterly", "distribution", "press release", "conference call",
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 10**6  # we window by hand; silence the length warning
    return tokenizer, AutoModel.from_pretrained(model_name).to(device).eval()


_EMBEDDINGS = {}


class Embedder(BaseEstimator, TransformerMixin):
    """Frozen encoder: windows the substantive text, pools each window, and averages the windows.

    ``cache=True`` memoises embeddings in-process so cross-validation encodes each filing once; leave it off when
    timing inference.
    """

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", pooling="cls", max_chunks=4, window=510, cache=False):
        self.model_name = model_name
        self.pooling = pooling
        self.max_chunks = max_chunks
        self.window = window
        self.cache = cache

    def fit(self, X, y=None):
        return self

    def embed(self, text):
        import torch

        device = best_device()
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
            key = (self.model_name, self.max_chunks, hashlib.sha256(text.encode()).hexdigest())
            if self.cache and key in _EMBEDDINGS:
                vectors.append(_EMBEDDINGS[key])
                continue
            vector = self.embed(text)
            if self.cache:
                _EMBEDDINGS[key] = vector
            vectors.append(vector)
        return np.vstack(vectors)


class FineTunedEncoder(ClassifierMixin, BaseEstimator):
    """End-to-end fine-tune of an encoder (default FinBERT) on the first ``max_length`` substantive tokens.

    Each ``fit`` starts again from the pretrained weights. Class-weighted cross-entropy handles the 69% / 27% / 5%
    class imbalance. This is the like-for-like comparison for SFT on Laya: same data, same labels, a small model.
    """

    def __init__(self, model_name="ProsusAI/finbert", max_length=512, epochs=3, lr=2e-5, batch_size=8, seed=0):
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def encode(self, tokenizer, texts, device):
        batch = tokenizer(
            [substantive_text(text) for text in texts],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        return batch.to(device)

    def fit(self, X, y):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.array(sorted(set(y)))
        targets = np.searchsorted(self.classes_, y)
        device = best_device()
        self.device_ = device
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.classes_), ignore_mismatched_sizes=True
        ).to(device)
        counts = np.bincount(targets, minlength=len(self.classes_))
        weights = torch.tensor(len(targets) / (len(self.classes_) * np.maximum(counts, 1)), dtype=torch.float, device=device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * -(-len(X) // self.batch_size)
        schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
        self.model_.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(order), self.batch_size):
                index = order[start : start + self.batch_size]
                batch = self.encode(self.tokenizer_, [X[i] for i in index], device)
                logits = self.model_(**batch).logits
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.tensor(targets[index], device=device), weight=weights
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
        self.model_.eval()
        return self

    def predict(self, X):
        import torch

        predictions = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = self.encode(self.tokenizer_, list(X[start : start + self.batch_size]), self.device_)
                predictions.extend(self.model_(**batch).logits.argmax(-1).cpu().tolist())
        return self.classes_[predictions]


def embedding_pipeline(model_name, pooling, cache):
    return make_pipeline(
        Embedder(model_name, pooling, cache=cache),
        StandardScaler(),
        inner_search(
            LogisticRegression(max_iter=2000, class_weight="balanced"), {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
        ),
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
            FunctionTransformer(lambda texts: [substantive_text(t) for t in texts]),
            CountVectorizer(binary=True, ngram_range=(1, 2), min_df=3, stop_words="english"),
            inner_search(
                make_pipeline(SelectKBest(chi2), LogisticRegression(max_iter=1000, class_weight="balanced")),
                {"selectkbest__k": [10, 25, 50, 100, 250]},
            ),
        ),
    ),
    "tfidf": (
        "TF-IDF uni/bi-grams on the full body + balanced logistic regression (as in the original benchmark)",
        lambda cache: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    ),
    "bge_lr": (
        "Frozen bge-small-en-v1.5 window embeddings + logistic regression",
        lambda cache: embedding_pipeline("BAAI/bge-small-en-v1.5", "cls", cache),
    ),
    "finbert_lr": (
        "Frozen ProsusAI/finbert window embeddings + logistic regression",
        lambda cache: embedding_pipeline("ProsusAI/finbert", "mean", cache),
    ),
    "finbert_ft": (
        "ProsusAI/finbert fine-tuned end to end (3 epochs, class-weighted)",
        lambda cache: FineTunedEncoder(),
    ),
}


def make(name, cache=False):
    if name not in APPROACHES:
        raise ValueError(f"Unknown baseline: {name}")
    return APPROACHES[name][1](cache)
