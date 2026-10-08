"""Raw-document baselines; encoder features come from encoders.py, never filing-specific cleaning."""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


FITTED_MODELS = ("majority", "tfidf", "char", "combined", "finbert", "bge")


def make(kind, C=1.0, class_weight=None, max_iter=2000, classifier="logistic"):
    """Return an unfitted pipeline; FinBERT/BGE inputs are document vectors, others are raw bodies."""
    if kind not in FITTED_MODELS:
        raise ValueError(f"Unknown baseline: {kind}")
    if kind == "majority":
        return DummyClassifier(strategy="most_frequent")
    if kind == "combined":
        features = FeatureUnion([(name, make(name)[0]) for name in ("tfidf", "char")])
    elif kind in ("tfidf", "char"):
        features = TfidfVectorizer(
            analyzer="word" if kind == "tfidf" else "char_wb",
            ngram_range=(1, 2) if kind == "tfidf" else (3, 5),
            min_df=2,
            max_features=100_000,
        )
    else:
        features = StandardScaler()
    estimator = {"logistic": LogisticRegression, "svm": LinearSVC}[classifier]
    return make_pipeline(features, estimator(C=C, class_weight=class_weight, max_iter=max_iter, random_state=42))
