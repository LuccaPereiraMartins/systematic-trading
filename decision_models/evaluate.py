"""Cross-validated comparison of the trainable baselines on all labelled filings.

The 50-filing pilot in ``benchmark.py`` is too small to rank anything: with 3 ``unclear`` rows and ±10-point
confidence intervals, most gaps are noise. Here every filing is predicted exactly once, by a model that never saw it
or any other filing from the same registrant (StratifiedGroupKFold), and each metric gets a bootstrap 95% CI over the
500 out-of-fold predictions. Filings are only ever compared with the reference label (human if present, else LLM), so
scores remain agreement with provisional labels, not accuracy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from baselines import APPROACHES, make, registrant
from benchmark import Benchmark

HERE = Path(__file__).resolve().parent
LABELS = Benchmark.labels
DEFAULT = ("majority", "length_raw", "length_item", "keyword_prior", "keyword_learned", "tfidf", "bge_lr", "finbert_lr")


def f1_scores(truth, prediction):
    """Per-label F1 for integer-coded arrays."""
    scores = []
    for index in range(len(LABELS)):
        tp = np.sum((truth == index) & (prediction == index))
        precision = tp / max(np.sum(prediction == index), 1)
        recall = tp / max(np.sum(truth == index), 1)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return np.array(scores)


def summarise(truth, prediction, resamples, seed=0):
    rng = np.random.default_rng(seed)
    agreement, macro = [], []
    for _ in range(resamples):
        pick = rng.integers(0, len(truth), len(truth))
        agreement.append(np.mean(truth[pick] == prediction[pick]))
        macro.append(f1_scores(truth[pick], prediction[pick]).mean())
    interval = lambda values: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]  # noqa: E731
    per_label = f1_scores(truth, prediction)
    return {
        "agreement": float(np.mean(truth == prediction)),
        "agreement_ci95": interval(agreement),
        "macro_f1": float(per_label.mean()),
        "macro_f1_ci95": interval(macro),
        "f1": dict(zip(LABELS, map(float, per_label))),
        "prediction_distribution": {label: int(np.sum(prediction == i)) for i, label in enumerate(LABELS)},
    }


def folds(benchmark, truth, splits, seed):
    groups = np.unique([registrant(r["body"]) for r in benchmark.records], return_inverse=True)[1]
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(truth)), truth, groups)), len(set(groups))


def evaluate(names, splits=5, seed=0, resamples=2000):
    benchmark = Benchmark()
    texts = [r["body"] for r in benchmark.records]
    reference = np.array([benchmark.reference(r) for r in benchmark.records])
    truth = np.array([LABELS.index(label) for label in reference])
    split_indices, companies = folds(benchmark, truth, splits, seed)
    print(f"{len(texts)} filings, {companies} registrants, {splits} folds (grouped by registrant)")
    results = {}
    for name in names:
        started = time.perf_counter()
        prediction = np.empty(len(texts), dtype=int)
        for train, test in split_indices:
            model = clone(make(name, cache=True))
            model.fit([texts[i] for i in train], reference[train])
            prediction[test] = [LABELS.index(p) for p in model.predict([texts[i] for i in test])]
        results[name] = {
            "description": APPROACHES[name][0],
            "seconds": time.perf_counter() - started,
            **summarise(truth, prediction, resamples),
            "predictions": [LABELS[p] for p in prediction],
        }
        row = results[name]
        print(
            f"{name:16s} agreement {row['agreement']:.3f} {row['agreement_ci95'][0]:.3f}-{row['agreement_ci95'][1]:.3f}"
            f" | macro F1 {row['macro_f1']:.3f} {row['macro_f1_ci95'][0]:.3f}-{row['macro_f1_ci95'][1]:.3f}"
            f" | {row['seconds']:.0f}s",
            flush=True,
        )
    return {"splits": splits, "seed": seed, "registrants": companies, "reference": reference.tolist(), "models": results}


def markdown(report):
    lines = [
        "| Approach | Agreement (95% CI) | Macro F1 (95% CI) | F1 routine / review / unclear |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in report["models"].items():
        a, m, f = row["agreement_ci95"], row["macro_f1_ci95"], row["f1"]
        lines.append(
            f"| {name} | {row['agreement']:.1%} ({a[0]:.1%}–{a[1]:.1%}) | {row['macro_f1']:.3f} ({m[0]:.3f}–{m[1]:.3f})"
            f" | {f['routine']:.2f} / {f['review_worthy']:.2f} / {f['unclear']:.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(APPROACHES), default=DEFAULT)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    report = evaluate(args.models, args.splits, args.seed, args.resamples)
    output = HERE / "benchmark_results" / "cv.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists():  # keep earlier runs of other approaches
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("seed") == report["seed"] and previous.get("splits") == report["splits"]:
            report["models"] = {**previous["models"], **report["models"]}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n" + markdown(report))
    print(f"Saved to {output}")
