"""Fit simple baselines on train; choose regularization, weights and decision offsets on validation."""

import argparse
import hashlib
import itertools
import json
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from benchmark import Benchmark
from label import annotation, save
from schemas import load_split


HERE = Path(__file__).resolve().parent
LABELS = Benchmark.labels
OFFSET_GRID = np.arange(-3., 3.01, 0.25)


def predictions(records, probabilities):
    return [{"body_sha256": hashlib.sha256(row["body"].encode()).hexdigest(),
             "reference": annotation(row)["label"], "prediction": LABELS[p.argmax()],
             "probabilities": p.tolist()} for row, p in zip(records, probabilities)]


def best_offsets(rows):
    """Search two relative decision offsets; probabilities remain uncalibrated."""
    logits = np.log(np.maximum([row["probabilities"] for row in rows], 1e-12))
    truth = np.array([LABELS.index(row["reference"]) for row in rows])
    best, offsets = f1_score(truth, logits.argmax(1), average="macro", labels=[0, 1, 2]), [0., 0., 0.]
    # Review-worthy stays fixed; two relative offsets are sufficient for three classes.
    for a, b in itertools.product(OFFSET_GRID, repeat=2):
        bias = [float(a), 0., float(b)]
        score = f1_score(truth, (logits+bias).argmax(1), average="macro", labels=[0, 1, 2])
        if score > best + 1e-12:
            best, offsets = score, bias
    return float(best), offsets


def tune_offsets(directory, splits=HERE / "data/splits"):
    result = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    rows = result["predictions"]
    reference = {hashlib.sha256(row["body"].encode()).hexdigest(): annotation(row)["label"]
                 for row in load_split(splits, "validation")}
    if len(rows) != len(reference) or {row["body_sha256"]: row["reference"] for row in rows} != reference:
        raise ValueError("Decision tuning requires the complete, unchanged validation split")
    best, offsets = best_offsets(rows)
    decision = {"offsets": offsets, "validation_macro_f1": best,
                "search_grid": {"minimum": float(OFFSET_GRID[0]), "maximum": float(OFFSET_GRID[-1]), "step": 0.25},
                "raw_macro_f1": result["macro_f1"], "selection": "validation only; relative log-probability offsets",
                "validation_sha256": hashlib.sha256((directory / "validation.json").read_bytes()).hexdigest()}
    artifact = next(directory/name for name in ("model.joblib", "best.pt", "config.json") if (directory/name).exists())
    decision["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    save(decision, directory / "decision.json")
    print(json.dumps(decision), flush=True)
    return decision


def blend(directories, output, splits):
    """Average two complementary models; select their mixing weight using validation only."""
    records = load_split(splits, "validation")
    hashes = [hashlib.sha256(row["body"].encode()).hexdigest() for row in records]
    probabilities, members = [], []
    manifest = json.loads((splits/"manifest.json").read_text(encoding="utf-8"))
    for directory in directories:
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        if config["split_manifest"] != manifest:
            raise ValueError("Blend members use different data splits")
        rows = json.loads((directory/"validation.json").read_text(encoding="utf-8"))["predictions"]
        by_hash = {row["body_sha256"]: row for row in rows}
        if set(by_hash) != set(hashes) or len(rows) != len(records):
            raise ValueError("Blend members need the entire validation split")
        if any(by_hash[h]["reference"] != annotation(row)["label"] for h, row in zip(hashes, records)):
            raise ValueError("Blend references differ from the frozen validation labels")
        probabilities.append(np.array([by_hash[h]["probabilities"] for h in hashes]))
        artifact = directory/("model.joblib" if config.get("family") == "linear" else "best.pt")
        members.append({"directory": directory.resolve().relative_to(HERE).as_posix(),
                        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
    output.mkdir(parents=True, exist_ok=True)
    best, trials = -1, []
    for weight in (0., 0.25, 0.5, 0.75, 1.):
        p = weight*probabilities[0] + (1-weight)*probabilities[1]
        rows = predictions(records, p)
        scores = Benchmark.scores(rows)
        tuned, offsets = best_offsets(rows)
        trials.append({"weights": [weight, 1-weight], "macro_f1": scores["macro_f1"],
                       "tuned_macro_f1": tuned, "offsets": offsets})
        if tuned > best:
            best = tuned
            save({"family": "blend", "kind": "blend", "members": members, "weights": [weight, 1-weight],
                  "labels": list(LABELS), "split_manifest": manifest}, output/"config.json")
            save({**scores, "predictions": rows}, output/"validation.json")
    save({"trials": trials}, output/"search.json")
    tune_offsets(output, splits)


def embeddings(kind, records, split, device):
    import torch
    from encoders import EncoderModel, MODELS

    torch.set_num_threads(4)
    fingerprint = hashlib.sha256(json.dumps({"bodies": [row["body"] for row in records],
        "model": MODELS[kind], "raw": True, "window": 510, "pooling": "mean of all windows, then L2",
        "device": device, "versions": {n: version(n) for n in ("torch", "transformers", "numpy")},
        "source_sha256": hashlib.sha256((HERE/"encoders.py").read_bytes()).hexdigest()}).encode()).hexdigest()
    cache = HERE / "data/cache" / f"{kind}-{split}-{fingerprint[:12]}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    vectors = []
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved["fingerprint"]) != fingerprint:
                raise ValueError(f"Embedding cache differs: {cache}")
            vectors = list(saved["vectors"])
    if len(vectors) < len(records):
        model = EncoderModel(device, kind=kind).eval()
        with torch.no_grad():
            for i in range(len(vectors), len(records)):
                vector = model.features(model.windows(records[i]["body"])).mean(0)
                vectors.append(torch.nn.functional.normalize(vector, dim=0).cpu().numpy())
                if len(vectors) % 250 == 0 or len(vectors) == len(records):
                    temporary = cache.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, vectors=np.array(vectors), fingerprint=fingerprint)
                    for attempt in range(5):
                        try:
                            temporary.replace(cache)
                            break
                        except PermissionError:
                            if attempt == 4:
                                raise
                            time.sleep(1)
                    print(f"{kind} {split}: {len(vectors)}/{len(records)}", flush=True)
    return np.array(vectors)


def fit(args):
    train, validation = [load_split(args.splits, name) for name in ("train", "validation")]
    y = [annotation(row)["label"] for row in train]
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "source"
    snapshot.mkdir(exist_ok=True)
    for name in ("tune.py", "encoders.py", "schemas.py", "benchmark.py", "label.py"):
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    started = time.perf_counter()
    if args.model in ("tfidf", "char"):
        transform = TfidfVectorizer(ngram_range=(1, 2) if args.model == "tfidf" else (3, 5),
            analyzer="word" if args.model == "tfidf" else "char_wb", min_df=2, max_features=100_000)
        x = transform.fit_transform([row["body"] for row in train])
        vx = transform.transform([row["body"] for row in validation])
    else:
        transform = StandardScaler()
        x = transform.fit_transform(embeddings(args.model, train, "train", args.device))
        vx = transform.transform(embeddings(args.model, validation, "validation", args.device))
    best, trials = -1, []
    for weight, c in itertools.product((None, "balanced"), args.regularization):
        model = LogisticRegression(C=c, class_weight=weight, max_iter=2000, random_state=42)
        with threadpool_limits(limits=4):
            model.fit(x, y)
        p = model.predict_proba(vx)[:, [list(model.classes_).index(label) for label in LABELS]]
        rows = predictions(validation, p)
        scores = Benchmark.scores(rows)
        tuned, offsets = best_offsets(rows)
        trials.append({"C": c, "class_weight": weight, "macro_f1": scores["macro_f1"],
                       "agreement": scores["agreement"], "tuned_macro_f1": tuned, "offsets": offsets})
        print(trials[-1], flush=True)
        if tuned > best:
            best = tuned
            joblib.dump(make_pipeline(transform, model), args.output / "model.joblib")
            save({**scores, "predictions": rows}, args.output / "validation.json")
            from encoders import MODELS
            model_id, revision = MODELS.get(args.model, (None, None))
            save({"kind": args.model, "family": "linear", "model": model_id, "revision": revision,
                "embedding_device": args.device if model_id else None,
                "selected": trials[-1], "labels": list(LABELS),
                "split_manifest": json.loads((args.splits / "manifest.json").read_text(encoding="utf-8")),
                "versions": {name: version(name) for name in ("scikit-learn", "numpy", "torch", "transformers")},
                "source_sha256": {name: hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
                                  for name in ("tune.py", "encoders.py", "schemas.py", "benchmark.py")}}, args.output / "config.json")
        save({"trials": trials, "seconds": time.perf_counter()-started}, args.output / "search.json")
    tune_offsets(args.output, args.splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tfidf", "char", "finbert", "bge"), default="tfidf")
    parser.add_argument("--splits", type=Path, default=HERE / "data/splits")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--offsets-only", action="store_true")
    parser.add_argument("--blend", type=Path, nargs=2, metavar="RUN", help="Blend two fitted run directories")
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1., 10.])
    args = parser.parse_args()
    if args.blend:
        blend(args.blend, args.output, args.splits)
    elif args.offsets_only:
        tune_offsets(args.output, args.splits)
    else:
        fit(args)
