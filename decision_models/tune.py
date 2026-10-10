"""Fit simple baselines on train; choose regularization, weights and decision offsets on validation."""

import argparse
from contextlib import closing
import hashlib
import itertools
import json
import sqlite3
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from baselines import FITTED_MODELS, make
from benchmark import Benchmark
from schemas import LABELS, annotation, body_hash, load_split, save


HERE = Path(__file__).resolve().parent
OFFSET_GRID = np.arange(-3.0, 3.01, 0.25)


def predictions(records, probabilities):
    return [
        {
            "body_sha256": hashlib.sha256(row["body"].encode()).hexdigest(),
            "reference": annotation(row)["label"],
            "prediction": LABELS[p.argmax()],
            "probabilities": p.tolist(),
        }
        for row, p in zip(records, probabilities)
    ]


def best_offsets(rows):
    """Search two relative decision offsets; probabilities remain uncalibrated."""
    logits = np.log(np.maximum([row["probabilities"] for row in rows], 1e-12))
    truth = np.array([LABELS.index(row["reference"]) for row in rows])
    best, offsets = f1_score(truth, logits.argmax(1), average="macro", labels=[0, 1, 2]), [0.0, 0.0, 0.0]
    # Review-worthy stays fixed; two relative offsets are sufficient for three classes.
    for a, b in itertools.product(OFFSET_GRID, repeat=2):
        bias = [float(a), 0.0, float(b)]
        score = f1_score(truth, (logits + bias).argmax(1), average="macro", labels=[0, 1, 2])
        if score > best + 1e-12:
            best, offsets = score, bias
    return float(best), offsets


def tune_offsets(directory, splits=HERE / "data/splits"):
    result = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    rows = result["predictions"]
    reference = {
        hashlib.sha256(row["body"].encode()).hexdigest(): annotation(row)["label"]
        for row in load_split(splits, "validation")
    }
    if len(rows) != len(reference) or {row["body_sha256"]: row["reference"] for row in rows} != reference:
        raise ValueError("Decision tuning requires the complete, unchanged validation split")
    best, offsets = best_offsets(rows)
    decision = {
        "offsets": offsets,
        "validation_macro_f1": best,
        "search_grid": {"minimum": float(OFFSET_GRID[0]), "maximum": float(OFFSET_GRID[-1]), "step": 0.25},
        "raw_macro_f1": result["macro_f1"],
        "selection": "validation only; relative log-probability offsets",
        "validation_sha256": hashlib.sha256((directory / "validation.json").read_bytes()).hexdigest(),
    }
    artifact = next(
        directory / name for name in ("model.joblib", "best.pt", "config.json") if (directory / name).exists()
    )
    decision["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    save(decision, directory / "decision.json")
    print(json.dumps(decision), flush=True)
    return decision


def blend(directories, output, splits):
    """Average two complementary models; select their mixing weight using validation only."""
    records = load_split(splits, "validation")
    hashes = [hashlib.sha256(row["body"].encode()).hexdigest() for row in records]
    probabilities, members = [], []
    manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    for directory in directories:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["split_manifest"] != manifest:
            raise ValueError("Blend members use different data splits")
        rows = json.loads((directory / "validation.json").read_text(encoding="utf-8"))["predictions"]
        by_hash = {row["body_sha256"]: row for row in rows}
        if set(by_hash) != set(hashes) or len(rows) != len(records):
            raise ValueError("Blend members need the entire validation split")
        if any(by_hash[h]["reference"] != annotation(row)["label"] for h, row in zip(hashes, records)):
            raise ValueError("Blend references differ from the frozen validation labels")
        probabilities.append(np.array([by_hash[h]["probabilities"] for h in hashes]))
        artifact = directory / ("model.joblib" if config.get("family") in ("linear", "baseline") else "best.pt")
        members.append(
            {
                "directory": directory.resolve().relative_to(HERE).as_posix(),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    best, trials = -1, []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = weight * probabilities[0] + (1 - weight) * probabilities[1]
        rows = predictions(records, p)
        scores = Benchmark.scores(rows)
        tuned, offsets = best_offsets(rows)
        trials.append(
            {
                "weights": [weight, 1 - weight],
                "macro_f1": scores["macro_f1"],
                "tuned_macro_f1": tuned,
                "offsets": offsets,
            }
        )
        if tuned > best:
            best = tuned
            save(
                {
                    "family": "blend",
                    "kind": "blend",
                    "members": members,
                    "weights": [weight, 1 - weight],
                    "labels": list(LABELS),
                    "split_manifest": manifest,
                },
                output / "config.json",
            )
            save({**scores, "predictions": rows}, output / "validation.json")
    save({"trials": trials}, output / "search.json")
    tune_offsets(output, splits)


def embeddings(kind, records, split, device, details=None):
    import torch
    from encoders import EncoderModel, MODELS

    torch.set_num_threads(4)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": MODELS[kind],
                "cache_format": "document-f32-v1",
                "raw": True,
                "window": 510,
                "pooling": "mean of all windows, then L2",
                "device": device,
                "versions": {n: version(n) for n in ("torch", "transformers", "numpy")},
                "source_sha256": hashlib.sha256((HERE / "encoders.py").read_bytes()).hexdigest(),
            }
        ).encode()
    ).hexdigest()
    cache = HERE / "data/cache" / f"{kind}-documents-{fingerprint[:12]}.sqlite"
    cache.parent.mkdir(parents=True, exist_ok=True)
    vectors, model = [], None
    stats = {"documents": len(records), "computed": 0, "reused": 0, "recorded_compute_seconds": 0.0,
             "peak_vram_bytes": 0, "model_loading_seconds": 0.0, "cache_fingerprint": fingerprint}
    started = time.perf_counter()
    with closing(sqlite3.connect(cache, timeout=30)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS recipe (fingerprint TEXT PRIMARY KEY, width INTEGER)")
        db.execute("CREATE TABLE IF NOT EXISTS features (body TEXT PRIMARY KEY, vector BLOB, sha TEXT, seconds REAL, vram INTEGER)")
        recipe = db.execute("SELECT fingerprint,width FROM recipe").fetchall()
        if recipe and (len(recipe) != 1 or recipe[0][0] != fingerprint):
            raise ValueError(f"Embedding cache differs: {cache}")
        width = recipe[0][1] if recipe else None
        for row in records:
            digest = body_hash(row)
            saved = db.execute("SELECT vector,sha,seconds,vram FROM features WHERE body=?", (digest,)).fetchone()
            if saved:
                blob, checksum, seconds, vram = saved
                if hashlib.sha256(blob).hexdigest() != checksum:
                    raise ValueError(f"Cached feature checksum differs: {digest}")
                vector = np.frombuffer(blob, dtype="<f4").copy()
                stats["reused"] += 1
            else:
                if model is None:
                    loading = time.perf_counter()
                    model = EncoderModel(device, kind=kind).eval()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    stats["model_loading_seconds"] = time.perf_counter() - loading
                    if width is not None and width != model.encoder.config.hidden_size:
                        raise ValueError("Cached feature width differs from the pinned encoder")
                    width = model.encoder.config.hidden_size
                    db.execute("INSERT OR IGNORE INTO recipe VALUES (?,?)", (fingerprint, width))
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                computing = time.perf_counter()
                with torch.no_grad():
                    vector = model.embed(row["body"]).cpu().numpy().astype("<f4")
                seconds = time.perf_counter() - computing
                vram = torch.cuda.max_memory_allocated() if device == "cuda" else 0
                blob = vector.tobytes()
                if vector.ndim != 1 or vector.size != width or not np.isfinite(vector).all():
                    raise ValueError("Invalid encoder feature")
                db.execute("INSERT INTO features VALUES (?,?,?,?,?)",
                           (digest, blob, hashlib.sha256(blob).hexdigest(), seconds, vram))
                stats["computed"] += 1
                if stats["computed"] % 250 == 0:
                    db.commit()
                    print(f"{kind} {split}: {len(vectors) + 1}/{len(records)}; {stats['computed']} new features", flush=True)
            if vector.size != width or not np.isfinite(vector).all() or seconds < 0 or vram < 0:
                raise ValueError("Invalid cached encoder feature")
            vectors.append(vector)
            stats["recorded_compute_seconds"] += seconds
            stats["peak_vram_bytes"] = max(stats["peak_vram_bytes"], vram)
    stats["preparation_seconds"] = time.perf_counter() - started
    if details is not None:
        details.update(stats)
    return np.array(vectors)


def fit(args):
    """Fit features on train only; choose the classifier and decision offsets on validation."""
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    train, validation = [load_split(args.splits, name) for name in ("train", "validation")]
    texts, y = [r["body"] for r in train], [annotation(r)["label"] for r in train]
    vtexts = [r["body"] for r in validation]
    prototype = make(args.model)
    if args.model == "majority":
        prefix, x, vx, candidates = [], texts, vtexts, [({}, prototype)]
    else:
        if args.model in ("finbert", "bge"):
            texts = embeddings(args.model, train, "train", args.device)
            vtexts = embeddings(args.model, validation, "validation", args.device)
        # Fit vocabulary/scaling once on train; validation never updates these features.
        transform = prototype[:-1]
        x, vx = transform.fit_transform(texts), transform.transform(vtexts)
        prefix = transform.steps
        candidates = [
            ({"C": c, "class_weight": weight}, clone(prototype[-1]).set_params(C=c, class_weight=weight))
            for weight, c in itertools.product((None, "balanced"), args.regularization)
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "source"
    source.mkdir(exist_ok=True)
    files = ("baselines.py", "tune.py", "encoders.py", "benchmark.py", "schemas.py", "label.py")
    for filename in files:
        (source / filename).write_bytes((HERE / filename).read_bytes())
    model_id, revision = None, None
    if args.model in ("finbert", "bge"):
        from encoders import MODELS

        model_id, revision = MODELS[args.model]
    packages = ("scikit-learn", "numpy") + (("torch", "transformers") if model_id else ())
    config = {
        "kind": args.model,
        "family": "linear",
        "model": model_id,
        "revision": revision,
        "embedding_device": args.device if model_id else None,
        "labels": list(LABELS),
        "split_manifest": json.loads((args.splits / "manifest.json").read_text(encoding="utf-8")),
        "precision": "bf16 encoder CUDA / fp32 classifier" if model_id else "fp32",
        "source_sha256": {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in files},
        "versions": {n: version(n) for n in packages},
    }
    best, trials = -1, []
    started = time.perf_counter()
    for params, classifier in candidates:
        with threadpool_limits(limits=4):
            classifier.fit(x, y)
        order = [list(classifier.classes_).index(label) for label in LABELS]
        rows = predictions(validation, classifier.predict_proba(vx)[:, order])
        scores = Benchmark.scores(rows)
        tuned, offsets = best_offsets(rows)
        trial = {
            "parameters": params,
            "macro_f1": scores["macro_f1"],
            "agreement": scores["agreement"],
            "tuned_macro_f1": tuned,
            "offsets": offsets,
        }
        trials.append(trial)
        print(trial, flush=True)
        if tuned > best:
            best = tuned
            model = Pipeline([*prefix, ("classifier", classifier)]) if prefix else classifier
            joblib.dump(model, args.output / "model.joblib")
            save({**scores, "predictions": rows}, args.output / "validation.json")
            save({**config, "selected": trial}, args.output / "config.json")
        save({"trials": trials, "seconds": time.perf_counter() - started}, args.output / "search.json")
    tune_offsets(args.output, args.splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=FITTED_MODELS,
        default="tfidf",
    )
    parser.add_argument("--splits", type=Path, default=HERE / "data/splits")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--offsets-only", action="store_true")
    parser.add_argument("--blend", type=Path, nargs=2, metavar="RUN", help="Blend two fitted run directories")
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0])
    args = parser.parse_args()
    if args.blend:
        blend(args.blend, args.output, args.splits)
    elif args.offsets_only:
        tune_offsets(args.output, args.splits)
    else:
        fit(args)
