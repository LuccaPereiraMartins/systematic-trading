"""Fit research baselines on frozen nested training subsets; never load the test split."""

import argparse
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from baselines import FITTED_MODELS, make
from protocol import breakdown, calibrate, discard, fingerprint, prediction_rows, probabilities, speed, training_subset
from schemas import HERE, LABELS, annotation, load_split, replace_file, save


GRID = (.001, .01, .1, 1, 10, 30, 100)


def classifier_logits(model, inputs):
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        values = np.log(np.maximum(model.predict_proba(inputs), 1e-12))
    else:
        values = model.decision_function(inputs)
        if values.ndim == 1:
            values = np.column_stack((-values, values))
    aligned = np.full((len(values), len(LABELS)), -30.0)
    for index, label in enumerate(classes):
        aligned[:, LABELS.index(label)] = values[:, index]
    return aligned


class LinearModel:
    def __init__(self, directory, device="cuda"):
        self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.model = joblib.load(directory / "model.joblib")
        if fingerprint(directory / "model.joblib") != self.config["artifact_sha256"]:
            raise ValueError("Fitted artifact changed")
        self.encoder = None
        if self.config["kind"] in ("finbert", "bge"):
            from encoders import EncoderModel
            self.encoder = EncoderModel(device, config=self.config).eval()

    def logits(self, bodies):
        inputs = bodies
        if self.encoder:
            import torch
            with torch.no_grad():
                inputs = np.array([self.encoder.embed(body).cpu().numpy() for body in bodies])
        with threadpool_limits(limits=4):
            return classifier_logits(self.model, inputs)


def fit(args):
    train = training_subset(args.splits, args.samples)
    selection, calibration = [load_split(args.splits, name) for name in ("selection", "calibration")]
    sources = ("baselines.py", "fit_baselines.py", "protocol.py", "encoders.py", "schemas.py", "prepare.py",
               "tune.py", "benchmark.py")
    manifest = json.loads((args.splits / "manifest.json").read_text(encoding="utf-8"))
    config = {"family": "research_linear", "kind": args.model, "classifier": args.classifier,
              "labels": list(LABELS), "seed": 42, "samples": len(train), "split_manifest": manifest,
              "regularization": list(args.regularization), "device": args.device,
              "selection": "Macro F1 on selection groups; ties retain first grid candidate",
              "source_sha256": {name: fingerprint(HERE / name) for name in sources},
              "versions": {name: version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib")},
              "python": platform.python_version(), "api_cost_usd": 0.0,
              "cost_note": "Local compute; hardware, electricity and collection excluded"}
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume:
            raise ValueError("Output exists; use --resume or another directory")
        previous = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError("Resume input/settings/source changed")
        if (args.output / "complete.json").exists():
            completed = json.loads((args.output / "complete.json").read_text(encoding="utf-8"))
            if any(fingerprint(args.output / f"{name}.json") != completed[f"{name}_sha256"]
                   for name in ("config", "decision")):
                raise ValueError("Completed run metadata changed")
            LinearModel(args.output, args.device)
            print(f"Already completed {args.output}", flush=True)
            return
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "source"
    snapshot.mkdir(exist_ok=True)
    for name in sources:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    save(config, args.output / "config.json")
    started = time.perf_counter()
    texts, labels = [r["body"] for r in train], [annotation(r)["label"] for r in train]
    stext, ctext = [r["body"] for r in selection], [r["body"] for r in calibration]
    encoder = args.model in ("finbert", "bge")
    if encoder:
        from encoders import MODELS
        from tune import embeddings
        config["model"], config["revision"] = MODELS[args.model]
        config["versions"].update({name: version(name) for name in ("torch", "transformers")})
        feature_started = time.perf_counter()
        features, preparation = [], {}
        for rows, name in ((train, "train"), (selection, "selection"), (calibration, "calibration")):
            preparation[name] = {}
            features.append(embeddings(args.model, rows, name, args.device, preparation[name]))
        texts, stext, ctext = features
        config["feature_preparation"] = preparation
        config["feature_preparation_seconds"] = time.perf_counter() - feature_started
        config["feature_cost_note"] = "Body/recipe caches share frozen features across subsets; recorded compute is separate from this fit's cache preparation"
    prototype = make(args.model, classifier=args.classifier)
    if args.model == "majority":
        prefix, x, sx, cx, candidates = [], texts, stext, ctext, [({}, prototype)]
    else:
        transform = prototype[:-1]
        with threadpool_limits(limits=4):
            x = transform.fit_transform(texts)
            sx, cx = transform.transform(stext), transform.transform(ctext)
        prefix = transform.steps
        candidates = [({"C": c, "class_weight": weight}, clone(prototype[-1]).set_params(C=c, class_weight=weight))
                      for weight, c in itertools.product((None, "balanced"), args.regularization)]
    trials, best, chosen = [], -1, None
    for parameters, estimator in candidates:
        with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(x, labels)
        rows = prediction_rows(selection, classifier_logits(estimator, sx))
        scores = breakdown(rows)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        trial = {"parameters": parameters, "macro_f1": scores["overall"]["macro_f1"],
                 "converged": converged, "warnings": [str(w.message) for w in caught]}
        trials.append(trial)
        print(trial, flush=True)
        if converged and trial["macro_f1"] > best:
            best, chosen = trial["macro_f1"], estimator
            config["selected"] = trial
        save({"trials": trials, "elapsed_seconds": time.perf_counter() - started}, args.output / "search.json")
    if chosen is None:
        raise ValueError("No grid candidate converged")
    model = Pipeline([*prefix, ("classifier", chosen)]) if prefix else chosen
    temporary = args.output / "model.tmp"
    joblib.dump(model, temporary)
    replace_file(temporary, args.output / "model.joblib")
    config.update(artifact_sha256=fingerprint(args.output / "model.joblib"),
                  fit_seconds=time.perf_counter() - started)
    save(config, args.output / "config.json")
    decision = calibrate(calibration, classifier_logits(chosen, cx), args.output)
    rows = prediction_rows(selection, classifier_logits(chosen, sx), decision)
    save({"metrics": breakdown(rows), "predictions": rows, "evaluation": "Selection fitting data"},
         args.output / "selection.json")
    del model, chosen, prototype, x, sx, cx, texts, stext, ctext
    runtime = LinearModel(args.output, args.device)
    def synchronize():
        pass
    if encoder and args.device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()
        synchronize = torch.cuda.synchronize
    def predict(bodies):
        p = probabilities(runtime.logits(bodies), decision["temperature"])
        return p, {key: discard(p, value["threshold"]) for key, value in decision["policies"].items()}
    measured = speed(predict, selection, synchronize)
    measured["artifact_bytes"] = (args.output / "model.joblib").stat().st_size
    measured["encoder_weight_size_note"] = "Pinned pretrained encoder weights are additional" if encoder else None
    measured["batch_definition"] = "Serial documents; batched windows within each document" if encoder else "Native sklearn document batch"
    measured["peak_vram_bytes"] = torch.cuda.max_memory_allocated() if encoder and args.device == "cuda" else 0
    save(measured, args.output / "performance.json")
    save({"config_sha256": fingerprint(args.output / "config.json"),
          "decision_sha256": fingerprint(args.output / "decision.json")}, args.output / "complete.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=FITTED_MODELS, required=True)
    parser.add_argument("--classifier", choices=("logistic", "svm"), default="logistic")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--regularization", type=float, nargs="+", default=GRID)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(c <= 0 for c in args.regularization):
        parser.error("Regularization must be positive")
    if args.model in ("majority", "finbert", "bge") and args.classifier == "svm":
        parser.error("SVM is a TF-IDF comparison only")
    if args.model in ("finbert", "bge") and args.device == "cuda":
        from filelock import FileLock
        lock = HERE / "data/research/gpu-job.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(lock):
            fit(args)
    else:
        fit(args)
