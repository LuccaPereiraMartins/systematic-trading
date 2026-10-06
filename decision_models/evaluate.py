"""Fit baselines on train/validation, freeze choices, then evaluate the common test."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

from baselines import FITTED_MODELS
from schemas import HERE, SPLITS, body_hash, load_split, save


def evaluate(models, splits=SPLITS, output=HERE / "training_runs", device="cuda", refit=False):
    manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    hashes = [{body_hash(r) for r in load_split(splits, name)} for name in ("train", "validation", "test")]
    if any(hashes[a] & hashes[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Dataset splits overlap")
    directories = []
    for name in models:
        directory = output / name
        config = directory / "config.json"
        if refit or not (directory / "decision.json").exists():
            print(f"Fitting {name} on train/validation", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "tune.py"),
                    "--model",
                    name,
                    "--splits",
                    str(splits),
                    "--output",
                    str(directory),
                    "--device",
                    device,
                ],
                check=True,
            )
        saved = json.loads(config.read_text(encoding="utf-8"))
        if saved["split_manifest"] != manifest:
            raise ValueError(f"Different splits: {directory}")
        for filename, digest in saved["source_sha256"].items():
            if hashlib.sha256((HERE / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed since fitting {directory}; use --refit")
        directories.append(directory)
    # Save every chosen artifact BEFORE any test inference; no tuning follows this point.
    selected = {
        d.name: {
            file: hashlib.sha256((d / file).read_bytes()).hexdigest()
            for file in ("model.joblib", "config.json", "decision.json")
        }
        for d in directories
    }
    save(
        {"created_utc": datetime.now(timezone.utc).isoformat(), "split_manifest": manifest, "models": selected},
        output / "selection.json",
    )
    for directory in directories:
        print(f"Evaluating {directory.name} on the frozen test", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(HERE / "benchmark.py"),
                "--saved",
                str(directory),
                "--splits",
                str(splits),
                "--device",
                device,
            ],
            check=True,
        )
    print("All baselines completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=FITTED_MODELS, default=FITTED_MODELS)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--output", type=Path, default=HERE / "training_runs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--refit", action="store_true", help="Replace fitted runs after code/data changes")
    args = parser.parse_args()
    evaluate(args.models, args.splits, args.output, args.device, args.refit)
