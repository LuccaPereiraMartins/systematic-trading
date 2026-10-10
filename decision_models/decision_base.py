"""The pinned Laya SDK reference on selection/calibration only; no training or test access."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch

from protocol import breakdown, calibrate, discard, fingerprint, prediction_rows, probabilities, speed
from schemas import HERE, LABELS, LAYA_MODEL, LAYA_QUESTIONS, LAYA_REVISION, load_split, save


class BaseLaya:
    def __init__(self, config, device="cuda"):
        import laya
        self.config = config
        self.device = device
        warnings.filterwarnings("ignore", message=r"laya:.*invalid temperatures")
        self.agent = laya.load(config["model"], device=device, revision=config["revision"])

    def answers(self, bodies):
        return [self.agent.predict_long(body, self.config["questions"])["answers"]["triage"] for body in bodies]

    def logits(self, bodies):
        vectors = []
        for answer in self.answers(bodies):
            p = answer["probabilities"]
            vector = [p[label] for label in LABELS] if isinstance(p, dict) else p
            vector = np.asarray(vector, dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all() or (vector < 0).any() or vector.sum() <= 0:
                raise ValueError("SDK did not return a valid three-class distribution")
            vectors.append(np.log(np.maximum(vector / vector.sum(), 1e-12)))
        return np.array(vectors)


def fit(args):
    torch.set_num_threads(4)
    manifest = json.loads((args.splits / "manifest.json").read_text(encoding="utf-8"))
    if "prepared" not in manifest:
        raise ValueError("Use the fresh research splits")
    files = ("decision_base.py", "protocol.py", "schemas.py", "prepare.py")
    config = {"family": "research_sdk", "model": LAYA_MODEL, "revision": LAYA_REVISION,
              "questions": LAYA_QUESTIONS, "labels": list(LABELS), "device": args.device,
              "split_manifest": manifest, "source_sha256": {name: fingerprint(HERE / name) for name in files},
              "versions": {name: version(name) for name in ("laya", "torch", "transformers")},
              "selection": "Unadapted SDK reference, no training", "api_cost_usd": 0.0}
    if args.output.exists() and any(args.output.iterdir()):
        previous = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        if not args.resume or any(previous.get(key) != value for key, value in config.items()):
            raise ValueError("Output exists or resume inputs/settings/source changed")
        if (args.output / "complete.json").exists():
            completed = json.loads((args.output / "complete.json").read_text(encoding="utf-8"))
            for name, filename in (("config", "config.json"), ("decision", "decision.json"), ("artifact", "base.json")):
                if fingerprint(args.output / filename) != completed[f"{name}_sha256"]:
                    raise ValueError("Completed SDK run changed")
            print(f"Already completed {args.output}", flush=True)
            return
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "source"
    snapshot.mkdir(exist_ok=True)
    for name in files:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    save(config, args.output / "config.json")
    started = time.perf_counter()
    runtime = BaseLaya(config, args.device)
    config["load_seconds"] = time.perf_counter() - started
    config["context"] = {key: runtime.agent.cfg[key] for key in ("max_len", "head_max_len")}
    config["pooling"] = "Native SDK predict_long, recorded separately from fixed uniform adapted pooling"
    save({key: config[key] for key in ("model", "revision", "questions", "context", "pooling")}, args.output / "base.json")
    config["artifact_sha256"] = fingerprint(args.output / "base.json")
    save(config, args.output / "config.json")
    selection, calibration = [load_split(args.splits, name) for name in ("selection", "calibration")]
    def score(name, records):
        from schemas import body_hash
        path = args.output / f"{name}-logits.jsonl"
        saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        if len(saved) > len(records) or any(row["body_sha256"] != body_hash(record)
                                          for row, record in zip(saved, records)):
            raise ValueError("Cached SDK predictions differ from frozen records")
        with path.open("a", encoding="utf-8") as stream:
            for record in records[len(saved):]:
                row = {"body_sha256": body_hash(record), "logits": runtime.logits([record["body"]])[0].tolist()}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                saved.append(row)
                if len(saved) % 100 == 0:
                    print(f"SDK {name}: {len(saved)}/{len(records)}", flush=True)
        return np.array([row["logits"] for row in saved])
    logits, clogits = score("selection", selection), score("calibration", calibration)
    decision = calibrate(calibration, clogits, args.output)
    rows = prediction_rows(selection, logits, decision)
    save({"metrics": breakdown(rows), "predictions": rows, "evaluation": "Selection data; no training"},
         args.output / "selection.json")
    def predict(bodies):
        p = probabilities(runtime.logits(bodies), decision["temperature"])
        return p, {key: discard(p, value["threshold"]) for key, value in decision["policies"].items()}
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    measured = speed(predict, selection, torch.cuda.synchronize if args.device == "cuda" else lambda: None)
    measured.update(batch_definition="Serial SDK document calls; native per-document windows",
                    peak_vram_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                    elapsed_seconds=time.perf_counter() - started,
                    cost_note="Local compute; pinned base weights, electricity and hardware are additional")
    save(measured, args.output / "performance.json")
    save({"config_sha256": fingerprint(args.output / "config.json"), "decision_sha256": fingerprint(args.output / "decision.json"),
          "artifact_sha256": fingerprint(args.output / "base.json")}, args.output / "complete.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda":
        from filelock import FileLock
        lock = HERE / "data/research/gpu-job.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(lock):
            fit(args)
    else:
        fit(args)
