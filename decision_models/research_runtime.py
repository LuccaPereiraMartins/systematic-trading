"""Verified research artifacts and one whole-document inference interface."""

from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np

from protocol import breakdown, calibrate, fingerprint, prediction_rows, speed
from schemas import HERE, LABELS, load_split, save


def verify(directory, manifest, sources=True):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if config["split_manifest"] != manifest or config["labels"] != list(LABELS):
        raise ValueError(f"Different data or label order: {directory}")
    for key, name in (("config", "config.json"), ("decision", "decision.json")):
        if fingerprint(directory / name) != complete[f"{key}_sha256"]:
            raise ValueError(f"Completed metadata changed: {directory / name}")
    artifact = "model.joblib" if config["family"] == "research_linear" else (
        "base.json" if config["family"] in ("research_sdk", "research_prompt") else "best.pt")
    expected = complete.get("artifact_sha256", config.get("artifact_sha256"))
    if not expected or fingerprint(directory / artifact) != expected:
        raise ValueError(f"Completed artifact changed: {directory}")
    if sources:
        for name, digest in config["source_sha256"].items():
            if fingerprint(HERE / name) != digest:
                raise ValueError(f"Inference/training source changed since fitting: {name}")
        for name, expected in config["versions"].items():
            if version(name) != expected:
                raise ValueError(f"Runtime version changed since fitting: {name}")
    return config, artifact


class Runtime:
    def __init__(self, directory, device="cuda"):
        import torch
        torch.set_num_threads(4)
        self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.family = self.config["family"]
        self.device = device
        if self.family == "research_linear":
            from fit_baselines import LinearModel
            self.model = LinearModel(directory, device)
            if self.config["kind"] not in ("finbert", "bge"):
                self.device = "cpu"
        elif self.family == "research_sdk":
            from decision_base import BaseLaya
            self.model = BaseLaya(self.config, device)
        elif self.family == "research_prompt":
            from causal import CausalModel
            self.model = CausalModel(device, config=self.config).eval()
        else:
            from encoders import load_model
            self.model = load_model(device, checkpoint=directory / "best.pt").eval()

    def score(self, body):
        import torch
        with torch.no_grad():
            if self.family in ("research_linear", "research_sdk", "research_prompt"):
                logits = self.model.logits([body])[0]
            else:
                logits = self.model(self.model.windows(body))[0].cpu().numpy()
        result = {"logits": np.asarray(logits, dtype=float).tolist()}
        if hasattr(self.model, "coverage"):
            result.update(coverage=self.model.coverage(body),
                          answer_probability_mass=self.model.answer_probability_mass)
        return result

    def logits(self, bodies):
        return np.array([self.score(body)["logits"] for body in bodies])


def zero_control(directory, splits, output):
    """Calibrate the preserved pre-update weights; no training and no test access."""
    import torch
    manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    config, _ = verify(directory, manifest)
    if (output / "complete.json").exists():
        verify(output, manifest)
        return
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directory / "epoch-zero.pt", output / "best.pt")
    config.update(family="research_zero", control_of=str(directory), samples=0,
                  selection="Matched pre-update model; no training",
                  artifact_sha256=fingerprint(output / "best.pt"), api_cost_usd=0.0)
    save(config, output / "config.json")
    model = Runtime(output)
    selection, calibration = [load_split(splits, name) for name in ("selection", "calibration")]
    decision = calibrate(calibration, model.logits([r["body"] for r in calibration]), output)
    rows = prediction_rows(selection, model.logits([r["body"] for r in selection]), decision)
    save({"metrics": breakdown(rows), "predictions": rows, "evaluation": "Matched epoch zero; selection groups"},
         output / "selection.json")
    torch.cuda.reset_peak_memory_stats()
    measured = speed(model.logits, selection, torch.cuda.synchronize)
    measured.update(adapter_bytes=(output / "best.pt").stat().st_size,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(), training_peak_vram_bytes=0,
                    batch_definition="Serial documents; original model window batch",
                    cost_note="No training; local inference and calibration")
    save(measured, output / "performance.json")
    save({"config_sha256": fingerprint(output / "config.json"),
          "decision_sha256": fingerprint(output / "decision.json"),
          "artifact_sha256": fingerprint(output / "best.pt")}, output / "complete.json")


if __name__ == "__main__":
    import argparse
    from filelock import FileLock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with FileLock(HERE / "data/research/gpu-job.lock"):
        zero_control(args.zero, args.splits, args.output)
