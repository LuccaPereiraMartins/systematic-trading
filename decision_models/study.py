"""Serial validation-only experiment matrix, immutable selection freeze, then test/report."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from protocol import fingerprint
from research_runtime import verify
from schemas import HERE, load_split, save


def nearest(sizes, target):
    return max((n for n in sizes if n <= target), default=min(sizes))


class Study:
    def __init__(self, splits, output):
        self.splits, self.output = splits.resolve(), output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
        if "prepared" not in self.manifest:
            raise ValueError("Use the fresh research benchmark")
        subsets = json.loads((splits / "subsets.json").read_text(encoding="utf-8"))
        self.sizes = sorted(map(int, subsets["subsets"]))
        plan = {"manifest_sha256": fingerprint(splits / "manifest.json"),
                "subsets_sha256": fingerprint(splits / "subsets.json"), "seed": 42,
                "sizes": self.sizes, "screen_samples": nearest(self.sizes, 1000),
                "context_samples": nearest(self.sizes, 4000),
                "epochs": 3, "lora_rank": 8, "context_grid": [512, 1024, 2048, 4096],
                "policy_targets": [.01, .05], "bootstrap_replicates": 2000,
                "source_sha256": {name: fingerprint(HERE / name) for name in
                                  ("study.py", "research_runtime.py", "evaluate_study.py")},
                "sample_efficiency_note": "Recipes screened at 1000/4000 or nearest attainable sizes; tuning labels are additional",
                "api_execution": "None; labels use the separate shared Luna ledger"}
        path = output / "plan.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Study data, matrix or source changed; preserve this study and use a new output")
        if not path.exists():
            save(plan, path)
        state = output / "state.json"
        self.state = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {"jobs": {}, "attempts": []}
        self.comparisons = {}

    def job(self, name, script, arguments, role="screen", control=None, fallback_of=None):
        directory = self.output / "runs" / name
        command = [sys.executable, str(HERE / script), "--splits", str(self.splits), "--output", str(directory),
                   *map(str, arguments)]
        record = {"script": script, "arguments": list(map(str, arguments)), "directory": str(directory.relative_to(self.output)),
                  "role": role, "control_of": control, "fallback_of": fallback_of}
        previous = self.state["jobs"].get(name)
        if previous and any(previous[key] != record[key] for key in ("script", "arguments", "directory")):
            raise ValueError(f"Job changed: {name}")
        if (directory / "complete.json").exists():
            verify(directory, self.manifest)
            record["status"] = "completed"
        elif previous and previous["status"] == "oom":
            record["status"] = "oom"
        else:
            directory.mkdir(parents=True, exist_ok=True)
            resumable = directory / ("last.pt" if script == "train.py" else "config.json")
            if resumable.exists() and script != "research_runtime.py":
                command.append("--resume")
            record["status"] = "running"
            self.state["jobs"][name] = record
            save(self.state, self.output / "state.json")
            print(f"START {name}", flush=True)
            logs = self.output / "logs"
            logs.mkdir(exist_ok=True)
            log = logs / f"{name}.log"
            with log.open("a", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            attempt = {"job": name, "exit_code": code, "utc": datetime.now(timezone.utc).isoformat()}
            self.state["attempts"].append(attempt)
            oom = "out of memory" in log.read_text(encoding="utf-8").lower()
            record["status"] = "completed" if code == 0 else "oom" if oom else "failed"
            self.state["jobs"][name] = record
            save(self.state, self.output / "state.json")
            if code and not oom:
                raise RuntimeError(f"Job failed; inspect {log}")
            if code == 0:
                verify(directory, self.manifest)
        self.state["jobs"][name] = record
        save(self.state, self.output / "state.json")
        if record["status"] == "oom":
            if "--window-batch" in arguments and str(arguments[arguments.index("--window-batch") + 1]) != "1":
                smaller = list(arguments)
                smaller[smaller.index("--window-batch") + 1] = 1
                return self.job(name + "-wb1", script, smaller, role, control, fallback_of=name)
            return None
        if role != "screen":
            self.comparisons[name] = {"directory": record["directory"], "role": role, "control_of": control}
        return name

    def neural(self, kind, adaptation, n, context, objective="ce", weight=0, lr=1e-4, role="screen"):
        name = f"{kind}-{adaptation}-{objective}-w{weight}-lr{lr:g}-n{n}-c{context}"
        arguments = ["--model", kind, "--adaptation", adaptation, "--samples", n, "--context", context,
                     "--objective", objective, "--class-weight-power", weight, "--learning-rate", lr,
                     "--epochs", 3, "--window-batch", 1 if kind.startswith("qwen") else 8]
        if adaptation == "lora":
            arguments += ["--lora-rank", 8]
        return self.job(name, "train.py", arguments, role)

    def choose(self, candidates):
        candidates = [name for name in candidates if name]
        if not candidates:
            raise RuntimeError("No feasible recipe; preserve failure logs and resolve before freezing")
        def score(name):
            result = self.output / self.state["jobs"][name]["directory"] / "selection.json"
            return json.loads(result.read_text(encoding="utf-8"))["metrics"]["overall"]["macro_f1"]
        return max(candidates, key=score)  # Stable input order breaks ties.

    def recipe(self, name):
        directory = self.output / self.state["jobs"][name]["directory"]
        return json.loads((directory / "config.json").read_text(encoding="utf-8"))

    def fit(self):
        if (self.output / "freeze.json").exists() or (self.output / "test-started.json").exists():
            raise ValueError("Model selection is frozen; resume test/report without fitting")
        for name in ("train", "selection", "calibration"):
            load_split(self.splits, name)  # Verify inputs; deliberately never open test.
        small, context_n = nearest(self.sizes, 1000), nearest(self.sizes, 4000)
        for n in self.sizes:
            for kind in ("majority", "tfidf", "char", "combined", "finbert", "bge"):
                for classifier in (("logistic", "svm") if kind in ("tfidf", "char", "combined") else ("logistic",)):
                    self.job(f"{kind}-{classifier}-n{n}", "fit_baselines.py",
                             ["--model", kind, "--classifier", classifier, "--samples", n,
                              "--device", "cuda" if kind in ("finbert", "bge") else "cpu"], "curve")
        self.job("laya-sdk", "decision_base.py", ["--model", "laya"], "reference")
        for kind in ("qwen17", "qwen4"):
            self.job(kind + "-prompt", "decision_base.py", ["--model", kind], "reference")
        choices = {}
        for adaptation in ("head", "lora"):
            candidates = [self.neural("laya", adaptation, small, 1024, loss, weight,
                                      3e-5 if adaptation == "head" else 1e-4)
                          for loss in ("ce", "brier") for weight in (0, 1)]
            selected = self.recipe(self.choose(candidates))
            contexts = [self.neural("laya", adaptation, context_n, context, selected["loss_kind"],
                                    selected["class_weight_power"], selected["learning_rate"], "context")
                        for context in (512, 1024, 2048, 4096)]
            winner = self.choose(contexts)
            recipe = self.recipe(winner)
            choices[f"laya-{adaptation}"] = {"screen": selected, "context_winner": winner}
            for n in self.sizes:
                self.neural("laya", adaptation, n, recipe["max_len"], recipe["loss_kind"],
                            recipe["class_weight_power"], recipe["learning_rate"], "curve")
        for kind in ("finbert", "bge", "modernbert", "qwen17", "qwen4"):
            for adaptation in (("lora",) if kind.startswith("qwen") else ("head", "lora")):
                contexts = (512, 4096) if kind == "modernbert" else (4096,) if kind.startswith("qwen") else (512,)
                rates = (5e-5, 1e-4) if kind.startswith("qwen") else (3e-5, 1e-4)
                candidates = [self.neural(kind, adaptation, small, context, lr=rate)
                              for context in contexts for rate in rates]
                winner = self.choose(candidates)
                recipe = self.recipe(winner)
                choices[f"{kind}-{adaptation}"] = {"screen_winner": winner}
                sizes = [context_n] if kind in ("modernbert", "qwen4") else self.sizes
                for n in sizes:
                    self.neural(kind, adaptation, n, recipe["max_len"], lr=recipe["learning_rate"], role="curve")
        # Preserve matched controls for every final neural context/recipe, once per identical setup.
        controls = {}
        for name, comparison in list(self.comparisons.items()):
            config = self.recipe(name)
            if config["family"] != "research_neural":
                continue
            key = tuple(config[k] for k in ("kind", "adaptation", "max_len", "window_batch"))
            if key not in controls:
                directory = self.output / comparison["directory"]
                controls[key] = self.job(name + "-zero", "research_runtime.py", ["--zero", str(directory)], "control", name)
            comparison["matched_zero"] = controls[key]
        save({"choices": choices, "comparisons": self.comparisons,
              "criterion": "Raw selection macro F1; no test access"}, self.output / "selection.json")

    def freeze(self):
        if (self.output / "freeze.json").exists():
            return
        if (self.output / "test-started.json").exists():
            raise ValueError("Test was opened before a valid freeze")
        selected = json.loads((self.output / "selection.json").read_text(encoding="utf-8"))
        if any(job["status"] in ("running", "failed") for job in self.state["jobs"].values()):
            raise ValueError("Unfinished or failed core jobs remain")
        entries = {}
        for name, comparison in selected["comparisons"].items():
            directory = self.output / comparison["directory"]
            config, artifact = verify(directory, self.manifest)
            files = ("config.json", "complete.json", "decision.json", "selection.json", "performance.json", artifact)
            entries[name] = {**comparison, "sha256": {f: fingerprint(directory / f) for f in files},
                             "selection_macro_f1": json.loads((directory / "selection.json").read_text())["metrics"]["overall"]["macro_f1"],
                             "samples": config.get("samples", 0), "kind": config.get("kind", "laya"),
                             "family": config["family"]}
        maximum = max(self.sizes)
        linear = [n for n, e in entries.items() if e["family"] == "research_linear" and e["samples"] == maximum]
        decision = [n for n, e in entries.items() if e["family"] == "research_neural" and
                    e["kind"] == "laya" and e["samples"] == maximum]
        def choose(names):
            return max(names, key=lambda n: entries[n]["selection_macro_f1"])
        freeze = {"plan_sha256": fingerprint(self.output / "plan.json"),
                  "selection_sha256": fingerprint(self.output / "selection.json"),
                  "state_sha256": fingerprint(self.output / "state.json"),
                  "manifest": self.manifest, "entries": entries,
                  "primary_linear": choose(linear), "primary_decision": choose(decision),
                  "criterion": "All selections frozen before test inference", "utc": datetime.now(timezone.utc).isoformat()}
        save(freeze, self.output / "freeze.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit", "freeze", "test", "report", "all"), default="all")
    parser.add_argument("--human", type=Path, help="Optional resolved human-review overlay; report only")
    args = parser.parse_args()
    study = Study(args.splits, args.output)
    if args.stage == "fit" and (args.output / "freeze.json").exists():
        parser.error("The study is frozen; use test/report to resume")
    if args.stage in ("fit", "all") and not (args.output / "freeze.json").exists():
        study.fit()
    if args.stage in ("freeze", "all"):
        study.freeze()
    if args.stage in ("test", "all"):
        from evaluate_study import evaluate
        evaluate(args.splits, args.output)
    if args.stage in ("report", "all"):
        from report import build
        build(args.splits, args.output, args.human)
