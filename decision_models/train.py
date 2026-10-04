"""Supervised Laya head + document attention; train/validation only, with resumable checkpoints."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import laya
import torch
from laya.common import build_sequence
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from benchmark import Benchmark
from label import annotation, save as save_json
from schemas import load_split


HERE = Path(__file__).resolve().parent
SEED = 42
LEARNING_RATE = 1e-4
DOCUMENT_BATCH = 8             # Accumulate gradients from eight complete documents.
WINDOW_BATCH = 8               # Windows per forward pass; every window is retained.
CHECKPOINT_STEPS = 25


class DocumentModel(nn.Module):
    def __init__(self, device="cuda", checkpoint=None):
        super().__init__()
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
        self.training_config = saved["config"] if saved else None
        self.selected_epoch = saved["state"]["best_epoch"] if saved else None
        config = self.training_config or {"model": Benchmark.laya_model, "revision": Benchmark.laya_revision}
        warnings.filterwarnings("ignore", message=r"laya:.*invalid temperatures")
        agent = laya.load(config["model"], device=device, revision=config["revision"])
        self.base, self.tokenizer, self.cfg = agent.model, agent.tok, agent.cfg
        self.questions = config.get("questions", Benchmark.questions)
        if tuple(self.questions["triage"]["criteria"]) != Benchmark.labels:
            raise ValueError("Question options must follow the benchmark label order")
        self.window_batch = config.get("window_batch", WINDOW_BATCH)
        if self.training_config:
            if config["labels"] != list(Benchmark.labels):
                raise ValueError("Checkpoint label order differs from benchmark")
            self.cfg.update(max_len=config["max_len"], head_max_len=config["head_max_len"])
        for name, parameter in self.base.named_parameters():
            parameter.requires_grad_(not name.startswith(("encoder.", "act_head.")))
        self.base.head_checkpointing = True
        self.pool = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1, bias=False))
        nn.init.zeros_(self.pool[-1].weight)  # Begin with uniform document pooling.
        self.to(self.device)
        if saved:
            expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
            if set(saved["weights"]) != expected:
                raise ValueError("Checkpoint does not match the trainable head/pooling parameters")
            self.load_state_dict(saved["weights"], strict=False)

    def train(self, mode=True):
        super().train(mode)
        self.base.encoder.eval()  # Keep frozen encoder features deterministic.
        self.base.act_head.eval()
        return self

    def windows(self, body):
        question = self.questions["triage"]
        q = {"t": "choice", "ins": question["instructions"], "crit": question["criteria"]}
        tokens = self.tokenizer.encode(body.replace(self.tokenizer.mask_token, " "),
                                       add_special_tokens=False, truncation=False)
        budget = self.cfg["max_len"] - self.cfg["head_max_len"] - 8
        items = []
        for start in range(0, max(1, len(tokens)), budget // 2):
            chunk = tokens[start:start + budget]
            ids, markers = build_sequence(self.tokenizer, "", q, self.cfg["max_len"],
                                          self.cfg["head_max_len"], state_ids=chunk)
            if len(markers) != 3 or ids[-len(chunk)-1:-1] != chunk:
                raise ValueError("Window formatting dropped input tokens or label markers")
            items.append((ids, markers))
            if start + budget >= len(tokens):
                break
        return items

    def forward(self, items):
        scores = []
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for start in range(0, len(items), self.window_batch):
                chunk = items[start:start + self.window_batch]
                sequences = [torch.tensor(ids, device=self.device) for ids, _ in chunk]
                ids = pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                lengths = torch.tensor([len(seq) for seq in sequences], device=self.device)
                mask = torch.arange(ids.shape[1], device=self.device)[None, :] < lengths[:, None]
                markers = torch.tensor([positions for _, positions in chunk], device=self.device)
                logits, _ = self.base(ids, mask, markers, torch.ones_like(markers, dtype=torch.bool),
                                      torch.zeros(len(chunk), dtype=torch.long, device=self.device),
                                      detach_encoder=True)
                scores.append(logits)
            scores = torch.cat(scores)
            # Gate on relative scores so arbitrary common logit offsets cannot affect attention.
            attention = self.pool(scores.log_softmax(-1)).float().softmax(0)
            return (attention * scores).sum(0), attention[:, 0]


def evaluate(model, items):
    model.eval()
    rows, loss = [], 0.0
    with torch.no_grad():
        for windows, label in items:
            logits, _ = model(windows)
            loss += nn.functional.cross_entropy(logits[None], torch.tensor([label], device=model.device)).item()
            rows.append({"reference": Benchmark.labels[label],
                         "prediction": Benchmark.labels[logits.argmax().item()]})
    return {"loss": loss / len(items), **Benchmark.scores(rows)}


def save_checkpoint(path, model, optimizer, state, config):
    weights = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
               if parameter.requires_grad}
    temporary = path.with_suffix(".tmp")
    torch.save({"weights": weights, "optimizer": optimizer.state_dict(), "state": state,
                "config": config, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if model.device.type == "cuda" else []}, temporary)
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(1)


def run(args):
    torch.manual_seed(SEED)
    model = DocumentModel(args.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=LEARNING_RATE, weight_decay=0.01)
    manifest = json.loads((args.splits / "manifest.json").read_text(encoding="utf-8"))
    config = {
        "model": Benchmark.laya_model, "revision": Benchmark.laya_revision,
        "seed": SEED, "epochs": args.epochs, "limit": args.limit,
        "learning_rate": LEARNING_RATE, "document_batch": DOCUMENT_BATCH, "window_batch": WINDOW_BATCH,
        "max_len": model.cfg["max_len"], "head_max_len": model.cfg["head_max_len"],
        "pooling": "attention over per-window log-probabilities; weighted sum of raw logits",
        "objective": "unweighted document-level cross-entropy", "precision": "bf16 CUDA / fp32 CPU",
        "split_manifest": manifest, "labels": list(Benchmark.labels), "questions": Benchmark.questions,
        "versions": {name: version(name) for name in ("laya", "torch", "transformers")},
        "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ("train.py", "benchmark.py", "schemas.py", "label.py")},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    last = args.output / "last.pt"
    state = {"epoch": 0, "offset": 0, "steps": 0, "loss_sum": 0.0,
             "best_f1": None, "best_epoch": None, "history": [], "seconds": 0.0}
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=True)
        for key in config:
            if key not in ("epochs", "git_revision", "source_sha256", "gpu") and config[key] != saved["config"][key]:
                raise ValueError(f"Resume configuration changed: {key}")
        model.load_state_dict(saved["weights"], strict=False)
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        if state["epoch"] >= args.epochs:
            print(f"Already completed {state['epoch']} epoch(s); selected epoch {state['best_epoch']}", flush=True)
            return
        state.setdefault("resumes", []).append({key: config[key] for key in ("git_revision", "source_sha256", "gpu")})
        torch.set_rng_state(saved["rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    elif last.exists():
        raise ValueError("Run already exists; use --resume or a different --output")
    source_id = hashlib.sha256(json.dumps(config["source_sha256"], sort_keys=True).encode()).hexdigest()[:12]
    snapshot = args.output / "source" / source_id
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in config["source_sha256"]:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    save_json(config, snapshot / "config.json")
    save_json(config, args.output / "config.json")
    started, previous_seconds = time.perf_counter(), state["seconds"]
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def prepare(name):
        records = load_split(args.splits, name)
        random.Random(SEED).shuffle(records)
        if args.limit:
            records = records[:args.limit]
        items = [(model.windows(row["body"]), Benchmark.labels.index(annotation(row)["label"]))
                 for row in records]
        print(f"Prepared {name}: {len(items)} documents, {sum(len(w) for w, _ in items)} windows", flush=True)
        return items

    train, validation = prepare("train"), prepare("validation")

    def checkpoint(path=last):
        state["seconds"] = previous_seconds + time.perf_counter() - started
        state["peak_vram_bytes"] = torch.cuda.max_memory_allocated() if args.device == "cuda" else 0
        save_checkpoint(path, model, optimizer, state, config)
        save_json(state, args.output / "history.json")

    if not state["history"]:
        baseline = evaluate(model, validation)
        state["history"].append({"epoch": 0, "validation": baseline})
        state["best_f1"], state["best_epoch"] = baseline["macro_f1"], 0
        checkpoint(args.output / "best.pt")
        checkpoint()
        print(f"Initial validation: agreement={baseline['agreement']:.3f}, macro F1={baseline['macro_f1']:.3f}", flush=True)
    while state["epoch"] < args.epochs:
        model.train()
        order = list(range(len(train)))
        random.Random(SEED + state["epoch"]).shuffle(order)
        for offset in range(state["offset"], len(order), DOCUMENT_BATCH):
            group = order[offset:offset + DOCUMENT_BATCH]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                windows, label = train[index]
                logits, _ = model(windows)
                loss = nn.functional.cross_entropy(logits[None], torch.tensor([label], device=model.device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite document loss; last checkpoint remains available")
                (loss / len(group)).backward()
                state["loss_sum"] += loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            state["offset"] = offset + len(group)
            state["steps"] += 1
            if state["steps"] % CHECKPOINT_STEPS == 0:
                checkpoint()
                print(f"Epoch {state['epoch']+1}: {state['offset']}/{len(train)}; loss={state['loss_sum']/state['offset']:.3f}", flush=True)
        scores = evaluate(model, validation)
        state["epoch"] += 1
        state["history"].append({"epoch": state["epoch"], "train_loss": state["loss_sum"] / len(train),
                                 "validation": scores})
        state["offset"], state["loss_sum"] = 0, 0.0
        if scores["macro_f1"] > state["best_f1"]:
            state["best_f1"], state["best_epoch"] = scores["macro_f1"], state["epoch"]
            checkpoint(args.output / "best.pt")
        checkpoint()
        print(f"Epoch {state['epoch']} validation: agreement={scores['agreement']:.3f}, macro F1={scores['macro_f1']:.3f}", flush=True)
    print(f"Selected epoch {state['best_epoch']}; saved {args.output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=HERE / "data/splits")
    parser.add_argument("--output", type=Path, default=HERE / "training_runs/head")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Small train/validation smoke run; never reads test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("epochs and limit must be positive")
    run(args)
