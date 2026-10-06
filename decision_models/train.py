"""Train document attention heads or encoder adapters on train/validation; resume checkpoints."""

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
from schemas import annotation, load_split, replace_file, save as save_json


HERE = Path(__file__).resolve().parent
SEED = 42
LEARNING_RATE = 1e-4
DOCUMENT_BATCH = 8  # Accumulate gradients from eight complete documents.
WINDOW_BATCH = 8  # Windows per forward pass; every window is retained.
CHECKPOINT_STEPS = 25


class DocumentModel(nn.Module):
    def __init__(self, device="cuda", checkpoint=None, lora_rank=0):
        super().__init__()
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
        self.training_config = saved["config"] if saved else None
        self.selected_epoch = saved["state"]["epoch"] if saved else None
        config = self.training_config or {"model": Benchmark.laya_model, "revision": Benchmark.laya_revision}
        self.model_id, self.revision = config["model"], config["revision"]
        self.kind = "laya"
        self.lora_rank = config.get("lora_rank", lora_rank)
        warnings.filterwarnings("ignore", message=r"laya:.*invalid temperatures")
        agent = laya.load(config["model"], device=device, revision=config["revision"], fast=False)
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
        if self.lora_rank:
            from peft import LoraConfig, get_peft_model

            self.base.encoder = get_peft_model(
                self.base.encoder,
                LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=2 * self.lora_rank,
                    lora_dropout=0.05,
                    target_modules=["Wqkv"],
                    bias="none",
                ),
            )
            self.base.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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
        if not self.lora_rank:
            self.base.encoder.eval()  # Keep frozen encoder features deterministic.
        self.base.act_head.eval()
        return self

    def windows(self, body):
        question = self.questions["triage"]
        q = {"t": "choice", "ins": question["instructions"], "crit": question["criteria"]}
        tokens = self.tokenizer.encode(
            body.replace(self.tokenizer.mask_token, " "), add_special_tokens=False, truncation=False, verbose=False
        )
        budget = self.cfg["max_len"] - self.cfg["head_max_len"] - 8
        items = []
        for start in range(0, max(1, len(tokens)), budget // 2):
            chunk = tokens[start : start + budget]
            ids, markers = build_sequence(
                self.tokenizer, "", q, self.cfg["max_len"], self.cfg["head_max_len"], state_ids=chunk
            )
            if len(markers) != 3 or ids[-len(chunk) - 1 : -1] != chunk:
                raise ValueError("Window formatting dropped input tokens or label markers")
            items.append((ids, markers))
            if start + budget >= len(tokens):
                break
        return items

    def forward(self, items):
        scores = []
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for start in range(0, len(items), self.window_batch):
                chunk = items[start : start + self.window_batch]
                sequences = [torch.tensor(ids, device=self.device) for ids, _ in chunk]
                ids = pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                lengths = torch.tensor([len(seq) for seq in sequences], device=self.device)
                mask = torch.arange(ids.shape[1], device=self.device)[None, :] < lengths[:, None]
                markers = torch.tensor([positions for _, positions in chunk], device=self.device)
                logits, _ = self.base(
                    ids,
                    mask,
                    markers,
                    torch.ones_like(markers, dtype=torch.bool),
                    torch.zeros(len(chunk), dtype=torch.long, device=self.device),
                    detach_encoder=not self.lora_rank,
                )
                scores.append(logits)
            scores = torch.cat(scores)
            # Gate on relative scores so arbitrary common logit offsets cannot affect attention.
            attention = self.pool(scores.log_softmax(-1)).float().softmax(0)
            return (attention * scores).sum(0), attention[:, 0]


def load_model(device="cuda", checkpoint=None, kind="laya", adaptation="head", lora_rank=0):
    if checkpoint:
        config = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
        kind = config.get("kind", "laya")
    if kind == "laya":
        return DocumentModel(device, checkpoint, lora_rank)
    from encoders import EncoderModel

    return EncoderModel(device, kind, adaptation, lora_rank, checkpoint)


def evaluate(model, items):
    model.eval()
    rows, loss = [], 0.0
    with torch.no_grad():
        for index, (windows, label, body_hash) in enumerate(items, 1):
            logits, _ = model(windows)
            loss += nn.functional.cross_entropy(logits[None], torch.tensor([label], device=model.device)).item()
            rows.append(
                {
                    "reference": Benchmark.labels[label],
                    "body_sha256": body_hash,
                    "prediction": Benchmark.labels[logits.argmax().item()],
                    "probabilities": logits.softmax(-1).tolist(),
                }
            )
            if index % 250 == 0:
                print(f"Validation: {index}/{len(items)}", flush=True)
    return {"loss": loss / len(items), **Benchmark.scores(rows), "predictions": rows}


def save_checkpoint(path, model, optimizer, state, config):
    weights = {
        name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    save_torch(
        {
            "weights": weights,
            "optimizer": optimizer.state_dict(),
            "state": state,
            "config": config,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if model.device.type == "cuda" else [],
        },
        path,
    )


def save_torch(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    replace_file(temporary, path)


def run(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = load_model(args.device, kind=args.model, adaptation=args.adaptation, lora_rank=args.lora_rank)
    encoder_params, head_params = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder_params if "encoder." in name else head_params).append(parameter)
    groups = [
        {"params": head_params, "lr": args.learning_rate},
        {"params": encoder_params, "lr": args.encoder_learning_rate or args.learning_rate},
    ]
    optimizer = torch.optim.AdamW(groups if encoder_params else groups[:1], weight_decay=0.01)
    records = load_split(args.splits, "train")
    counts = torch.tensor(
        [sum(annotation(row)["label"] == label for row in records) for label in Benchmark.labels], dtype=torch.float
    )
    if (counts == 0).any():
        raise ValueError("Training requires examples of all three labels")
    weights = (len(records) / (3 * counts)).pow(args.class_weight_power)
    weights /= (weights * counts / len(records)).sum()
    weights = weights.to(model.device)
    manifest = json.loads((args.splits / "manifest.json").read_text(encoding="utf-8"))
    config = {
        "kind": args.model,
        "adaptation": args.adaptation,
        "model": model.model_id,
        "revision": model.revision,
        "seed": SEED,
        "epochs": args.epochs,
        "limit": args.limit,
        "learning_rate": args.learning_rate,
        "document_batch": DOCUMENT_BATCH,
        "window_batch": WINDOW_BATCH,
        "lora_rank": args.lora_rank,
        "class_weight_power": args.class_weight_power,
        "encoder_learning_rate": args.encoder_learning_rate or args.learning_rate,
        "class_weights": weights.tolist(),
        "cpu_threads": 4,
        "cache_window_features": args.model != "laya" and args.adaptation == "head",
        "checkpoint_selection": args.selection,
        "max_len": model.cfg["max_len"],
        "head_max_len": model.cfg["head_max_len"],
        "pooling": "attention over per-window log-probabilities; weighted sum of raw logits"
        if args.model == "laya"
        else "attention over all raw document window embeddings",
        "objective": "class-weighted document-level cross-entropy",
        "precision": "bf16 CUDA / fp32 CPU",
        "split_manifest": manifest,
        "labels": list(Benchmark.labels),
        "questions": Benchmark.questions if args.model == "laya" else {},
        "versions": {name: version(name) for name in ("laya", "torch", "transformers", "peft")},
        "source_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("train.py", "benchmark.py", "schemas.py", "label.py", "encoders.py", "tune.py")
        },
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    last = args.output / "last.pt"
    state = {
        "epoch": 0,
        "offset": 0,
        "steps": 0,
        "loss_sum": 0.0,
        "best_f1": None,
        "best_epoch": None,
        "history": [],
        "seconds": 0.0,
    }
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=True)
        previous_config = {
            "kind": "laya",
            "adaptation": "head",
            "encoder_learning_rate": saved["config"]["learning_rate"],
            "cache_window_features": False,
            "checkpoint_selection": "raw",
            **saved["config"],
        }
        for key in config:
            if key not in ("epochs", "git_revision", "source_sha256", "gpu") and config[key] != previous_config.get(
                key
            ):
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
            records = records[: args.limit]
        items = [
            (
                model.windows(row["body"]),
                Benchmark.labels.index(annotation(row)["label"]),
                hashlib.sha256(row["body"].encode()).hexdigest(),
            )
            for row in records
        ]
        if config["cache_window_features"]:
            key = {
                "model": model.model_id,
                "revision": model.revision,
                "context": model.cfg,
                "versions": config["versions"],
                "device": args.device,
                "bodies": [row[2] for row in items],
                "source_sha256": config["source_sha256"]["encoders.py"],
            }
            digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
            cache = HERE / "data/cache" / f"{args.model}-{name}-{digest[:12]}.pt"
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists():
                features = torch.load(cache, map_location="cpu", weights_only=True)
            else:
                model.eval()
                features = []
                with torch.no_grad():
                    for i, (windows, _, _) in enumerate(items):
                        features.append(model.features(windows).cpu())
                        if (i + 1) % 250 == 0:
                            print(f"Cached {name} features: {i + 1}/{len(items)}", flush=True)
                save_torch(features, cache)
            if len(features) != len(items):
                raise ValueError("Incomplete window feature cache")
            items = [(vector, label, body_hash) for vector, (_, label, body_hash) in zip(features, items)]
        print(f"Prepared {name}: {len(items)} documents, {sum(len(w) for w, _, _ in items)} windows", flush=True)
        return items

    train, validation = prepare("train"), prepare("validation")

    def checkpoint(path=last):
        state["seconds"] = previous_seconds + time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else 0
        state["peak_vram_bytes"] = max(state.get("peak_vram_bytes", 0), peak)
        save_checkpoint(path, model, optimizer, state, config)
        save_json(state, args.output / "history.json")

    def selection_score(result):
        result["selection_macro_f1"] = result["macro_f1"]
        if args.selection == "tuned":
            from tune import best_offsets

            result["selection_macro_f1"], result["offsets"] = best_offsets(result["predictions"])
        return result["selection_macro_f1"]

    if not state["history"]:
        baseline = evaluate(model, validation)
        selected = selection_score(baseline)
        save_json(baseline, args.output / "validation.json")
        baseline.pop("predictions")
        state["history"].append({"epoch": 0, "validation": baseline})
        state["best_f1"], state["best_epoch"] = selected, 0
        checkpoint(args.output / "best.pt")
        checkpoint()
        print(
            f"Initial validation: agreement={baseline['agreement']:.3f}, macro F1={baseline['macro_f1']:.3f}",
            flush=True,
        )
    while state["epoch"] < args.epochs:
        model.train()
        order = list(range(len(train)))
        random.Random(SEED + state["epoch"]).shuffle(order)
        for offset in range(state["offset"], len(order), DOCUMENT_BATCH):
            group = order[offset : offset + DOCUMENT_BATCH]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                windows, label, _ = train[index]
                logits, _ = model(windows)
                # A one-row weighted MEAN would divide by its own weight and cancel it.
                loss = (
                    nn.functional.cross_entropy(logits[None], torch.tensor([label], device=model.device))
                    * weights[label]
                )
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
                print(
                    f"Epoch {state['epoch'] + 1}: {state['offset']}/{len(train)}; loss={state['loss_sum'] / state['offset']:.3f}",
                    flush=True,
                )
        scores = evaluate(model, validation)
        selected = selection_score(scores)
        save_json(scores, args.output / f"validation-epoch-{state['epoch'] + 1}.json")
        if selected > state["best_f1"]:
            save_json(scores, args.output / "validation.json")
        scores.pop("predictions")
        state["epoch"] += 1
        state["history"].append(
            {"epoch": state["epoch"], "train_loss": state["loss_sum"] / len(train), "validation": scores}
        )
        state["offset"], state["loss_sum"] = 0, 0.0
        if selected > state["best_f1"]:
            state["best_f1"], state["best_epoch"] = selected, state["epoch"]
            checkpoint(args.output / "best.pt")
        checkpoint()
        print(
            f"Epoch {state['epoch']} validation: agreement={scores['agreement']:.3f}, raw F1={scores['macro_f1']:.3f}, selection F1={selected:.3f}",
            flush=True,
        )
    if args.selection == "tuned" and args.limit is None:
        from tune import tune_offsets

        tune_offsets(args.output, args.splits)
    print(f"Selected epoch {state['best_epoch']}; saved {args.output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=HERE / "data/splits")
    parser.add_argument("--output", type=Path, default=HERE / "training_runs/head")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Small train/validation smoke run; never reads test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--selection", choices=("raw", "tuned"), default="tuned", help="Validation macro F1 used to choose checkpoints"
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--encoder-learning-rate", type=float)
    parser.add_argument("--model", choices=("laya", "finbert", "bge"), default="laya")
    parser.add_argument("--adaptation", choices=("head", "lora"), default="head")
    parser.add_argument("--class-weight-power", type=float, choices=(0.0, 0.5, 1.0), default=0.0)
    parser.add_argument("--lora-rank", type=int, default=0, help="Rank of LoRA updates (requires --adaptation lora)")
    args = parser.parse_args()
    if args.epochs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("epochs and limit must be positive")
    if (
        args.learning_rate <= 0
        or (args.encoder_learning_rate is not None and args.encoder_learning_rate <= 0)
        or args.lora_rank < 0
    ):
        parser.error("learning rate must be positive and LoRA rank nonnegative")
    if args.adaptation == "lora" and args.lora_rank < 1:
        parser.error("LoRA adaptation requires a positive --lora-rank")
    if args.adaptation == "head" and args.lora_rank:
        parser.error("Head adaptation does not use --lora-rank; select --adaptation lora")
    run(args)
