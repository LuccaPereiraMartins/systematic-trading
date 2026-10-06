# Financial-document triage

Classify a document as `routine`, `review_worthy` or `unclear` for investment-analyst review. Current references are Luna labels, with human annotations taking precedence. Scores measure agreement with those references, not verified investment usefulness.

## Code

| File | Purpose |
| --- | --- |
| `build_model.py` | Package the balanced word TF-IDF API model from the frozen training split |
| `schemas.py` | Shared records, labels, Laya task settings and verified split loading |
| `collect.py` / `label.py` | EDGAR collection and resumable Luna/Flex labeling |
| `split.py` | Reproduce the frozen train/validation/test partition |
| `baselines.py` | Five raw-text recipes: majority, word/character TF-IDF, FinBERT and BGE |
| `tune.py` | Fit linear models and select hyperparameters, decision offsets or blends on validation |
| `evaluate.py` | Fit the core baselines, freeze selections, then evaluate the common test |
| `benchmark.py` | Common scoring, latency and per-run cost for base Laya, saved models and OpenAI |
| `encoders.py` | Neural model definitions, raw context windows and document pooling |
| `train.py` | One training loop for heads/LoRA, caching, validation selection and resumable checkpoints |
| `laya_infer.py` | Small CPU/GPU inference example |

Run commands from the repository root. Data, checkpoints, predictions and the detailed local `experiments.md` notebook are ignored. Only this guide and rerunnable code are maintained as documentation; Git retains earlier documentation and code versions.

## Dataset and protocol

The committed `data/filings-10k-splits.zip` contains 8,000 train / 1,000 validation / 1,000 test JSONL rows, their manifest and the historical pilot inputs. Extract once:

```bash
python -m zipfile -e decision_models/data/filings-10k-splits.zip decision_models/data/splits
```

The 10,000 unique bodies span October 2025 through September 2026. Split seed 42, stratified by label, with 101 previously evaluated pilot bodies reserved for training. Test supports: 247 routine, 656 review-worthy, 97 unclear. The manifest records file hashes; loaders reject changed splits. Exact body overlap is zero, but issuers and near-duplicate templates can cross splits. This is an exploratory split, not a temporal or issuer-disjoint evaluation.

Each record contains only `date`, `body`, `llm` and `human`. Each annotation has `label` and `uncertainty`; human uncertainty stays its own value, including null or 0.0. Canonical bodies stay raw throughout the active pipelines. There is no Item/SIGNATURES trimming or registrant extraction; callers can supply cleaned text if they choose. Other source types use the same body input.

Collection and labeling are independent; they resume existing files:

```bash
python decision_models/collect.py --number 10000 --start 2025-10-01 --end 2026-09-30 --sample
python decision_models/label.py
python decision_models/split.py
```

These commands use ignored `data/dataset.json`. Reproducing the split requires the exact original source JSON bytes, not a fresh live SEC pull; use the archive for the established benchmark. `split.py` refuses to overwrite a different frozen partition. New collections should use another `--output`; future splits need their own protocol.

Set `SEC_USER_AGENT` and `OPENAI_API_KEY` in the root `.env` for collection/labeling. EDGAR requests use edgartools, async fetching and a shared 10-request/second limit. Exact dates come from filing indexes; main filing bodies exclude exhibits. Luna labeling uses four workers, low reasoning, 1,024 output tokens, Flex and capacity backoff; it skips labeled rows, saves progress and never upgrades the service tier. Other text sources can supply the same record schema.

## Benchmarks

```bash
# Core baselines: train -> validation selection -> freeze -> test
python decision_models/evaluate.py
# CPU-only subset
python decision_models/evaluate.py --models majority tfidf char --device cpu
# Base Laya only (default); OpenAI runs require explicit selection and incur charges
python decision_models/benchmark.py --models laya --device cuda
python decision_models/benchmark.py --models luna sol
```

`evaluate.py` reuses fitted runs only when their sources and manifest match; use `--refit` intentionally after changes. It does not rerun neural training. Saved-model evaluation recomputes predictions and latency. All fitted approaches use the same fixed validation selection.

For another fitted variant:

```bash
python decision_models/tune.py --model char --regularization 0.001 0.01 0.1 1 10 30 100 --output decision_models/training_runs/char-new
python decision_models/benchmark.py --saved decision_models/training_runs/char-new
```

The shared `FITTED_MODELS` list in `baselines.py` defines five approaches: majority, word TF-IDF, character TF-IDF, FinBERT and BGE. Keep character TF-IDF because it is our strongest cheap baseline. Majority is the useful class-imbalance floor; a random baseline adds little here. Length/keyword rules, SEC-specific cleaning, capped encoders and their separate cross-validation implementation have been removed.

Vocabularies/scalers fit training only. Classifier regularization, class weights and log-probability decision offsets maximize validation macro F1; test is scored after freezing choices. Offsets change decisions, not probability calibration. Avoid choosing further variants from test scores.

**Frozen encoder** means its pretrained weights never update: FinBERT/BGE encode generic 510-token context windows, produce one vector per window, then average all window vectors and normalize into one document vector for logistic regression. This retains the entire raw body without requiring it to fit in a single encoder call. `encoders.py` also supplies their attention heads and LoRA models; base Laya and OpenAI inference do not use those encoders. Neural adaptation learns attention over windows instead of averaging them, with one whole-document loss.

### Common 1,000-row test

| Approach | Agreement | Macro F1 | Mean ms | P50 ms | P95 ms | API cost/run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority | 65.6% | 0.264 | 0.13 | 0.13 | 0.18 | $0 |
| Word TF-IDF (fixed C, balanced) | 75.5% | 0.684 | 1.84 | 1.61 | 3.12 | $0 |
| Tuned word TF-IDF | 76.9% | 0.687 | 1.93 | 1.68 | 3.34 | $0 |
| Tuned character TF-IDF | 77.7% | 0.693 | 5.67 | 4.55 | 11.02 | $0 |
| Frozen FinBERT (raw/all windows) | 72.5% | 0.654 | 30.74 | 26.55 | 51.04 | $0 |
| FinBERT LoRA | 77.5% | 0.708 | 32.69 | 28.45 | 54.41 | $0 |
| Frozen BGE (raw/all windows) | 70.7% | 0.619 | 16.08 | 14.12 | 24.84 | $0 |
| BGE LoRA | 73.4% | 0.669 | 18.83 | 16.39 | 27.61 | $0 |
| Base Laya | 57.3% | 0.352 | 290.16 | 205.40 | 565.47 | $0 |
| Initial Laya head (raw decisions) | 65.7% | 0.347 | 256.18 | 178.57 | 535.29 | $0 |
| Weighted Laya head | 61.9% | 0.539 | 257.27 | 176.25 | 542.36 | $0 |
| Laya LoRA (one epoch) | 73.9% | 0.663 | 274.41 | 188.17 | 580.39 | $0 |
| Character TF-IDF + BGE LoRA | 77.3% | 0.695 | 25.26 | 22.00 | 38.53 | $0 |

These are completed results, not a rerun of this refactor. Retired length/keyword and cleaned/capped experiments remain in the ignored `experiments.md` and Git history; their scores are not presented as raw-input baselines. Historical Laya runs asked specifically about an 8-K; the shared default now says financial document. Saved checkpoints retain their original question, including on resume.

Agreement and macro F1 use the same frozen references. Latency is warmed, single-document inference, including transformation/tokenization and GPU synchronization; loading, fitting and result-file writes are excluded. Local runs have $0 API cost per run; hardware/electricity are excluded. RTX 3060 12 GB, global Python 3.13, PyTorch 2.14/CUDA 12.6; shared-machine timings are approximate.

The majority baseline is strong on agreement because 65.6% of references are review-worthy. Macro F1 weights all three labels equally. Earlier 50/500-row and cross-validation results are historical experiments, not rows in this comparable table. The test was used in earlier reports; a fresh human-reviewed holdout is needed before firm quality claims.

## Document adaptation (already completed; optional to reproduce)

Whole-document supervision: arbitrary context windows feed a learned attention pool and one document-level classification loss. No document label is assigned separately to each window. Laya uses native 312-token windows, stride 156; FinBERT/BGE use all 510-token windows.

```bash
# Supervised head only; frozen encoder
python decision_models/train.py --model laya --adaptation head --class-weight-power 1 --learning-rate 0.00003 --epochs 2 --output decision_models/training_runs/head-new
# FinBERT LoRA (NOT full encoder fine-tuning)
python decision_models/train.py --model finbert --adaptation lora --lora-rank 8 --class-weight-power 1 --learning-rate 0.001 --encoder-learning-rate 0.0001 --epochs 3 --output decision_models/training_runs/finbert-lora-new
python decision_models/benchmark.py --saved decision_models/training_runs/finbert-lora-new
```

`train.py` keeps model construction/windowing in `encoders.py`, leaving one training loop. `tune.py` handles fixed-split linear fitting and validation-only selection; it no longer unwraps nested cross-validation factories. Checkpoint state, feature caches and source snapshots are retained because interruptions and long runs are common on local hardware.

Use `--resume` with the same settings to continue; `--limit` makes a separate mechanics smoke run and never reads test. Training stores source snapshots, pinned revisions, settings, seed/runtime, optimizer/RNG state, validation history and best checkpoint. Selection includes epoch zero and uses validation F1 after offsets by default. Loss weights are applied before averaging document losses. LoRA updates query/value projections (Laya Wqkv), with rank 8, alpha 16 and dropout .05 in completed runs. Full FinBERT fine-tuning is excluded.

Other completed recipes use the same commands with these settings (new output directories):

| Variant | Model | Adaptation | Head LR | Encoder LR | Weight power | Epochs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Weighted Laya head | laya | head | .00003 | frozen | 1 | 2 |
| FinBERT LoRA | finbert | lora | .001 | .0001 | 1 | 3 |
| BGE LoRA | bge | lora | .001 | .0001 | 1 | 2 |
| Laya LoRA | laya | lora | .00003 | .0001 | 1 | 1 |

Completed word/character/frozen-linear grids use C = .001, .01, .1, 1, 10, 30, 100 and both unweighted/balanced classifiers. For a validation-selected blend, supply two fitted directories:

```bash
python decision_models/tune.py --blend decision_models/training_runs/char-new decision_models/training_runs/bge-lora-new --output decision_models/training_runs/blend-new
python decision_models/benchmark.py --saved decision_models/training_runs/blend-new
```

Checkpoints, search trials, predictions and source snapshots remain local under `training_runs/` and `benchmark_results/`. The exhaustive local `experiments.md` consolidates successes, failed trials, checks and interpretation; these generated artifacts are not committed. Future work should strengthen labeling, issuer/temporal splits and baseline robustness before further post-training or RL.
