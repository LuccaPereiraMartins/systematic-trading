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
| `laya_infer.py` | Standalone CPU/GPU timing example with its own toy labels; not the held-out benchmark |

Run commands from the repository root after the [local setup](../README.md#local-setup). The root README covers the API and CI; this guide covers research. Generated data, checkpoints, predictions and the local `experiments.md` log are ignored, except for the committed split archive.

## Dataset and protocol

The committed `data/filings-10k-splits.zip` contains 8,000 train / 1,000 validation / 1,000 test JSONL rows, their manifest and the historical pilot inputs. Extract once:

```bash
python -m zipfile -e decision_models/data/filings-10k-splits.zip decision_models/data/splits
```

The 10,000 unique bodies span October 2025 through September 2026. Split seed 42, stratified by label, with 101 previously evaluated pilot bodies reserved for training. Test supports: 247 routine, 656 review-worthy, 97 unclear. The manifest records file hashes; loaders reject changed splits. Exact body overlap is zero, but issuers and near-duplicate templates can cross splits. This is an exploratory split, not a temporal or issuer-disjoint evaluation.

Each record contains only `date`, `body`, `llm` and `human`. Each annotation has `label` and `uncertainty`; human uncertainty stays its own value, including null or 0.0. Canonical bodies stay raw throughout the active pipelines. There is no Item/SIGNATURES trimming or registrant extraction; callers can supply cleaned text if they choose. Other source types use the same body input.

Collection and labeling are independent and resume existing files. To acquire a new dataset, use a separate file:

```bash
python decision_models/collect.py --number 10000 --start 2025-10-01 --end 2026-09-30 --sample --output decision_models/data/new-dataset.json
python decision_models/label.py --dataset decision_models/data/new-dataset.json
```

`--sample` distributes the target across calendar months; without arguments the collector selects the latest 50 indexed filings. Without explicit paths, collection and labeling use ignored `data/dataset.json`.

Use the archive for the established benchmark. Recreating its partition with `split.py` requires the exact original source JSON bytes and pilot inputs; a fresh live SEC pull will differ. `split.py` refuses to overwrite a different frozen partition. New datasets need a separate split output and evaluation protocol.

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

The shared `FITTED_MODELS` list in `baselines.py` defines five approaches: majority, word TF-IDF, character TF-IDF, FinBERT and BGE. Majority measures the class-imbalance floor; character TF-IDF is the strongest cheap baseline in the completed runs.

Vocabularies/scalers fit training only. Classifier regularization, class weights and log-probability decision offsets maximize validation macro F1; test is scored after freezing choices. Offsets change decisions, not probability calibration. Avoid choosing further variants from test scores.

**Frozen encoder** means its pretrained weights never update: FinBERT/BGE encode generic 510-token context windows, produce one vector per window, then average all window vectors and normalize into one document vector for logistic regression. This retains the entire raw body without requiring it to fit in a single encoder call. `encoders.py` also defines FinBERT/BGE attention heads and LoRA, plus adapted Laya models. Base Laya inference uses the Laya SDK directly; OpenAI inference uses the API.

Neural adaptation learns attention over windows instead of averaging them, with one whole-document loss.

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

These are completed results, not a rerun of the refactor. The fixed balanced word model is the API recipe; the core benchmark tunes its own word model. Historical Laya runs asked specifically about an 8-K; the shared default now says financial document. Saved checkpoints retain their original question, including on resume.

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

`train.py` owns the training loop, feature cache and resumable checkpoints; `encoders.py` owns model/window details. `tune.py` handles linear fitting and validation-only selection. Frozen FinBERT/BGE head training caches window features; LoRA recomputes features because encoder weights change.

Append `--resume` to the original training command to continue; `--limit` makes a separate mechanics smoke run and never reads test. Training stores source snapshots, pinned revisions, settings, seed/runtime, optimizer/RNG state, validation history and best checkpoint. Selection includes epoch zero and uses validation F1 after offsets by default. Loss weights are applied before averaging document losses. LoRA requires `--lora-rank 8` for the completed recipes: query/value projections (Laya Wqkv), alpha 16 and dropout .05. Full encoder fine-tuning is outside the current training options.

Completed recipes use these settings and separate output directories. For LoRA, include `--lora-rank 8`:

| Variant | Model | Adaptation | Head LR | Encoder LR | Weight power | Epochs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Weighted Laya head | laya | head | .00003 | frozen | 1 | 2 |
| FinBERT LoRA | finbert | lora | .001 | .0001 | 1 | 3 |
| BGE LoRA | bge | lora | .001 | .0001 | 1 | 2 |
| Laya LoRA | laya | lora | .00003 | .0001 | 1 | 1 |

Completed word/character/frozen-linear grids use C = .001, .01, .1, 1, 10, 30, 100 and both unweighted/balanced classifiers. Supply these values through `--regularization` to match that search; the default grid ends at 10. An optional blend selects its mixing weight and decision offsets on validation:

```bash
python decision_models/tune.py --blend decision_models/training_runs/char-new decision_models/training_runs/bge-lora-new --output decision_models/training_runs/blend-new
python decision_models/benchmark.py --saved decision_models/training_runs/blend-new
```

## Artifacts and historical experiments

`training_runs/` stores fitted models, validation/search results, configuration, source snapshots and neural checkpoints. `benchmark_results/` stores test predictions and metrics. `data/cache/` stores reusable frozen features. All are ignored; `experiments.md` is the single local log of completed and failed trials.

Retired length/keyword and cleaned/capped variants remain in that log and Git history. Their custom-transformer joblibs require the original source version; they are not supported by the active pipeline. Existing pure sklearn artifacts and neural checkpoints remain loadable. Source changes require `--refit` when using `evaluate.py`.

These historical runs are exploratory. The expanded study below adds fresh labels, temporal evaluation and
source coverage before further adaptation. Performance on news has not yet been established.

## Expanded research corpus

The [approved research programme](research.md) defines five dependent PRs. New inputs and outputs live under
`data/research/`; the original 10,000-row archive and its splits remain unchanged.

```bash
# Pilot each of 8k / releases / 6k / fed / ecb / news before scaling its target.
python decision_models/collect_sources.py --family fed --number 50 --output decision_models/data/research/pilot/fed.jsonl
# Prepare independently collected inputs; legacy rows are allowed in training only.
python decision_models/prepare.py --corpora decision_models/data/research/corpus/fed.jsonl decision_models/data/research/corpus/6k.jsonl --output decision_models/data/research/benchmark
# Label validation/test first, then training, all against ONE shared $3 ceiling.
python decision_models/label.py --dataset decision_models/data/research/benchmark/validation-input.jsonl
python decision_models/label.py --dataset decision_models/data/research/benchmark/test-input.jsonl
python decision_models/label.py --dataset decision_models/data/research/benchmark/train-input.jsonl
python decision_models/prepare.py --output decision_models/data/research/benchmark --freeze
python decision_models/review.py --splits decision_models/data/research/benchmark --output decision_models/data/research/review
```

Collection retains original responses in a SQLite cache, extraction hashes, source URLs, publication-date
precision and reuse attribution. Early pilots used equivalent raw-byte/metadata files, also readable by the
cache loader. SEC collection processes share one OS lock and the SDK's eight-request/second limit.
JSONL writes are resumable; failed requests remain retryable, permanent extraction/rights exclusions are logged.
Changing dates or family requires a new output. Monthly sampling uses stable hash ordering, including month
order, so small pilots do not always cover only the oldest months. Volume shortfalls are reported.

Sources use institutional public text: [SEC reuse policy](https://www.sec.gov/files/about/webmaster-faq.htm),
[Fed public-domain policy](https://www.federalreserve.gov/disclaimer.htm), and
[ECB attribution/accuracy conditions](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html).
ECB author-named documents are excluded. News collection accepts
[VOA-original text](https://www.voanews.com/p/5338.html) (agency material excluded) and
[Wikinews text](https://en.wikinews.org/wiki/Wikinews:Copyright) with its publication-date-specific CC BY license.
GDELT discovers links; it grants no publisher-content license. Only article text is retained, not image assets.

Luna Flex labeling records response IDs, rubric/model provenance, token usage and charges. Its shared ledger
reserves a conservative maximum before dispatch; interrupted or ambiguous requests retain reservations and
cannot be automatically duplicated. Capacity rejections release their reservation. SDK automatic retries are
disabled, service tier never upgrades, and successful ledger responses can restore an interrupted dataset.
Never change `--ledger` to bypass the study-wide ceiling. Remaining user credits are reserved for future rubric
changes. Independent API benchmarking and cloud compute are outside the currently authorised paid execution.

Preparation excludes old bodies/near duplicates from validation/test and quarantines related groups crossing
the March/June/September 2026 boundaries. Near-duplicate candidates use 128-permutation MinHash with seed 42;
merges require verified five-word-shingle Jaccard similarity of at least .90. Labeling is followed by an explicit
freeze; changed frozen outputs are rejected. Selection and calibration sets do not share related groups.
Human-review forms hide model labels; conflicting double reviews stay unresolved. The review audit is not an
expert assessment of investment usefulness.
