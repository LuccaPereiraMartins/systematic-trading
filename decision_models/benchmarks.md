# Financial text triage benchmarks

The pilot uses a fixed, class-stratified sample of 50 from `dataset.json` (13 routine, 34 review-worthy, 3 unclear). The sample manifest stores body hashes in `benchmark_sample.json`, so every runner evaluates the same filings. TF-IDF trains on the other 450 records.

All 500 records currently use GPT-6 Luna labels. Scores below are **agreement with provisional references**, not human accuracy; GPT-6 Luna's score is a self-consistency check. Only three pilot examples are `unclear`, so that class's metrics are especially noisy.

| Approach | Agreement | Macro F1 | Inference p50 / p95 | Estimated cost per 50-filing run |
| --- | ---: | ---: | ---: | ---: |
| Laya on CUDA | 66% | 0.376 | 230 / 542 ms | $0 |
| GPT-6 Luna, Flex, low | 90% | 0.812 | 1.43 / 3.25s | $0.0056 |
| GPT-6.1 Sol, Flex, low | 94% | 0.794 | 3.36 / 4.98s | $0.1052 |
| TF-IDF + logistic regression | 76% | 0.589 | 1.25 / 2.40ms | $0 |

OpenAI costs are estimates from returned token usage and the short-context Flex rates at run time; check account billing for the actual charge. The estimate uses rates in [OpenAI API pricing](https://developers.openai.com/api/docs/pricing?tab=suite). The tiny sample is a pipeline smoke benchmark, not enough to rank models confidently.

## Run

Install the benchmark dependencies into global Python (versions used for this run):

```powershell
uv pip install --system laya==0.3.21 openai==3.22.1 python-dotenv==1.2.2 pydantic==2.13.5 scikit-learn==1.9.1
```

Run all four approaches from the repository root:

```powershell
python decision_models/benchmark.py
# Or select approaches and an explicit Laya device:
python decision_models/benchmark.py --models laya tfidf --device cuda
```

`benchmark.py` contains one `Benchmark` class. Its `run()` method runs the selected approaches, sharing the sample, reference labels, scoring, timing and checkpoint saving. Model names, pricing, reasoning effort and concurrency are class attributes; each approach has its own inference method. OpenAI models run sequentially, each with four concurrent Flex requests, using `OPENAI_API_KEY` from the repository `.env`.

Every successful prediction is saved under the ignored `decision_models/benchmark_results/` folder. Running again replaces that approach's local results. The fixed sample manifest and this summary are committed. The numbers above are from the original pilot; inference was not rerun for the refactor.

## Earlier Laya run

An earlier, separate run on all 500 filings took 126s on CUDA and achieved 69.4% agreement / 0.419 macro F1. It is retained as historical context and is not directly comparable with the fixed 50-row pilot above.

## Larger collection

Collect 10,000 filings across the last 12 complete months, independently of labeling:

```powershell
python decision_models/collect.py --number 10000 --start 2025-10-01 --end 2026-09-30 --sample --output decision_models/data/dataset.json
```

`--sample` selects filings in deterministic hash order, taking turns across months. Failed retrievals and duplicate bodies are replaced from the remaining candidates. Successful records are checkpointed every 100 additions; reruns preserve them and stop at the requested dataset size. The larger data folder is ignored by Git; the original labeled 500 and benchmark sample remain committed. Collection makes no OpenAI calls. `label.py` remains on Luna/Flex with low reasoning and a 1,024-token output allowance.

Label the collected file independently of collection:

```powershell
python decision_models/label.py --dataset decision_models/data/dataset.json
```

Labeling skips records that already have a human or LLM label and saves progress back to the selected file every 250 responses.

## Frozen 10k split

The collected 10,000 filings are now labeled with Luna/Flex, low reasoning. `split.py` uses a fixed seed (42) and label stratification to create 8,000 training, 1,000 validation and 1,000 test records across the year. The 101 bodies also present in the original 500-row pilot are reserved for training, so validation and test contain no previously benchmarked pilot bodies. Input is sorted by body hash before selection, making membership independent of source row order. Test is reserved for final comparisons; validation supplies checkpoint selection, tuning and calibration. This measures performance on new sampled documents; it is not a chronological or issuer-disjoint evaluation.

| Split | Rows | Dates (inclusive) | Routine / review-worthy / unclear |
| --- | ---: | --- | --- |
| Train | 8,000 | 2025-10-01 to 2026-09-30 | 1,981 / 5,253 / 766 |
| Validation | 1,000 | 2025-10-01 to 2026-09-30 | 247 / 656 / 97 |
| Test | 1,000 | 2025-10-01 to 2026-09-30 | 247 / 656 / 97 |

```powershell
python decision_models/split.py
python decision_models/benchmark.py --models laya --device cuda --splits decision_models/data/splits
```

`data/splits/` contains `train.jsonl`, `validation.jsonl`, `test.jsonl` and `manifest.json`. The manifest records counts, date ranges, the source fingerprint and split checksums. Rerunning the split produces identical files and refuses to overwrite different existing split contents. The benchmark verifies split checksums and excludes validation from TF-IDF training. Held-out results use separate `*-heldout.json` files under `benchmark_results/`; base Laya uses the same pinned checkpoint and `predict_long()` as the pilot, with a warm-up before timing.

The source JSON SHA-256 is `b2f8b418b116b07bafaae400279a21be5a7ccf7012d6d353d1abed2d3567829f`; the test JSONL SHA-256 is `0adfc7e1d37a0820cad3d2ac56ec7042d8d8b0d44ee55ff0ebb912a876545b42`. Extracted files remain ignored by Git; the frozen split ZIP is committed. Human corrections should create a new dataset/split version, preserving the benchmark's original references.

The committed `data/filings-10k-splits.zip` (14.1 MB compressed, 78.7 MB extracted) contains all 10,000 rows across the three JSONL files and the manifest. JSONL retains the full text and annotation schema, with one record per line. Extract it after cloning, then use the benchmark command above:

```powershell
python -m zipfile -e decision_models/data/filings-10k-splits.zip decision_models/data/splits
```

Regenerate the ZIP with:

```powershell
python -m zipfile -c decision_models/data/filings-10k-splits.zip decision_models/data/splits/train.jsonl decision_models/data/splits/validation.jsonl decision_models/data/splits/test.jsonl decision_models/data/splits/manifest.json
```

See [post_training.md](post_training.md) for the local training commands, experiment history and reproducibility requirements.

## Laya on the held-out 1,000

Run on 2026-10-04 using the frozen test split above, `laya==0.3.21`, `torch==2.14.0+cu126`, `transformers==5.17.0`, and an RTX 3060 (12 GB). The English checkpoint is pinned to revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. Base Laya evaluated the stored full bodies through `predict_long()` with default overlapping windows and its most-confident-window choice aggregation, without training, cleaning or prompt tuning. The supervised variant retains the raw bodies and uses learned document attention, as detailed below.

| Approach | Agreement with Luna | Macro F1 | Inference p50 / p95 | API cost per 1,000-filing run |
| --- | ---: | ---: | ---: | ---: |
| Base Laya, CUDA | 57.3% | 0.352 | 205 / 565 ms | $0 |
| Supervised head + attention, CUDA (epoch 1) | 65.7% | 0.347 | 179 / 535 ms | $0 |
| Always review-worthy (majority label in training) | 65.6% | 0.264 | — | $0 |

Mean inference latency was 290 ms for base Laya and 256 ms for the adapted model, after warm-up and including tokenization/window handling. Timing excludes result-file writes. Local costs exclude electricity and hardware. OneDrive file locks use the same checkpoint-write retries as labeling.

### Base per-label results

| Label | Reference count | Predicted count | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Routine | 247 | 351 | 27.9% | 39.7% | 0.328 |
| Review-worthy | 656 | 649 | 73.2% | 72.4% | 0.728 |
| Unclear | 97 | 0 | 0% | 0% | 0.000 |

Laya missed 181 of the 656 review-worthy references and never predicted unclear. Agreement is below the majority baseline, while macro F1 is higher. These provisional teacher labels are not human ground truth. The result supports investigating adaptation and document handling; it does not establish that either will solve the errors. `predict_long()` confidence comes from the selected window and is not calibrated for an entire filing. The checkpoint also warns about inherited temperature values, so confidence-based gating needs separate validation.

### First supervised adaptation

The selected epoch-1 checkpoint was trained on 8,000 raw documents and selected using macro F1 on the separate 1,000-document validation set. Training froze the encoder and adapted the existing head plus a small attention pooler, using one whole-document loss across all overlapping windows. One epoch took 44.0 minutes locally, including both validation passes, with 2.72 GiB peak allocated CUDA memory. This comparison changes both head weights and document aggregation; it does not isolate either contribution.

```powershell
python decision_models/train.py
python decision_models/benchmark.py --models head --device cuda --splits decision_models/data/splits --checkpoint decision_models/training_runs/head/best.pt
```

| Label | Reference count | Predicted count | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Routine | 247 | 11 | 54.5% | 2.4% | 0.047 |
| Review-worthy | 656 | 960 | 66.5% | 97.3% | 0.790 |
| Unclear | 97 | 29 | 44.8% | 13.4% | 0.206 |

The adapted model missed 18 review-worthy references, versus 181 for the base model, but flagged 96% of documents as review-worthy. Agreement is only 0.1 percentage point above the majority baseline, and macro F1 is slightly below base Laya. The training pipeline works; this checkpoint does not yet offer useful filtering. Next compare class-balanced loss and learning rates on validation. Document probabilities remain uncalibrated. All 1,000 test body hashes, the split manifest and the selected checkpoint fingerprint were verified; results are saved locally in `benchmark_results/head-attention-heldout.json`.
