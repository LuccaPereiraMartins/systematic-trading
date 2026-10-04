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

## Cross-validated baselines

The 50-filing pilot cannot rank approaches (about ±10 points of noise). `evaluate.py` instead predicts every one of the
500 filings out of fold: 5 folds, stratified by label and grouped by registrant (466 distinct names, so one company's
filings never sit on both sides of a split), with bootstrap 95% confidence intervals. Scores are still agreement with
GPT-6 Luna labels, not accuracy. The run is seed 0 and not repeated, so fold assignment adds some uncertainty beyond the
intervals.

| Baseline | What it is | Agreement (95% CI) | Macro F1 | F1 routine / review / unclear |
| --- | --- | ---: | ---: | ---: |
| `majority` | Always the most common label | 68.6% (64.6–72.6) | 0.271 | 0.00 / 0.81 / 0.00 |
| `length_raw` | One length cut on the whole body | 56.2% (51.8–60.4) | 0.350 | 0.36 / 0.69 / 0.00 |
| `length_item` | One length cut on the Item-to-signature text | 62.0% (57.8–66.2) | 0.368 | 0.35 / 0.75 / 0.00 |
| `keyword_prior` | Hand-written material/routine lexicon + logistic regression | 58.0% (53.6–62.2) | 0.446 | 0.34 / 0.75 / 0.25 |
| `keyword_learned` | Top-k chi-squared uni/bi-grams + logistic regression | 72.0% (68.2–76.0) | 0.577 | 0.66 / 0.80 / 0.27 |
| `tfidf` | TF-IDF uni/bi-grams + logistic regression | 75.4% (71.6–79.0) | 0.639 | 0.72 / 0.83 / 0.37 |
| `bge_lr` | Frozen bge-small-en-v1.5 embeddings + logistic regression | 79.6% (76.2–83.0) | 0.703 | 0.71 / 0.86 / 0.54 |
| `finbert_lr` | Frozen FinBERT embeddings + logistic regression | 82.2% (78.6–85.4) | 0.709 | 0.73 / 0.88 / 0.51 |

`finbert_ft` (FinBERT fine-tuned end to end) is implemented but has not been run yet.

Reading the table:

- **Majority is the floor to beat.** Always answering `review_worthy` agrees 68.6% of the time, above the pilot's
  Laya result (66%, a different and smaller sample, so only indicative).
- **Length alone is weak.** Longer filings are more often review-worthy, but a single cut beats majority on macro F1
  only, not on agreement.
- **Hand-written keywords are weak, learned ones are better, but partly spurious.** `keyword_learned` chose k=250, the
  top of its search grid, and its strongest "routine" terms include dates ("august 31", "dated september"). Treat part
  of its signal as an artifact of 3 days of data.
- **Frozen encoders are the strongest baselines so far.** Their intervals overlap each other and TF-IDF's upper end,
  so only "encoder > keyword/length" is clear, not bge vs FinBERT.
- **`unclear` is hard for everything** (best F1 about 0.5, from 23 examples).
- Text baselines read only the substantive text (first "Item X.XX" heading to the signature block) except `length_raw`
  and `tfidf`, which use the whole body as in the original pilot. Encoders embed up to 4 windows of 510 tokens.

How the length and keyword baselines were designed (and the first length model that failed) is documented at the top of
[baselines.py](baselines.py).

## Run

Install dependencies with uv (see the repository README), then run from the repository root:

```bash
uv sync --group models --group encoders --group laya

# 50-filing pilot
uv run python decision_models/benchmark.py                         # laya, luna, sol, tfidf
uv run python decision_models/benchmark.py --models laya tfidf --device cuda
uv run python decision_models/benchmark.py --models majority length_item keyword_learned bge_lr

# Cross-validation over all 500 filings (writes decision_models/benchmark_results/cv.json)
uv run python decision_models/evaluate.py
uv run python decision_models/evaluate.py --models finbert_ft      # slow: trains 5 models
```

`benchmark.py` contains one `Benchmark` class. Its `run()` method runs the selected approaches, sharing the sample, reference labels, scoring, timing and checkpoint saving. Model names, pricing, reasoning effort and concurrency are class attributes; each approach has its own inference method. OpenAI models run sequentially, each with four concurrent Flex requests, using `OPENAI_API_KEY` from the repository `.env`.

Every successful prediction is saved under the ignored `decision_models/benchmark_results/` folder. Running again replaces that approach's local results. The fixed sample manifest and this summary are committed. The pilot numbers above are from the original run; inference was not rerun for the refactor.

## Earlier Laya run

An earlier, separate run on all 500 filings took 126s on CUDA and achieved 69.4% agreement / 0.419 macro F1. It is retained as historical context and is not directly comparable with the fixed 50-row pilot above.
