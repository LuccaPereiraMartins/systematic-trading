# Financial text triage benchmarks

The pilot uses a fixed, class-stratified sample of 50 from `dataset.json` (13 routine, 34 review-worthy, 3 unclear). The sample manifest stores body hashes in `benchmark_sample.json`, so every runner evaluates the same filings. TF-IDF trains on the other 450 records.

All 500 records currently use GPT-6 Luna labels. Scores below are **agreement with provisional references**, not human accuracy; GPT-6 Luna's score is a self-consistency check. Only three pilot examples are `unclear`, so that class's metrics are especially noisy.

| Approach | Agreement | Macro F1 | Inference p50 / p95 | Run time | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya on CUDA | 66% | 0.376 | 230 / 542 ms | 21.5s including 8.2s model load | $0 |
| GPT-6 Luna, Flex, low | 90% | 0.812 | 1.43 / 3.25s | 21.5s | $0.0056 |
| GPT-6.1 Sol, Flex, low | 94% | 0.794 | 3.36 / 4.98s | 45.8s | $0.1052 |
| TF-IDF + logistic regression | 76% | 0.589 | 1.25 / 2.40ms | 0.72s including 0.65s fit | $0 |

OpenAI costs are estimates from returned token usage and the short-context Flex rates at run time; check account billing for the actual charge. The estimate uses rates in [OpenAI API pricing](https://developers.openai.com/api/docs/pricing?tab=suite). The tiny sample is a pipeline smoke benchmark, not enough to rank models confidently.

## Run

Install the benchmark dependencies into global Python (versions used for this run):

```powershell
uv pip install --system laya==0.3.21 openai==3.22.1 python-dotenv==1.2.2 pydantic==2.13.5 scikit-learn==1.9.1
```

Run each approach separately from the repository root:

```powershell
python decision_models/benchmark.py --device cuda
python decision_models/benchmark_luna.py
python decision_models/benchmark_sol.py
python decision_models/benchmark_tfidf.py
```

The OpenAI scripts use `OPENAI_API_KEY` from the repository `.env`. All prediction JSON is written under the ignored `decision_models/benchmark_results/` folder; the fixed sample manifest and this summary are committed.

## Earlier Laya run

An earlier, separate run on all 500 filings took 126s on CUDA and achieved 69.4% agreement / 0.419 macro F1. It is retained as historical context and is not directly comparable with the fixed 50-row pilot above.
