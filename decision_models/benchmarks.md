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
