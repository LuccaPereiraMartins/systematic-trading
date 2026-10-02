# Decision model benchmarks

## Laya 8-K triage baseline

Compared with the current frontier-model labels on 500 SEC 8-Ks. These are agreement metrics, not ground-truth accuracy; the labels have not been human-reviewed yet.

| Model | Label agreement | Macro F1 | Runtime | Cost |
| --- | ---: | ---: | ---: | --- |
| Laya (`55cf4c4`) on CUDA | 69.4% | 41.9% | 126s for 500 filings (0.25s/filing) | No API cost; local GPU use |

Laya predicted 373 `review_worthy`, 127 `routine`, and no `unclear` filings. Its `review_worthy` recall was 84.0%; macro F1 reflects weaker performance on `routine` and `unclear`. Confidence values are not calibrated.

Run `python decision_models/benchmark.py` to reproduce this baseline. The script writes per-filing results to the ignored local file `decision_models/laya_benchmark.json`.
