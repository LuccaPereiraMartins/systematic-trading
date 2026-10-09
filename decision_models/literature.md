# Research context and primary sources

Checked 2026-10-09. This is a source-content-only financial triage study: agreement with review-oriented
labels, dangerous review-to-routine errors, adaptation sample efficiency and throughput. It does not assess
sentiment quality, investment materiality, historical novelty or trading profitability.

## Why these comparisons

- [Laya's pinned model card](https://huggingface.co/convaiinnovations/laya/blob/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/README.md)
  describes a ModernBERT-large backbone with learned typed-decision heads and RLCD pretraining. Its financial
  triage behavior remains an empirical question. We use the English root checkpoint, not its router or a
  workflow-specialized checkpoint. Provider benchmarks use different tasks and cannot supply our accuracy,
  latency or calibration results. Supervised adaptation is compared with matched pre-update weights;
  ModernBERT controls for the backbone. A proper-scoring training objective does not establish calibration
  on a new financial source distribution.
- [Araci's FinBERT paper](https://arxiv.org/abs/1908.10063) and the
  [ProsusAI checkpoint card](https://huggingface.co/ProsusAI/finbert) concern financial sentiment. Our fresh
  triage classifier uses its encoder; its positive/negative/neutral sentiment head is not mapped to triage
  labels. [Yang et al.'s FinBERT](https://arxiv.org/abs/2006.08097) is a different checkpoint family and is
  not the model run here. Sentiment improvements do not prove triage improvements.
- [BGE's model card](https://huggingface.co/BAAI/bge-small-en-v1.5) describes an embedding/retrieval model.
  Its frozen representation and adapted classifier test transfer to this task rather than reproduce its
  retrieval benchmark. [ModernBERT](https://arxiv.org/abs/2412.13663) supplies a modern bidirectional encoder
  with native long-context capacity; this study independently measures its triage behavior and local cost.
- [The Qwen3 technical report](https://arxiv.org/abs/2505.09388) describes a decoder-model family with
  thinking and non-thinking operation. Our pinned 1.7B/4B models use non-thinking A/B/C scoring, a bounded
  4096-token input and conditional discriminative adaptation. These are specific deployment choices, not
  a claim to reproduce Qwen's full benchmark or a paid frontier model's quality.

## What the methods establish

- [LoRA](https://arxiv.org/abs/2106.09685) motivates low-rank parameter adaptation;
  [QLoRA](https://arxiv.org/abs/2305.14314) motivates NF4 quantized-base adaptation. Our fixed rank,
  objectives, data sizes and hardware are reported separately from the papers' setups and results.
- [Guo et al.](https://arxiv.org/abs/1706.04599) motivate scalar temperature scaling. We fit it on separate
  calibration groups and report raw and calibrated scores. A scalar temperature changes confidence but
  preserves argmax labels; empirical discard thresholds are not population miss-rate guarantees.
- Larger Laya windows change contiguous context and the number of votes under fixed pooling. Every source
  token remains covered at every tested window size. This tests the deployed windowing scheme's context
  tradeoff, rather than the benefit of exposing previously omitted source text.
- Paired event-group bootstrap intervals condition on the one trained seed and this cohort. Context,
  adaptation and sample-size differences use shared resamples; they do not quantify training instability
  or correct multiple comparisons. Tuning labels count in addition to each curve's training rows.

## Labels and the scope of the claim

- [Gilardi, Alizadeh and Kubli (PNAS, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10372638/)
  compare GPT-3.5 annotations with crowd workers using trained-human references for political tweets and
  news. Their results support the potential of low-cost model annotation, but concern different models,
  concepts and documents. They do not establish Luna's accuracy for financial review triage.
- [Pangakis, Wolken and Fasching (2023)](https://arxiv.org/abs/2306.00176) evaluate GPT-4 annotation across
  27 tasks and 11 social-science datasets. Performance varies with task and dataset; they argue for
  task-specific validation against human references. A large automatically labeled corpus alone does
  not resolve the validity of the labels.

In this study, Luna supplies the operational labeling convention. Training and evaluation labels come
from that teacher, so held-out scores measure how well each approach reproduces its triage decisions on
new documents. Comparisons can establish imitation quality, measured resource requirements and sample
efficiency under this convention. They cannot establish how often an analyst would miss an economically
material development. The returned non-expert rubric pilot informs clarity and ambiguous cases; its
disagreements do not override teacher labels or estimate analyst accuracy. Any later blinded non-expert
audit remains a separately qualified result. Independent expert validation would be needed for stronger
claims about analyst performance; it is not replaced by teacher self-reported uncertainty.

The contribution can be a negative adaptation result or a strong lexical baseline. It must rest on this
study's observed quality, misses, coverage and cost, with teacher agreement separated from independent
non-expert review. Missing recent journalistic coverage limits any wider financial-news claim.
