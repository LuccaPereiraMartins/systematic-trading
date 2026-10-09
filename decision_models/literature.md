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

The contribution can be a negative adaptation result or a strong lexical baseline. It must rest on this
study's observed quality, misses, coverage and cost, with teacher agreement separated from independent
non-expert review. Missing recent journalistic coverage limits any wider financial-news claim.
