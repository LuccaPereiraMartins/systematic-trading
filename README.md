# Systematic Trading

Two parts: a live MANU tick API (`app.py`) and offline SEC-filing triage models (`decision_models/`).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies are locked in `uv.lock`; commit it with any change to `pyproject.toml`.

```bash
cp env.example .env          # Windows: copy env.example .env
# fill in LSE_API_KEY, SEC_USER_AGENT, OPENAI_API_KEY (use your own keys)
uv sync --group models       # API + dev tools + models pipeline
uv sync --group models --group laya       # also Laya/torch (large)
uv sync --group models --group encoders    # also torch/transformers for the embedding and FinBERT baselines
```

## Tick API

```bash
uv run uvicorn app:app --reload   # http://localhost:8000/docs
# or: docker compose up --build
```

`GET /health` · `GET /tick`

## Decision models

Triage of SEC 8-K filings as `routine`, `review_worthy` or `unclear`. Run everything from the repository root; scores and
caveats are in [decision_models/benchmarks.md](decision_models/benchmarks.md).

| File | Purpose |
| --- | --- |
| `collect.py` | Download 8-K text from EDGAR into `dataset.json` |
| `label.py` | Label the dataset with an LLM (GPT-6 Luna); human labels in the `human` field take precedence |
| `baselines.py` | Baselines from trivial to fine-tuned: majority, length, keywords, TF-IDF, embeddings, FinBERT |
| `benchmark.py` | Fixed 50-filing pilot comparing Laya, OpenAI models and the baselines |
| `evaluate.py` | 5-fold cross-validation over all labelled filings with bootstrap confidence intervals |

```bash
uv run python decision_models/collect.py --number 50 --start 2025-01-01 --end 2025-03-31
uv run python decision_models/label.py

# 50-filing pilot (one results file per approach in decision_models/benchmark_results/)
uv run python decision_models/benchmark.py --models tfidf luna
uv run python decision_models/benchmark.py --models majority length_item keyword_learned
uv run python decision_models/benchmark.py --models laya --device cpu   # needs the laya group

# Cross-validated comparison of the trainable baselines (the more trustworthy numbers)
uv sync --group models --group encoders   # once, for the embedding and FinBERT baselines
uv run python decision_models/evaluate.py                       # cheap + frozen-encoder baselines
uv run python decision_models/evaluate.py --models finbert_ft   # fine-tuned FinBERT, much slower
```

Available baselines: `majority`, `length_raw`, `length_item`, `keyword_prior`, `keyword_learned`, `tfidf`, `bge_lr`,
`finbert_lr`, `finbert_ft`. How the length and keyword baselines were designed is explained in the docstring of
`baselines.py`.

All scores are agreement with LLM labels, not accuracy against ground truth, until the dataset has human labels.

## Development

```bash
uv run pytest
uv run ruff check .
```

CI (`.github/workflows/ci.yml`) runs the same checks, and the Docker image build runs separately.
