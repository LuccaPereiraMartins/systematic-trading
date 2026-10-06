# Systematic Trading

Live MANU tick API (`app.py`) and offline financial-document triage (`decision_models/`).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

```bash
cp env.example .env  # Windows: copy env.example .env
# Fill the keys needed for your task; never commit .env.
uv pip install --system --group models --group encoders --group laya
```

This installs into your existing Python. Omit the model groups for API-only work. GPU runs need a CUDA-enabled PyTorch installation; use the same runtime for comparable timings.

## Tick API

```bash
uvicorn app:app --reload  # http://localhost:8000/docs
# or: docker compose up --build
```

Endpoints: `GET /health`, `GET /tick`.

## Decision models

See [the benchmark guide](decision_models/README.md) for data, commands, results and reproducibility. Labels are `routine`, `review_worthy`, `unclear`: does this document merit an analyst's closer review?

## Development

```bash
python -m pytest
ruff check .
```

CI runs those checks; the Docker build runs separately. Update and commit `uv.lock` with dependency changes.
