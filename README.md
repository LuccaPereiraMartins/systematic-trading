# Financial document triage

Benchmarks, post-training experiments and a small CPU API for financial-document review.

## API

```bash
docker compose up --build
```

Open http://localhost:8000/docs. `GET /health` reports readiness; `POST /predict` accepts:

```json
{"body": "The company entered a definitive merger agreement."}
```

It returns `label` (`routine`, `review_worthy`, `unclear`) and class `probabilities`. The image builds a ~6 MB balanced word TF-IDF model from the frozen 8,000-row training split. It uses the completed baseline's validation-selected decision offsets; returned probabilities are uncalibrated and the selected label need not have the highest raw probability. No keys or GPU are needed for this API.

## Local setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Install into your existing Python:

```bash
uv export --frozen --no-dev --no-emit-project | uv pip install --system -r -
python -m decision_models.build_model
uvicorn app:app --reload
```

For collection, labeling and neural experiments:

```bash
uv pip install --system --group models --group encoders --group laya
cp env.example .env  # Windows: copy env.example .env
```

GPU experiments need CUDA-enabled PyTorch. Never commit `.env`. The API does not use it.

## Research

See [the benchmark guide](decision_models/README.md) for data, commands, comparable results and reproducibility. References remain provisional LLM labels rather than verified investment usefulness.

## Development

CI runs lint, builds the model/API image and checks health/inference on branch pushes and pull requests. There is no separate test suite. Update and commit `uv.lock` with dependency changes.
