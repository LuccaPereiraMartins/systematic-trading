FROM python:3.11-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project > /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt

COPY decision_models/baselines.py decision_models/schemas.py decision_models/build_model.py ./decision_models/
COPY decision_models/data/filings-10k-splits.zip ./decision_models/data/
RUN python -m decision_models.build_model
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
