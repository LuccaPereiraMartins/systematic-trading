FROM python:3.11-slim

WORKDIR /app

# install packages with uv from the lockfile (API dependencies only)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project > /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt

COPY app.py .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
