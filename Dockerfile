FROM python:3.13-slim

WORKDIR /app

# OS deps for duckdb, pandas wheels are sufficient on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy package manifest first so layer caches when only source changes.
COPY pyproject.toml README.md ./
COPY data data
COPY envs envs
COPY agents agents
COPY serve serve

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    DREAMER_REDIS_HOST=redis \
    DREAMER_REDIS_PORT=6379

EXPOSE 8000

CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
