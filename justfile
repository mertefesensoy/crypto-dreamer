# crypto-dreamer task runner. Install just: `winget install Casey.Just`
# or use `make <target>` which delegates here.

set windows-shell := ["cmd", "/c"]

# Default target lists what's available.
default:
    @just --list

# Spin up local Redis (portable Windows zip first, falls back to docker).
redis:
    @if exist .tools\redis\redis-server.exe ( \
        start /B .tools\redis\redis-server.exe --port 6379 --bind 127.0.0.1 --save "" --dir .tools\redis \
    ) else ( \
        docker run -d --rm --name dreamer-redis -p 6379:6379 redis:7-alpine \
    )

# Stop the local Redis (works for both portable and docker variants).
redis-stop:
    @docker rm -f dreamer-redis 2>nul || taskkill /IM redis-server.exe /F 2>nul || echo "no redis running"

# Pull historical 1m BTCUSDT klines into DuckDB. Idempotent on re-run.
ingest years="2":
    uv run python -m data.ingest --years {{years}}

# Start the FastAPI WebSocket bridge.
api:
    uv run uvicorn serve.api:app --host 127.0.0.1 --port 8000 --reload

# Run the random agent. `episodes`, `hours`, `speed`, `seed` overridable.
agent episodes="1" hours="24" speed="20" seed="42":
    uv run python -m agents.run_random \
        --episodes {{episodes}} --episode-hours {{hours}} \
        --speed {{speed}} --seed {{seed}}

# Run the dashboard dev server.
ui:
    npm run dev --prefix dashboard

# Run pytest.
test:
    uv run pytest -q

# Run typecheck on the dashboard.
typecheck:
    cd dashboard && npx tsc -p tsconfig.json --noEmit

# Run redis + api + ui together. Agent stays manual so you can pick episodes/seeds.
dev:
    npx -y concurrently --names "redis,api,ui" --prefix-colors "magenta,cyan,green" \
        "just redis" \
        "just api" \
        "just ui"

# Spin up the docker-compose stack (redis + api + ui).
up:
    docker compose up -d

# Stop the docker-compose stack.
down:
    docker compose down

# Tail logs from the docker-compose stack.
logs:
    docker compose logs -f
