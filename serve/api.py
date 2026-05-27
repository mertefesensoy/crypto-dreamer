"""
FastAPI WebSocket bridge plus history endpoints.

Live tail: agents PUBLISH to dreamer:steps / dreamer:episodes; this service
subscribes and forwards to any connected WebSocket. Decoupling is deliberate
- the trainer/agent doesn't know the dashboard exists.

History: agents also XADD to dreamer:steps:hist (capped) and
dreamer:episodes:hist. This service exposes /episodes and /history GET
endpoints so the dashboard can hydrate on mount and back-fill earlier steps.

Run:
    uvicorn serve.api:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

CHANNELS = ["dreamer:steps", "dreamer:episodes"]
STREAM_STEPS = "dreamer:steps:hist"
STREAM_EPISODES = "dreamer:episodes:hist"

REDIS_HOST = os.environ.get("DREAMER_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("DREAMER_REDIS_PORT", "6379"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.redis = aioredis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
    )
    try:
        yield
    finally:
        await app.state.redis.aclose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase 5.4: dream player endpoint. Lazy-loads the world model on first
# request so the API stays responsive when no checkpoint is present.
from serve.dream_endpoint import router as _dream_router  # noqa: E402

app.include_router(_dream_router)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/episodes")
async def episodes() -> list[dict[str, Any]]:
    """All episode summaries from dreamer:episodes:hist, oldest first."""
    entries = await app.state.redis.xrange(STREAM_EPISODES, "-", "+")
    return [json.loads(fields["data"]) for _id, fields in entries]


@app.get("/history")
async def history(
    episode: int,
    count: int = 2000,
    before_step: int | None = None,
) -> list[dict[str, Any]]:
    """Step events for `episode`, ordered chronologically.

    `count` is a cap on returned rows. `before_step`, if given, restricts
    to events strictly earlier than that step (used by the dashboard's
    'Load earlier' affordance).

    Implementation walks dreamer:steps:hist in reverse with a chunk
    sized to `count + headroom` so a single Redis call covers an entire
    random-agent run. Capped at 250k entries to keep the API responsive
    if the stream gets large (5M maxlen). v2 will move to a per-episode
    index when single-pass scan stops being fast enough.
    """
    if count <= 0:
        raise HTTPException(status_code=400, detail="count must be > 0")

    chunk_size = min(max(count * 4, 5_000), 250_000)
    entries = await app.state.redis.xrevrange(
        STREAM_STEPS, "+", "-", count=chunk_size
    )

    matched: list[dict[str, Any]] = []
    for _entry_id, fields in entries:
        payload = json.loads(fields["data"])
        if payload.get("episode") != episode:
            continue
        if before_step is not None and payload.get("step", 0) >= before_step:
            continue
        matched.append(payload)
        if len(matched) >= count:
            break

    matched.reverse()
    return matched


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    pubsub = app.state.redis.pubsub()
    await pubsub.subscribe(*CHANNELS)
    try:
        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            try:
                await websocket.send_json(
                    {"channel": msg["channel"], "data": json.loads(msg["data"])}
                )
            except (WebSocketDisconnect, RuntimeError, ConnectionError):
                # Client gone or socket closed mid-send. Exit listener cleanly.
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await pubsub.unsubscribe(*CHANNELS)
        finally:
            await pubsub.aclose()
