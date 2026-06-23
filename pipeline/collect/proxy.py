#!/usr/bin/env python3
"""DEPRECATED — Logging proxy between opencode and llama-server.

Session tracking is now built into opencode natively. This extra hop adds
latency and should not be used for new workflows. Kept for replaying legacy
traces only.

Forwards every request to the upstream OpenAI-compatible server and, for
/v1/chat/completions, writes one JSON record per request to raw/opencode/
containing BOTH the request and the (reassembled) response:

    {
      "timestamp": "...",
      "upstream": "<hostname>",
      "request":  { ... original body: model, messages, tools, ... },
      "response": {
        "message": {"role": "assistant", "content": ..., "reasoning_content": ...,
                     "tool_calls": [...]},
        "finish_reason": "stop" | "tool_calls" | ...,
        "usage": {...} | null
      },
      "elapsed_ms": 1234
    }

The upstream field captures the hostname of the target server (e.g., "127.0.0.1",
"modal-server", "api.example.com"). For OpenRouter routes, it records "openrouter".

Streamed (SSE) responses are relayed to the client untouched while being
accumulated for the log, so opencode behaves exactly as if connected directly.

Usage:
    python collect/proxy.py --port 8765 --upstream http://127.0.0.1:8080

Then point opencode at it (~/.config/opencode/opencode.jsonc):
    "baseURL": "http://127.0.0.1:8765/v1"

Frontier routing
----------------
Requests whose path starts with /openrouter/ are forwarded to
https://openrouter.ai/api with the OPENROUTER_API_KEY env var injected as the
Authorization header, and logged identically (records carry "upstream":
"openrouter" so the dataset builder can label them as distillation). Register
an opencode provider with baseURL http://127.0.0.1:8765/openrouter/v1 to
capture frontier trajectories in the opencode harness format.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(line_buffering=True)

LOG_DIR = Path(__file__).resolve().parent.parent / "raw" / "opencode"
LOG_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_BASE = "https://openrouter.ai/api"
OPENROUTER_PREFIX = "openrouter/"

_HOP_BY_HOP = {
    "host", "content-length", "accept-encoding", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade",
}


def _filter_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _mark_cache(message: dict) -> dict:
    """Attach an Anthropic cache_control breakpoint to a message's content."""
    m = dict(message)
    c = m.get("content")
    if isinstance(c, str) and c:
        m["content"] = [{"type": "text", "text": c,
                         "cache_control": {"type": "ephemeral"}}]
    elif isinstance(c, list) and c:
        parts = [dict(p) if isinstance(p, dict) else p for p in c]
        for p in reversed(parts):
            if isinstance(p, dict) and p.get("type") == "text":
                p["cache_control"] = {"type": "ephemeral"}
                break
        m["content"] = parts
    return m


def prepare_frontier_body(body: dict) -> dict:
    """Enable reasoning and (for Anthropic models) prompt caching on
    OpenRouter-bound chat requests. Reasoning is essential for distillation:
    without it the trajectories carry no thinking to train on."""
    out = dict(body)
    out.setdefault("reasoning", {"effort": "medium"})
    if str(out.get("model", "")).startswith("anthropic/"):
        msgs = out.get("messages")
        if isinstance(msgs, list) and msgs:
            msgs = list(msgs)
            # breakpoints: system prompt (covers tools too) + latest message,
            # so each agent step re-reads the previous step's prefix from cache
            if isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
                msgs[0] = _mark_cache(msgs[0])
            if len(msgs) > 1:
                msgs[-1] = _mark_cache(msgs[-1])
            out["messages"] = msgs
    return out


class StreamAccumulator:
    """Reassemble an assistant message from OpenAI streaming chunks."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self._buf = b""

    def feed(self, raw: bytes) -> None:
        self._buf += raw
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                self.usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    self.finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    self.content.append(delta["content"])
                # llama.cpp uses reasoning_content; OpenRouter uses reasoning
                if delta.get("reasoning_content"):
                    self.reasoning.append(delta["reasoning_content"])
                elif isinstance(delta.get("reasoning"), str) and delta["reasoning"]:
                    self.reasoning.append(delta["reasoning"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = self.tool_calls.setdefault(
                        idx,
                        {"id": None, "type": "function",
                         "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]

    def message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant",
                               "content": "".join(self.content) or None}
        if self.reasoning:
            msg["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            msg["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        return msg


def write_record(request_body: dict, response: dict, elapsed_ms: int,
                 upstream: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    rid = uuid.uuid4().hex[:8]
    out = LOG_DIR / f"{ts}-{rid}.json"
    record = {
        "timestamp": ts,
        "upstream": upstream,
        "request": request_body,
        "response": response,
        "elapsed_ms": elapsed_ms,
    }
    out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return out


def announce(body: dict, response: dict, log_path: Path, upstream: str) -> None:
    t = time.strftime("%H:%M:%S")
    n_msgs = len(body.get("messages") or [])
    msg = response.get("message") or {}
    n_tc = len(msg.get("tool_calls") or [])
    fin = response.get("finish_reason")
    print(f"[{t}] [{upstream}] chat/completions msgs={n_msgs} tool_calls={n_tc} "
          f"finish={fin} -> {log_path.name}")


def make_app(upstream: str) -> FastAPI:
    timeout = httpx.Timeout(connect=10.0, read=3600.0, write=120.0, pool=10.0)
    client = httpx.AsyncClient(base_url=upstream.rstrip("/"), timeout=timeout)
    or_client = httpx.AsyncClient(base_url=OPENROUTER_BASE, timeout=timeout)
    or_key = os.environ.get("OPENROUTER_API_KEY", "")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()
        await or_client.aclose()

    app = FastAPI(title="opencode trace proxy", lifespan=lifespan)

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def proxy(full_path: str, request: Request):
        raw_body = await request.body()
        body: Any = None
        if raw_body:
            with contextlib.suppress(json.JSONDecodeError):
                body = json.loads(raw_body)

        is_chat = full_path.rstrip("/").endswith("chat/completions")
        fwd_headers = _filter_headers(dict(request.headers))
        if full_path.startswith(OPENROUTER_PREFIX):
            upstream_name = "openrouter"
            http = or_client
            upstream_path = "/" + full_path[len(OPENROUTER_PREFIX):]
            if not or_key:
                return JSONResponse(
                    status_code=500,
                    content={"error": "OPENROUTER_API_KEY not set in proxy env"})
            fwd_headers["authorization"] = f"Bearer {or_key}"
            if is_chat and isinstance(body, dict):
                body = prepare_frontier_body(body)
                raw_body = json.dumps(body).encode()
        else:
            upstream_name = urlparse(upstream).hostname or "local"
            http = client
            upstream_path = "/" + full_path
        params = dict(request.query_params)
        is_stream = bool(isinstance(body, dict) and body.get("stream"))
        started = time.monotonic()

        if is_chat and is_stream and isinstance(body, dict):
            acc = StreamAccumulator()

            async def relay():
                try:
                    async with http.stream(
                        request.method, upstream_path,
                        content=raw_body or None,
                        headers=fwd_headers, params=params,
                    ) as resp:
                        async for chunk in resp.aiter_raw():
                            acc.feed(chunk)
                            yield chunk
                except Exception as exc:  # noqa: BLE001
                    err = json.dumps({"error": f"upstream failed: {exc!r}"})
                    yield f"data: {err}\n\ndata: [DONE]\n\n".encode()
                    return
                elapsed = int((time.monotonic() - started) * 1000)
                response = {
                    "message": acc.message(),
                    "finish_reason": acc.finish_reason,
                    "usage": acc.usage,
                }
                log_path = write_record(body, response, elapsed, upstream_name)
                announce(body, response, log_path, upstream_name)

            return StreamingResponse(relay(), media_type="text/event-stream")

        try:
            resp = await http.request(
                request.method, upstream_path,
                content=raw_body or None,
                headers=fwd_headers, params=params,
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=502,
                                content={"error": f"upstream failed: {exc!r}"})

        content_type = resp.headers.get("content-type", "")
        if is_chat and isinstance(body, dict) and resp.status_code == 200 \
                and content_type.startswith("application/json"):
            with contextlib.suppress(ValueError):
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message")
                if isinstance(message, dict) and message.get("reasoning") \
                        and not message.get("reasoning_content"):
                    message = {**message,
                               "reasoning_content": message["reasoning"]}
                response = {
                    "message": message,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage"),
                }
                elapsed = int((time.monotonic() - started) * 1000)
                log_path = write_record(body, response, elapsed, upstream_name)
                announce(body, response, log_path, upstream_name)

        if content_type.startswith("application/json"):
            with contextlib.suppress(ValueError):
                return JSONResponse(content=resp.json(),
                                    status_code=resp.status_code)
        return PlainTextResponse(content=resp.text, status_code=resp.status_code,
                                 media_type=content_type or "text/plain")

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--upstream", default="http://127.0.0.1:63450")
    args = p.parse_args()

    print("WARNING: proxy.py is deprecated. Session tracking is built into "
          "opencode natively — this extra hop is unnecessary latency.",
          file=sys.stderr)
    print(f"opencode trace proxy: http://{args.host}:{args.port} -> {args.upstream}")
    key_state = "set" if os.environ.get("OPENROUTER_API_KEY") else "NOT SET"
    print(f"frontier route: /openrouter/* -> {OPENROUTER_BASE} "
          f"(OPENROUTER_API_KEY {key_state})")
    print(f"traces: {LOG_DIR}")

    import uvicorn
    uvicorn.run(make_app(args.upstream), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
