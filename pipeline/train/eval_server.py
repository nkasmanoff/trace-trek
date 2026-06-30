#!/usr/bin/env python3
"""A tiny OpenAI-compatible server that wraps an in-memory (in-training) model.

Why this exists: `eval/run_evals.py` evaluates a *served* model over HTTP — it
does not take a model object. To run that gate against the model *while it is
training* (no second GPU, no vLLM), we expose the live weights at
`/v1/chat/completions` from inside the training process.

The one non-trivial bit is the model's native output format. Models do not emit
OpenAI `tool_calls`; they emit native XML-ish text. Laguna:

    <think> reasoning... </think>
    <tool_call>calc
    <arg_key>expr</arg_key>
    <arg_value>2+2</arg_value>
    </tool_call>

Qwen3.6 (qwen3_coder format):

    <think> reasoning... </think>
    <tool_call>
    <function=calc>
    <parameter=expr>
    2+2
    </parameter>
    </tool_call>

`parse_generation()` converts either into OpenAI-style `content` +
`reasoning_content` + `tool_calls` (arguments as a JSON string), which is what
the matching vLLM tool-call parser does at deploy time. This is eval-only and
intentionally minimal (greedy, non-streaming, serialized).
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- Laguna native format ---
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ARG_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
    re.DOTALL,
)
# --- Qwen3.6 (qwen3_coder) native format ---
_QWEN_FUNC_RE = re.compile(r"<function=([^>\n]+)>")
_QWEN_PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
# wrappers/markers stripped from the user-facing `content`
_STRIP = ("<assistant>", "</assistant>", "<think>", "</think>",
          "<|im_start|>", "<|im_end|>", "\u3008|EOS|\u3009")


def _split_reasoning(text: str) -> tuple[str | None, str]:
    """Split `<think>…</think>` reasoning from the rest. Shared by all formats."""
    end = text.find("</think>")
    if end != -1:
        return text[:end].replace("<think>", "").strip() or None, \
            text[end + len("</think>"):]
    return None, text


def parse_generation(text: str) -> tuple[str, str | None, list[dict]]:
    """(content, reasoning_content, tool_calls) from raw Laguna generation."""
    reasoning, rest = _split_reasoning(text)

    tool_calls = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(rest)):
        body = m.group(1)
        name = body.split("\n", 1)[0].strip()
        args = {k.strip(): v.strip() for k, v in _ARG_RE.findall(body)}
        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })

    content = _TOOL_CALL_RE.sub("", rest)
    for tok in _STRIP:
        content = content.replace(tok, "")
    return content.strip(), reasoning, tool_calls


def parse_generation_qwen(text: str) -> tuple[str, str | None, list[dict]]:
    """(content, reasoning_content, tool_calls) from raw Qwen3.6 generation."""
    reasoning, rest = _split_reasoning(text)

    tool_calls = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(rest)):
        body = m.group(1)
        fn = _QWEN_FUNC_RE.search(body)
        if not fn:
            continue
        name = fn.group(1).strip()
        args = {k.strip(): v.strip() for k, v in _QWEN_PARAM_RE.findall(body)}
        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })

    content = _TOOL_CALL_RE.sub("", rest)
    for tok in _STRIP:
        content = content.replace(tok, "")
    return content.strip(), reasoning, tool_calls


_PARSERS = {"laguna": parse_generation, "cohere": parse_generation,
            "qwen": parse_generation_qwen}


class InProcessModelServer:
    """Serve `/v1/chat/completions` (+ `/v1/models`, `/health`) from a live HF
    model. Generation is serialized behind a lock and runs greedily under
    `no_grad`; the caller is responsible for putting the model in eval mode."""

    def __init__(self, model, tokenizer, *, host: str = "127.0.0.1",
                 port: int = 8848, max_new_tokens: int = 2048,
                 enable_thinking: bool = True, chat_format: str = "laguna",
                 served_model: str = "local-code-model"):
        self.model = model
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.chat_format = chat_format
        self._parse = _PARSERS.get(chat_format, parse_generation)
        self.served_model = served_model
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _device(self):
        return self.model.get_input_embeddings().weight.device

    @staticmethod
    def _normalize_messages(messages: list[dict]) -> list[dict]:
        """Parse assistant tool_call `arguments` from JSON strings back to dicts.
        OpenAI clients (opencode's `@ai-sdk/openai-compatible`) send prior-turn
        tool calls with `arguments` as a JSON *string*, but the Qwen chat
        template iterates `arguments.items()` and raises
        `TypeError: Can only get item pairs from a mapping.` on a string — which,
        on the streaming path, hangs the client until timeout. Mirrors
        train.py's `normalize_tool_args`."""
        out = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                m = dict(m)
                tcs = []
                for tc in m["tool_calls"]:
                    tc = dict(tc)
                    fn = dict(tc.get("function", {}))
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args) if args.strip() else {}
                        except (json.JSONDecodeError, ValueError):
                            fn["arguments"] = {}
                    tc["function"] = fn
                    tcs.append(tc)
                m["tool_calls"] = tcs
            out.append(m)
        return out

    def _infer(self, messages: list[dict], tools: list | None,
               max_tokens: int | None) -> tuple[str, str | None, list[dict], int, int]:
        """Run the model once and parse to (content, reasoning, tool_calls,
        n_in, n_out). Shared by the streaming and non-streaming responders."""
        import torch

        messages = self._normalize_messages(messages)
        enc = self.tokenizer.apply_chat_template(
            messages, tools=tools or None, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=self.enable_thinking,
        )
        dev = self._device()
        input_ids = enc["input_ids"].to(dev)
        attn = enc.get("attention_mask")
        attn = attn.to(dev) if attn is not None else None
        n_in = input_ids.shape[1]
        budget = min(max_tokens or self.max_new_tokens, self.max_new_tokens)
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos

        with self._lock, torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=budget,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad,
                eos_token_id=eos,
            )
        gen = out[0][n_in:]
        raw = self.tokenizer.decode(gen, skip_special_tokens=False)
        content, reasoning, tool_calls = self._parse(raw)
        return content, reasoning, tool_calls, int(n_in), int(gen.shape[0])

    def generate(self, messages: list[dict], tools: list | None,
                 max_tokens: int | None) -> dict:
        content, reasoning, tool_calls, n_in, n_out = self._infer(
            messages, tools, max_tokens)
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model,
            "choices": [{
                "index": 0,
                "message": msg,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {"prompt_tokens": n_in,
                      "completion_tokens": n_out,
                      "total_tokens": n_in + n_out},
        }

    def stream(self, messages: list[dict], tools: list | None,
               max_tokens: int | None):
        """Yield OpenAI-style chat.completion.chunk SSE events. opencode's
        `@ai-sdk/openai-compatible` provider always requests `stream: true` and
        parses the body as `text/event-stream`; a non-streaming JSON body yields
        an empty turn on its side. Generation is non-incremental here, so we run
        the model fully and then emit the result as a few well-formed deltas."""
        content, reasoning, tool_calls, n_in, n_out = self._infer(
            messages, tools, max_tokens)
        cid = f"chatcmpl-{int(time.time()*1000)}"
        created = int(time.time())

        def event(delta: dict, finish=None, extra: dict | None = None) -> str:
            payload = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": self.served_model,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}],
            }
            if extra:
                payload.update(extra)
            return "data: " + json.dumps(payload) + "\n\n"

        yield event({"role": "assistant"})
        if reasoning:
            yield event({"reasoning_content": reasoning})
        if content:
            yield event({"content": content})
        if tool_calls:
            tc_delta = [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]},
            } for i, tc in enumerate(tool_calls)]
            yield event({"tool_calls": tc_delta})
        yield event({}, finish="tool_calls" if tool_calls else "stop",
                    extra={"usage": {"prompt_tokens": n_in,
                                     "completion_tokens": n_out,
                                     "total_tokens": n_in + n_out}})
        yield "data: [DONE]\n\n"

    def start(self) -> str:
        if self._httpd is not None:
            return self.base_url
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence per-request logging
                pass

            def _send(self, code: int, payload: dict):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/health"):
                    self._send(200, {"status": "ok"})
                elif self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"object": "list", "data": [
                        {"id": server.served_model, "object": "model"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                try:
                    req = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError as exc:
                    self._send(400, {"error": f"bad json: {exc}"})
                    return
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._send(404, {"error": "not found"})
                    return
                try:
                    if req.get("stream"):
                        # SSE: opencode's openai-compatible provider parses the
                        # body as text/event-stream; a plain JSON body there
                        # produces an empty assistant turn. We have no
                        # Content-Length and don't chunk-encode, so the client
                        # can only detect end-of-body via connection close —
                        # `Connection: close` + close_connection avoids a
                        # keep-alive deadlock that hangs the client after the
                        # final event is written.
                        self.close_connection = True
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        for ev in server.stream(
                                req.get("messages", []), req.get("tools"),
                                req.get("max_tokens")):
                            self.wfile.write(ev.encode())
                            self.wfile.flush()
                    else:
                        resp = server.generate(
                            req.get("messages", []), req.get("tools"),
                            req.get("max_tokens"))
                        self._send(200, resp)
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    self._send(500, {"error": repr(exc)})

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
