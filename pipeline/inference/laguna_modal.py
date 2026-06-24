"""
Qwen3.6-35B-A3B on Modal.

Architecture
------------
  OpenCode  --->  [ FastAPI proxy on Modal's public port ]  --->  vLLM (localhost:8001)

Deploy:  modal deploy inference/laguna_modal.py
"""

# Run locally:  python -m modal serve inference/laguna_modal.py

import json
import subprocess
import time
import urllib.request

import modal

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
SERVED_NAME = "qwen3.6-35b-a3b"
GPU = "H100"
MAX_MODEL_LEN = 65536
SCALEDOWN_WINDOW = 30
FAST_BOOT = True
VLLM_PORT = 8001
MINUTES = 60

def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_NAME)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]<0.137",
        "httpx",
    )
    .run_function(
        download_model,
        secrets=[modal.Secret.from_name("huggingface-secret")],
        timeout=30 * MINUTES,
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen3.6-35b-a3b-vllm")


@app.cls(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
@modal.concurrent(max_inputs=32)
class QwenServer:
    @modal.enter()
    def start_engine(self):
        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--served-model-name", SERVED_NAME,
            "--host", "127.0.0.1",
            "--port", str(VLLM_PORT),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--tensor-parallel-size", "1",
            "--reasoning-parser", "qwen3",
            "--tool-call-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--default-chat-template-kwargs", json.dumps({"enable_thinking": True}),
            "--uvicorn-log-level=warning",
        ]
        cmd += ["--enforce-eager"] if FAST_BOOT else ["--no-enforce-eager"]
        print("Launching vLLM:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd)

        health = f"http://127.0.0.1:{VLLM_PORT}/health"
        for _ in range(10 * MINUTES):
            try:
                if urllib.request.urlopen(health, timeout=2).status == 200:
                    print("vLLM is up.")
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("vLLM did not become healthy in time")

    @modal.asgi_app()
    def proxy(self):
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response

        web = FastAPI()
        upstream = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{VLLM_PORT}", timeout=httpx.Timeout(None)
        )

        @web.get("/health")
        async def health():
            r = await upstream.get("/health")
            return Response(status_code=r.status_code)

        @web.middleware("http")
        async def require_api_key(request: Request, call_next):
            import os
            expected_key = os.environ.get("API_KEY", "")
            if expected_key:
                actual_key = request.headers.get("authorization", "")
                if not actual_key.startswith("Bearer ") or actual_key[7:] != expected_key:
                    return JSONResponse(status_code=401, content={"error": "unauthorized"})
            return await call_next(request)

        @web.api_route("/{path:path}", methods=["GET", "POST"])
        async def relay(path: str, request: Request):
            body = await request.body()
            fwd = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length")}
            is_stream = b'"stream": true' in body

            if is_stream:
                from fastapi.responses import StreamingResponse

                async def stream():
                    async with upstream.stream(
                        request.method, "/" + path, content=body, headers=fwd
                    ) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk

                return StreamingResponse(stream(), media_type="text/event-stream")

            r = await upstream.request(
                request.method, "/" + path, content=body, headers=fwd
            )
            return Response(
                content=r.content,
                status_code=r.status_code,
                headers={"content-type": r.headers.get("content-type", "application/json")},
            )

        return web

    @modal.exit()
    def shutdown(self):
        if getattr(self, "proc", None):
            self.proc.terminate()
