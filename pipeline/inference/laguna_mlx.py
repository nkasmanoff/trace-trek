"""
Laguna XS.2 on MLX (local, Apple Silicon).

Architecture
------------
   OpenCode  --->  [ mlx_lm.server (OpenAI-compatible, localhost:63450) ]  --->  Laguna (MLX)

This is the local-Mac counterpart to ``inference/laguna_modal.py`` (which serves
the FP8 checkpoint with vLLM on an H100). Here we serve an ``mlx-community`` MLX
build natively on Apple Silicon via ``mlx_lm.server``.

``mlx_lm.server`` auto-selects the Laguna tool-call parser from the model's
``model_type`` ("laguna"), so it converts the model's native
``<tool_call>...</tool_call>`` text into OpenAI-style structured ``tool_calls``
and splits ``<think>...</think>`` into a separate ``reasoning`` field -- exactly
what OpenCode expects.

Requirements
------------
  # Laguna support is not in a released mlx-lm yet; install from the PR branch:
  pip install "git+https://github.com/Blaizzy/mlx-lm.git@pc/add-lg"

Run
---
  python inference/laguna_mlx.py                     # 4-bit on :63450
  python inference/laguna_mlx.py --model mlx-community/Laguna-XS.2-8bit
  python inference/laguna_mlx.py --port 63450 --no-thinking
"""

import argparse
import json
import os
import subprocess
import sys

MODEL = "mlx-community/Laguna-XS.2-4bit"  # swap for -6bit / -8bit for higher quality
HOST = "127.0.0.1"
PORT = 63450  # high dynamic port — unlikely to clash
MAX_TOKENS = 65536


def download_model(model: str) -> None:
    from huggingface_hub import snapshot_download

    print(f"Ensuring {model} is downloaded ...", flush=True)
    snapshot_download(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help="MLX model repo or local path")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable interleaved reasoning (enable_thinking=false).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Don't pre-fetch weights (server will fetch lazily on first load).",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_model(args.model)

    chat_template_args = {"enable_thinking": not args.no_thinking}

    cmd = [
        sys.executable, "-m", "mlx_lm", "server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-tokens", str(args.max_tokens),
        "--chat-template-args", json.dumps(chat_template_args),
        "--log-level", "INFO",
    ]

    print("Launching MLX server:\n  " + " ".join(cmd), flush=True)
    print(
        f"\nOpenAI-compatible endpoint: http://{args.host}:{args.port}/v1"
        f"\nServed model id: {args.model}\n",
        flush=True,
    )

    try:
        return subprocess.run(cmd, env=os.environ.copy()).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
