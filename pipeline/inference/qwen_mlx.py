"""

Use the conda environment torch_env before running.

Qwen3.6-35B-A3B on MLX (local, Apple Silicon).

Architecture
------------
   OpenCode  --->  [ mlx_lm.server (OpenAI-compatible, localhost:63451) ]  --->  Qwen3.6 (MLX)

This is the Qwen counterpart to ``inference/laguna_mlx.py``. It serves the
``mlx-community/Qwen3.6-35B-A3B-4bit`` MoE build natively on Apple Silicon via
``mlx_lm.server`` and runs on a different port (63451) so it can coexist with the
Laguna server (63450).

``mlx_lm.server`` auto-selects the qwen3 tool-call parser from the model's
``model_type`` ("qwen3_5_moe"), converting native ``<tool_call>...</tool_call>``
text into OpenAI-style structured ``tool_calls`` and splitting
``<think>...</think>`` into a separate ``reasoning`` field -- exactly what
OpenCode expects.

Requirements
------------
  pip install -U mlx-lm

Run
---
  python inference/qwen_mlx.py                          # 4-bit on :63451
  python inference/qwen_mlx.py --model mlx-community/Qwen3.6-35B-A3B-8bit
  python inference/qwen_mlx.py --port 63451 --no-thinking
"""

import argparse
import json
import os
import subprocess
import sys

MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"  # swap for -6bit / -8bit for higher quality
HOST = "127.0.0.1"
PORT = 63451  # one above the Laguna server so both can run at once
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
    import os as _os
    _os._exit(main())
