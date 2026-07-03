"""Serve a model over the in-process OpenAI-compatible server and run the FULL
agent-problem-pack against it. Used post-training to measure the start (base) vs
latest (merged) pack pass rate on the complete benchmark.

Usage:
    python run_full_bench.py --model-path Qwen/Qwen3.6-27B --tag baseline
    python run_full_bench.py --model-path ../outputs/merged --tag lora-merged
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True, help="label, e.g. start|latest")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--port", type=int, default=8850)
    args = ap.parse_args()

    from train import load_dotenv
    load_dotenv(HERE)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eval_server import InProcessModelServer

    # same cuDNN-SDPA workaround the training run needed on Qwen3.6 full-attn
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:  # noqa: BLE001
        pass

    print(f"[{args.tag}] loading {args.model_path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    # MoE-only; dense Qwen3.6-27B rejects this kwarg.
    if any(x in args.model_path.lower() for x in ("a3b", "moe", "35b-a3b")):
        load_kwargs["experts_implementation"] = "grouped_mm"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    # router logits during cached generation crash on shape mismatch; off for eval
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False

    srv = InProcessModelServer(
        model, tok, port=args.port, max_new_tokens=args.max_new_tokens,
        chat_format="qwen", served_model="local-code-model")
    url = srv.start()
    print(f"[{args.tag}] server up at {url}", flush=True)

    eval_dir = HERE.parent / "eval"
    runner = eval_dir / "run_problem_pack.py"
    home = HERE.parent / "outputs" / f"fullbench-home-{args.tag}"
    cmd = [sys.executable, str(runner),
           "--base-url", url,
           "--served-model", "local-code-model",
           "--model", "local/local-code-model",
           "--subset", args.subset,
           "--out", str(eval_dir),
           "--opencode-home", str(home),
           "--timeout", str(args.timeout)]
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.opencode/bin") + os.pathsep + env.get("PATH", "")
    before = {p.name for p in eval_dir.glob("pack-results-*.json")}
    print(f"[{args.tag}] running pack: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, env=env, check=False)
    print(f"[{args.tag}] pack wall time {time.time()-t0:.0f}s", flush=True)

    new = sorted(p for p in eval_dir.glob("pack-results-*.json")
                 if p.name not in before)
    if not new:
        print(f"[{args.tag}] NO pack results produced")
        srv.stop()
        return
    res = json.loads(new[-1].read_text())
    pack = res.get("problem_pack") or {}
    tagged = eval_dir / f"fullbench-{args.tag}.json"
    tagged.write_text(json.dumps(res, indent=2))
    total = pack.get("total", 0)
    passed = pack.get("passed", 0)
    rate = passed / total if total else 0.0
    print(f"[{args.tag}] RESULT pass {passed}/{total} ({rate:.1%}) -> {tagged}",
          flush=True)
    srv.stop()


if __name__ == "__main__":
    main()
