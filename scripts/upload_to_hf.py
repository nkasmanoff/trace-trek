#!/usr/bin/env python3
"""Upload an SFT JSONL (read from stdin) to a PRIVATE Hugging Face dataset repo.

Used by the viewer app's "Upload to HF" export path. The dataset content
is piped in on stdin so we never stage a temp file the size of the dataset.

Token resolution order:
  1. $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN
  2. HF_TOKEN=... in self-improve/.env (sibling of this repo)
  3. a cached `hf auth login` token

Usage:
  cat train.jsonl | python upload_to_hf.py --repo opencode-sft [--path-in-repo train.jsonl]

On success prints a JSON line: {"ok": true, "repo_id": "...", "url": "...", "bytes": N}
On failure prints {"ok": false, "error": "..."} and exits non-zero.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path


def load_env_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    # fall back to .env at repo root
    here = Path(__file__).resolve()
    for base in here.parents:
        env = base / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip("'\"")
            break
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True,
                   help="dataset repo id; bare name is placed under your namespace")
    p.add_argument("--path-in-repo", default="train.jsonl",
                   help="destination filename inside the dataset repo")
    p.add_argument("--input-file",
                   help="read dataset from this file instead of stdin")
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--public", dest="private", action="store_false")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(json.dumps({"ok": False, "error": "huggingface_hub not installed"}))
        return 1

    token = load_env_token()
    if not token:
        print(json.dumps({"ok": False, "error": "no HF token (set HF_TOKEN or self-improve/.env)"}))
        return 1

    data = Path(args.input_file).read_bytes() if args.input_file else sys.stdin.buffer.read()
    if not data.strip():
        print(json.dumps({"ok": False, "error": "empty dataset (nothing to upload)"}))
        return 1

    api = HfApi(token=token)
    try:
        who = api.whoami()
        namespace = who.get("name")
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"auth failed: {e}"}))
        return 1

    repo_id = args.repo if "/" in args.repo else f"{namespace}/{args.repo}"

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset",
                        private=args.private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=args.path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update {args.path_in_repo} from trace-anatomy export",
        )
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"upload failed: {e}"}))
        return 1

    n_lines = data.count(b"\n")
    print(json.dumps({
        "ok": True,
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "bytes": len(data),
        "records": n_lines,
        "path": args.path_in_repo,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
