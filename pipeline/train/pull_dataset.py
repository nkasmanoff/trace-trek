#!/usr/bin/env python3
"""Pull an SFT dataset from HF Hub and split it into train/eval JSONL files.

`nkasmanoff/opencode-sft` ships a single `train` split (~500 opencode
sessions), so we deterministically hold out --eval-frac of it for evaluation.
The eval split is also what eval/harvest_eval_tasks.py turns into held-out
opencode-harness tasks, so the same rows are never trained on and evaluated.

Usage on the pod:
    export HF_TOKEN=...read-only-token...   # only needed for private repos
    python train/pull_dataset.py [--repo nkasmanoff/opencode-sft] \
        [--out dataset] [--eval-frac 0.1]
"""
import argparse
import copy
import json
from pathlib import Path

from datasets import load_dataset

LAGUNA_BASE = "poolside/Laguna-XS.2"


def normalize_tool_args(messages):
    """The dataset stores tool_call arguments as JSON strings, but the chat
    template expects dicts. Parse them back so the template renders."""
    msgs = copy.deepcopy(messages)
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    tc["function"]["arguments"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
    return msgs


def write_split(split, out_path: Path) -> int:
    with out_path.open("w") as f:
        for row in split:
            row = dict(row)
            if isinstance(row.get("tools"), str):
                try:
                    row["tools"] = json.loads(row["tools"])
                except (json.JSONDecodeError, TypeError):
                    pass
            f.write(json.dumps(row) + "\n")
    return len(split)


def sanity_check(sample: dict, base: str) -> None:
    """Render one sample with the model's tokenizer to confirm the chat
    template emits reasoning + tool calls and an assistant-token mask."""
    print(f"\nSanity check: rendering one sample with {base}'s tokenizer...")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base)
        messages = normalize_tool_args(sample["messages"])
        tools = sample.get("tools") or None
        text = tok.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=False, enable_thinking=True,
        )
        n_toks = len(tok(text, add_special_tokens=False).input_ids)
        enc = tok.apply_chat_template(
            messages, tools=tools, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True, add_generation_prompt=False,
            enable_thinking=True,
        )
        n_asst = sum(enc.get("assistant_masks") or [])
        print(f"  tokens: {n_toks}  assistant-masked: {n_asst}")
        print(f"  reasoning renders: {'<think>' in text}  "
              f"tool calls render: {'<tool_call>' in text}")
        if n_asst == 0:
            print("  WARNING: no assistant tokens masked — check the template")
        else:
            print("  Sanity check passed.")
    except Exception as e:  # noqa: BLE001
        print(f"  Sanity check skipped/failed: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="nkasmanoff/opencode-sft")
    p.add_argument("--out", type=Path, default=Path("dataset"))
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--base", default=LAGUNA_BASE,
                   help="model id used for the rendering sanity check")
    args = p.parse_args()

    print(f"Loading {args.repo}...")
    ds = load_dataset(args.repo)
    eval_key = next((k for k in ("eval", "test", "validation") if k in ds), None)
    if eval_key:
        train_split, eval_split = ds["train"], ds[eval_key]
        print(f"  using existing splits: train + {eval_key}")
    else:
        parts = ds["train"].train_test_split(test_size=args.eval_frac, seed=0)
        train_split, eval_split = parts["train"], parts["test"]
        print(f"  split single train split with eval_frac={args.eval_frac}")

    args.out.mkdir(parents=True, exist_ok=True)
    n_train = write_split(train_split, args.out / "sft.jsonl")
    n_eval = write_split(eval_split, args.out / "sft-eval.jsonl")
    print(f"  train: {n_train} -> {args.out / 'sft.jsonl'}")
    print(f"  eval:  {n_eval} -> {args.out / 'sft-eval.jsonl'}")

    sanity_check(dict(train_split[0]), args.base)


if __name__ == "__main__":
    main()
