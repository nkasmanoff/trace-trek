#!/usr/bin/env python3
"""Pull dataset from HF Hub and restore to train.py format.

Usage on the pod:
    export HF_TOKEN=...read-only-token...
    python pull_dataset.py [--repo nkasmanoff/local-code-sft] [--out .]
"""
import argparse
import json
from pathlib import Path
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="nkasmanoff/local-code-sft-json")
    p.add_argument("--out", type=Path, default=Path("."))
    args = p.parse_args()

    print(f"Loading {args.repo}...")
    # Use data_files to explicitly load JSONL, avoids parquet cache issues
    ds = load_dataset(
        args.repo,
        data_files={"train": "train.jsonl", "eval": "eval.jsonl"},
        # streaming=True avoids local cache write issues on some datasets versions
        # but we need lengths, so materialize after streaming if needed
    )
    # If load_dataset returns a DatasetDict with streaming, materialize
    if hasattr(ds["train"], "__len__"):
        print(f"  train: {len(ds['train'])} samples")
        print(f"  eval:  {len(ds['eval'])} samples")
    else:
        # Streaming - materialize to list
        ds = {k: list(v) for k, v in ds.items()}
        print(f"  train: {len(ds['train'])} samples")
        print(f"  eval:  {len(ds['eval'])} samples")

    for split in ("train", "eval"):
        out_path = args.out / f"sft{'eval' if split == 'eval' else ''}.jsonl"
        with out_path.open("w") as f:
            for row in ds[split]:
                if row.get("tools"):
                    row["tools"] = json.loads(row["tools"])
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {out_path}")

    # Quick sanity check: verify a sample renders with the model's tokenizer
    print("\nSanity check: rendering one sample with the model's tokenizer...")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("local-code-model")
        sample = ds["train"][0]
        if sample.get("tools"):
            sample["tools"] = json.loads(sample["tools"])
        messages = []
        for m in sample["messages"]:
            m = dict(m)
            if m.get("reasoning_content"):
                m["thinking"] = m.pop("reasoning_content")
            messages.append(m)
        rendered = tok.apply_chat_template(
            messages, tools=sample.get("tools"), tokenize=False, add_generation_prompt=False
        )
        n_toks = len(tok(rendered, add_special_tokens=False).input_ids)
        print(f"  tokens: {n_toks} (max-seq-len default: 65536)")
        if "PREFLIGHT_THINK" not in rendered and "thinking" in str(sample["messages"]):
            print("  WARNING: reasoning/thinking may not render in template")
        if "bash" in rendered or "calc" in rendered:
            print("  tool calls render OK")
        print("  Sanity check passed.")
    except Exception as e:
        print(f"  Sanity check skipped/failed: {e}")


if __name__ == "__main__":
    main()