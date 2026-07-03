#!/usr/bin/env python3
"""Pull an SFT dataset from HF Hub, CLEAN it, and split into train/eval JSONL.

`nkasmanoff/opencode-sft` ships a single `train` split (~500 opencode
sessions), so we deterministically hold out --eval-frac of it for evaluation.
The eval split is also what eval/harvest_eval_tasks.py turns into held-out
opencode-harness tasks, so the same rows are never trained on and evaluated.

Every pull is scrubbed through build_dataset.py's full cleaning pipeline so a
stale or dirty HF snapshot can never poison a training run:

  1. apply_filters — drop corrections, malformed/looping tools, incomplete
     exports, empty subagent results, trivial zero-tool readiness replies, and
     over-length samples
  2. sanitize — rewrite teacher model identity + replay worktree paths
  3. dedupe — collapse prefix/duplicate sessions to the longest trajectory
  4. decontaminate — drop rows overlapping the held-out agent-problem-pack
  5. check_contamination marker gate (when pack scripts are present)

Usage on the pod:
    export HF_TOKEN=...read-only-token...   # only needed for private repos
    python train/pull_dataset.py [--repo nkasmanoff/opencode-sft] \
        [--out dataset] [--eval-frac 0.1] [--model-name Qwen/Qwen3.6-35B-A3B]
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

QWEN_BASE = "Qwen/Qwen3.6-35B-A3B"

TRAIN_FILES = ("sft.jsonl", "train.jsonl", "data/train.jsonl")
EVAL_FILES = ("sft-eval.jsonl", "eval.jsonl", "test.jsonl", "validation.jsonl")


def _build_dataset_module():
    """Import pipeline/dataset/build_dataset.py (decontaminate + sanitize)."""
    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    if str(dataset_dir) not in sys.path:
        sys.path.insert(0, str(dataset_dir))
    import build_dataset
    return build_dataset


def clean_rows(rows: list[dict], model_name: str, workspace: str = "/workspace",
               pack_root: Path | None = None, decontam: bool = True,
               decontam_ngram: int = 13, do_sanitize: bool = True,
               max_tokens: int = 64000) -> list[dict]:
    """Run the same dedup/filter/decontam/sanitize path as build_dataset.py."""
    bd = _build_dataset_module()
    return bd.clean_sft_records(
        rows,
        max_tokens=max_tokens,
        model_name=model_name,
        workspace=workspace,
        pack_root=pack_root or bd.DEFAULT_PACK_ROOT,
        decontam=decontam,
        decontam_ngram=decontam_ngram,
        do_sanitize=do_sanitize,
    )


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


def read_jsonl(path: Path) -> list[dict]:
    """Read raw JSONL rows with tools normalized to lists. We fetch the raw
    file instead of using `datasets.load_dataset` because the rows have
    heterogeneous message schemas that Arrow refuses to cast (and would
    None-fill even when it succeeds)."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row.get("tools"), str):
                try:
                    row["tools"] = json.loads(row["tools"])
                except (json.JSONDecodeError, TypeError):
                    pass
            rows.append(row)
    return rows


def fetch_repo_rows(repo: str) -> tuple[list[dict], list[dict] | None]:
    """Download the dataset's raw JSONL from HF. Returns (train, eval-or-None)."""
    names = set(list_repo_files(repo, repo_type="dataset"))
    train_name = next((n for n in TRAIN_FILES if n in names), None)
    if train_name is None:
        jsonls = sorted(n for n in names if n.endswith(".jsonl"))
        if not jsonls:
            raise SystemExit(f"no .jsonl files found in {repo}: {sorted(names)}")
        train_name = jsonls[0]
    train_rows = read_jsonl(Path(hf_hub_download(
        repo, train_name, repo_type="dataset")))
    eval_name = next((n for n in EVAL_FILES if n in names), None)
    eval_rows = None
    if eval_name:
        eval_rows = read_jsonl(Path(hf_hub_download(
            repo, eval_name, repo_type="dataset")))
    print(f"  pulled {train_name} ({len(train_rows)} rows)"
          + (f" + {eval_name} ({len(eval_rows)} rows)" if eval_name else ""))
    return train_rows, eval_rows


def write_split(rows: list[dict], out_path: Path) -> int:
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def pull_split_clean(repo: str, out: Path, eval_frac: float, model_name: str,
                     workspace: str = "/workspace",
                     pack_root: Path | None = None, decontam: bool = True,
                     decontam_ngram: int = 13, do_sanitize: bool = True,
                     max_tokens: int = 64000,
                     ) -> tuple[Path, Path]:
    """Pull `repo`, clean it, and write out/sft.jsonl + out/sft-eval.jsonl.
    Also used by train.py --hf-repo so both download paths stay clean."""
    print(f"Loading {repo}...")
    train_rows, eval_rows = fetch_repo_rows(repo)
    if eval_rows is not None:
        print("  using existing train + eval splits")
        train_rows = clean_rows(train_rows, model_name, workspace,
                                pack_root, decontam, decontam_ngram, do_sanitize,
                                max_tokens)
        eval_rows = clean_rows(eval_rows, model_name, workspace,
                               pack_root, decontam, decontam_ngram, do_sanitize,
                               max_tokens)
    else:
        # Clean BEFORE splitting so decontam-dropped rows can't skew the split.
        rows = clean_rows(train_rows, model_name, workspace,
                          pack_root, decontam, decontam_ngram, do_sanitize,
                          max_tokens)
        indices = list(range(len(rows)))
        random.Random(0).shuffle(indices)
        n_eval = max(1, round(len(rows) * eval_frac)) if rows else 0
        eval_idx = set(indices[:n_eval])
        train_rows = [r for i, r in enumerate(rows) if i not in eval_idx]
        eval_rows = [r for i, r in enumerate(rows) if i in eval_idx]
        print(f"  split single train file with eval_frac={eval_frac} (seed 0)")

    out.mkdir(parents=True, exist_ok=True)
    data_path, eval_path = out / "sft.jsonl", out / "sft-eval.jsonl"
    print(f"  train: {write_split(train_rows, data_path)} -> {data_path}")
    print(f"  eval:  {write_split(eval_rows, eval_path)} -> {eval_path}")
    return data_path, eval_path


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
            # Qwen-style templates ship no {% generation %} blocks; train.py
            # falls back to marker-based masking for those, so this is only a
            # hard failure for templates that are supposed to emit the mask.
            print("  note: template emits no assistant mask "
                  "(train.py uses marker-based masking for such templates)")
        else:
            print("  Sanity check passed.")
    except Exception as e:  # noqa: BLE001
        print(f"  Sanity check skipped/failed: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="nkasmanoff/opencode-sft")
    p.add_argument("--out", type=Path, default=Path("dataset"))
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=64000,
                   help="approximate per-sample cap (chars/4); set below "
                        "train.py --max-seq-len")
    p.add_argument("--base", default=QWEN_BASE,
                   help="model id used for the rendering sanity check")
    p.add_argument("--model-name", default=None,
                   help="rewrite the teacher model identity in system prompts "
                        "to this id (default: --base)")
    p.add_argument("--workspace-path", default="/workspace",
                   help="stable workspace root substituted for replay "
                        "worktree paths during sanitize")
    p.add_argument("--no-sanitize", action="store_true",
                   help="keep teacher identities / worktree paths as pulled")
    p.add_argument("--pack-root", type=Path, default=None,
                   help="agent-problem-pack root for decontamination "
                        "(default: <repo>/agent-problem-pack)")
    p.add_argument("--no-decontam", action="store_true",
                   help="skip benchmark decontamination (NOT recommended)")
    p.add_argument("--decontam-ngram", type=int, default=13)
    args = p.parse_args()

    data_path, _ = pull_split_clean(
        args.repo, args.out, args.eval_frac,
        model_name=args.model_name or args.base,
        workspace=args.workspace_path,
        pack_root=args.pack_root,
        decontam=not args.no_decontam,
        decontam_ngram=args.decontam_ngram,
        do_sanitize=not args.no_sanitize,
        max_tokens=args.max_tokens,
    )

    first = data_path.read_text().splitlines()[0]
    sanity_check(json.loads(first), args.base)


if __name__ == "__main__":
    main()
