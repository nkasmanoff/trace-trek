#!/usr/bin/env python3
"""Build or filter an SFT dataset from collected traces or existing JSONL.

When ``--sft-input`` is given, reads already-built SFT records (one JSON object
per line), applies the quality gate + optional sanitization, and decontaminates
against the local ``agent-problem-pack/`` benchmark via 13-gram overlap.

Legacy path (``--opencode`` / ``--cursor``) reconstructs trajectories from raw
proxy logs — see the module docstring in agent-improver for details.

Usage (decontaminate a pulled HF split):
    python dataset/build_dataset.py --sft-input dataset/sft.jsonl --stdout \\
        > dataset/sft.clean.jsonl 2> dataset/decontam-train.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = REPO_ROOT / "agent-problem-pack"
DEFAULT_PACK_ROOT = PACK_ROOT
NGRAM_SIZE = 13

CORRECTION_RE = re.compile(
    r"\b(no[,.!]|that'?s (wrong|not right|incorrect)|doesn'?t work|not what i"
    r"|undo|revert|stop|wtf|broken|you broke|still (wrong|broken|failing))\b",
    re.IGNORECASE,
)

WORKTREE_RE = re.compile(
    r"(?:/private)?/(?:var/folders/[^/\s\"']+/[^/\s\"']+/T|tmp)/"
    r"improver-replay-[A-Za-z0-9_]+"
)
WORKTREE_NAME_RE = re.compile(r"\bimprover-replay-[A-Za-z0-9_]+")
POWERED_BY_RE = re.compile(
    r"You are powered by the model named (?P<named>.+?)\. "
    r"The exact model ID is (?P<id>\S+)"
)
USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".jsx", ".js", ".ts", ".tsx",
    ".toml", ".yaml", ".yml", ".sh", ".css", ".html", ".jinja", ".lock",
}


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return ""


def approx_tokens(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False)) // 4


def session_key(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        if m.get("role") == "system":
            parts.append(content_to_text(m.get("content")))
            break
    for m in messages:
        if m.get("role") == "user":
            parts.append(content_to_text(m.get("content")))
            break
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def has_malformed_tool_calls(messages: list[dict]) -> bool:
    for m in messages:
        for tc in m.get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments")
            if args is None:
                return True
            if isinstance(args, str):
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    return True
    return False


def has_tool_loop(messages: list[dict], threshold: int = 3) -> bool:
    prev, streak = None, 0
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            sig = (fn.get("name"), fn.get("arguments"))
            if sig == prev:
                streak += 1
                if streak >= threshold:
                    return True
            else:
                prev, streak = sig, 1
    return False


def ends_in_correction(messages: list[dict]) -> bool:
    for m in reversed(messages):
        if m.get("role") == "user":
            text = content_to_text(m.get("content"))
            return bool(CORRECTION_RE.search(text))
    return False


def teacher_model_ids(messages: list[dict]) -> set[str]:
    ids: set[str] = set()
    for m in messages:
        c = m.get("content")
        if not isinstance(c, str):
            continue
        for mm in POWERED_BY_RE.finditer(c):
            ids.add(mm.group("named"))
            ids.add(mm.group("id"))
    return ids


def _rewrite_text(text: str, model_name: str, workspace: str,
                  teacher_ids: list[str]) -> str:
    text = WORKTREE_RE.sub(workspace, text)
    text = WORKTREE_NAME_RE.sub(workspace.rstrip("/").rsplit("/", 1)[-1], text)
    if model_name:
        text = POWERED_BY_RE.sub(
            f"You are powered by the model named {model_name}. "
            f"The exact model ID is {model_name}",
            text,
        )
        for tid in teacher_ids:
            text = text.replace(tid, model_name)
    return text


def _rewrite(obj: Any, model_name: str, workspace: str,
             teacher_ids: list[str]) -> Any:
    if isinstance(obj, str):
        return _rewrite_text(obj, model_name, workspace, teacher_ids)
    if isinstance(obj, list):
        return [_rewrite(x, model_name, workspace, teacher_ids) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite(v, model_name, workspace, teacher_ids)
                for k, v in obj.items()}
    return obj


def sanitize(samples: list[dict], model_name: str,
             workspace: str) -> list[dict]:
    n_paths = n_model = 0
    for s in samples:
        ids = sorted(teacher_model_ids(s["messages"]), key=len, reverse=True)
        blob = json.dumps(s["messages"], ensure_ascii=False)
        if WORKTREE_NAME_RE.search(blob):
            n_paths += 1
        if ids:
            n_model += 1
        s["messages"] = _rewrite(s["messages"], model_name, workspace, ids)
        if s.get("tools"):
            s["tools"] = _rewrite(s["tools"], model_name, workspace, ids)
    print(f"  sanitize: rewrote worktree paths in {n_paths} samples, "
          f"model identity in {n_model} samples "
          f"(workspace={workspace!r}, model={model_name!r})",
          file=sys.stderr)
    return samples


def apply_filters(samples: list[dict], max_tokens: int) -> list[dict]:
    kept, dropped = [], {"correction": 0, "malformed": 0, "loop": 0, "long": 0}
    for s in samples:
        msgs = s["messages"]
        if ends_in_correction(msgs):
            dropped["correction"] += 1
            continue
        if has_malformed_tool_calls(msgs):
            dropped["malformed"] += 1
            continue
        if has_tool_loop(msgs):
            dropped["loop"] += 1
            continue
        if approx_tokens(msgs) > max_tokens:
            dropped["long"] += 1
            continue
        kept.append(s)
    print(f"  filters: kept {len(kept)}, dropped {dropped}", file=sys.stderr)
    return kept


def dedupe_prefix(samples: list[dict]) -> list[dict]:
    """Collapse exact duplicates and prefix trajectories (keep longest)."""
    by_key: dict[str, dict] = {}
    for s in samples:
        key = session_key(s["messages"])
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = s
            continue
        if len(s["messages"]) > len(prev["messages"]):
            survivor = s
        elif len(s["messages"]) < len(prev["messages"]):
            survivor = prev
        elif s.get("_split") == "eval":
            survivor = s
        else:
            survivor = prev
        if survivor.get("_split") != "eval" and (
                s.get("_split") == "eval" or prev.get("_split") == "eval"):
            survivor = dict(survivor)
            survivor["_split"] = "eval"
        by_key[key] = survivor
    kept = list(by_key.values())
    print(f"  dedupe: {len(samples)} -> {len(kept)} sessions", file=sys.stderr)
    return kept


def _normalize_for_ngrams(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _iter_ngrams(text: str, n: int = NGRAM_SIZE) -> Iterable[str]:
    words = _normalize_for_ngrams(text).split()
    if len(words) < n:
        return
    for i in range(len(words) - n + 1):
        yield " ".join(words[i:i + n])


def _pack_text_files(pack_root: Path) -> list[Path]:
    skip_dirs = {"runs", ".git", "__pycache__", "node_modules", ".venv"}
    files: list[Path] = []
    for path in sorted(pack_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        files.append(path)
    return files


def build_pack_ngrams(pack_root: Path) -> set[str]:
    ngrams: set[str] = set()
    for path in _pack_text_files(pack_root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ngrams.update(_iter_ngrams(text))
    return ngrams


def record_text(rec: dict) -> str:
    parts = []
    for m in rec.get("messages") or []:
        parts.append(content_to_text(m.get("content")))
        if m.get("reasoning_content"):
            parts.append(str(m["reasoning_content"]))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            parts.append(str(fn.get("name", "")))
            parts.append(str(fn.get("arguments", "")))
    if rec.get("tools"):
        parts.append(json.dumps(rec["tools"], ensure_ascii=False))
    return "\n".join(parts)


def decontaminate(samples: list[dict], pack_root: Path | None = None) -> list[dict]:
    """Drop samples whose text contains a 13-gram from agent-problem-pack."""
    root = pack_root or PACK_ROOT
    if not root.is_dir():
        print("decontam: SKIPPED (pack not found)", file=sys.stderr)
        return samples

    pack_files = _pack_text_files(root)
    pack_ngrams = build_pack_ngrams(root)
    if not pack_ngrams:
        print(f"decontam: SKIPPED (no n-grams from {root})", file=sys.stderr)
        return samples

    kept, ngram_drops = [], 0
    for s in samples:
        text = record_text(s)
        if set(_iter_ngrams(text)) & pack_ngrams:
            ngram_drops += 1
            continue
        kept.append(s)

    print(f"decontam: kept {len(kept)}, dropped {{'ngram': {ngram_drops}}} "
          f"(pack={len(pack_files)} files, {len(pack_ngrams)} {NGRAM_SIZE}-grams)",
          file=sys.stderr)
    return kept


def load_sft_input(src: Path | None, stream: Iterable[str]) -> list[dict]:
    samples: list[dict] = []
    if src is not None and str(src) != "-":
        lines = src.read_text(encoding="utf-8").splitlines()
    else:
        lines = list(stream)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec.get("messages"), list):
            samples.append(rec)
    return samples


def write_samples(samples: list[dict], out: Path | None, stdout: bool) -> None:
    fh = sys.stdout if stdout else out.open("w", encoding="utf-8")
    try:
        for s in samples:
            clean = {k: v for k, v in s.items() if not str(k).startswith("_")}
            fh.write(json.dumps(clean, ensure_ascii=False) + "\n")
    finally:
        if not stdout and fh is not sys.stdout:
            fh.close()


def normalize_message(m: dict) -> dict:
    out: dict[str, Any] = {"role": m.get("role")}
    out["content"] = content_to_text(m.get("content"))
    if m.get("reasoning_content"):
        out["reasoning_content"] = m["reasoning_content"]
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    if m.get("name"):
        out["name"] = m["name"]
    return out


def load_opencode(raw_dir: Path,
                  exclude_ids: set[str] | None = None) -> list[dict]:
    by_session: dict[str, dict] = {}
    n_records = n_replay_fail = n_holdout = 0
    for path in sorted(raw_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        replay = rec.get("replay")
        if isinstance(replay, dict) and exclude_ids \
                and replay.get("task_id") in exclude_ids:
            n_holdout += 1
            continue
        if isinstance(replay, dict) and not replay.get("pass"):
            n_replay_fail += 1
            continue
        req = rec.get("request") or {}
        resp = rec.get("response") or {}
        messages = req.get("messages")
        message = resp.get("message")
        if not isinstance(messages, list) or not isinstance(message, dict):
            continue
        n_records += 1
        if rec.get("upstream") == "openrouter":
            source = "frontier"
        elif replay:
            source = "replay"
        else:
            source = "opencode"
        convo = [normalize_message(m) for m in messages if isinstance(m, dict)]
        convo.append(normalize_message(message))
        key = session_key(convo)
        prev = by_session.get(key)
        if prev is None or len(convo) > len(prev["messages"]):
            by_session[key] = {
                "messages": convo,
                "tools": req.get("tools") or None,
                "source": source,
                "_group": (replay or {}).get("task_id") or key,
            }
    print(f"  opencode: {n_records} records -> {len(by_session)} sessions "
          f"({n_replay_fail} failed-replay, {n_holdout} held-out records "
          f"dropped)", file=sys.stderr)
    return list(by_session.values())


def load_cursor(raw_dir: Path, min_answer_chars: int = 200) -> list[dict]:
    samples = []
    files = sorted(raw_dir.rglob("*.jsonl"))
    for path in files:
        if path.name == "index.jsonl" or "subagents" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        query: str | None = None
        last_answer: str | None = None
        pairs: list[tuple[str, str]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = obj.get("role")
            if role not in ("user", "assistant"):
                continue
            text = content_to_text((obj.get("message") or {}).get("content"))
            if not text.strip():
                continue
            if role == "user":
                if query and last_answer:
                    pairs.append((query, last_answer))
                m = USER_QUERY_RE.search(text)
                query = (m.group(1) if m else text).strip()
                last_answer = None
            else:
                last_answer = text.strip()
        if query and last_answer:
            pairs.append((query, last_answer))
        for q, a in pairs:
            if len(a) < min_answer_chars:
                continue
            samples.append({
                "messages": [{"role": "user", "content": q},
                             {"role": "assistant", "content": a}],
                "tools": None,
                "source": "cursor",
            })
    print(f"  cursor: {len(files)} transcripts -> {len(samples)} QA pairs",
          file=sys.stderr)
    return samples


def balance(samples: list[dict], max_ratio: float, seed: int = 0) -> list[dict]:
    if max_ratio <= 0:
        return samples
    by_src: dict[str, list[dict]] = {}
    for s in samples:
        by_src.setdefault(s["source"], []).append(s)
    if len(by_src) < 2:
        return samples
    sizes = {k: len(v) for k, v in by_src.items()}
    small = min(sizes.values())
    cap = max(1, int(small * max_ratio))
    rng = random.Random(seed)
    out = []
    for src, group in by_src.items():
        if len(group) > cap:
            group = rng.sample(group, cap)
            print(f"  balance: {src} downsampled {sizes[src]} -> {cap}",
                  file=sys.stderr)
        out.extend(group)
    rng.shuffle(out)
    return out


def _drop_marker_contaminated(samples: list[dict]) -> list[dict]:
    """Drop rows whose serialized trace hits pack-specific contamination markers."""
    script = PACK_ROOT / "scripts" / "check_contamination.py"
    if not script.is_file():
        return samples
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_contamination", script)
    if spec is None or spec.loader is None:
        return samples
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kept: list[dict] = []
    dropped = 0
    for s in samples:
        if mod.scan_row(json.dumps(s, ensure_ascii=False)):
            dropped += 1
        else:
            kept.append(s)
    if dropped:
        print(f"  marker gate: dropped {dropped}/{len(samples)} contaminated rows",
              file=sys.stderr)
    return kept


def clean_sft_records(
    rows: list[dict],
    *,
    max_tokens: int = 64000,
    model_name: str = "local-model",
    workspace: str = "/workspace",
    pack_root: Path | None = None,
    decontam: bool = True,
    decontam_ngram: int = NGRAM_SIZE,
    do_sanitize: bool = True,
) -> list[dict]:
    """Run dedup/filter/decontam/sanitize on already-built SFT records."""
    samples = apply_filters(list(rows), max_tokens)
    if do_sanitize:
        samples = sanitize(samples, model_name, workspace)
    samples = dedupe_prefix(samples)
    if decontam:
        samples = decontaminate(samples, pack_root or PACK_ROOT)
    return _drop_marker_contaminated(samples)


def process_sft_input(args: argparse.Namespace) -> int:
    src = args.sft_input
    samples = load_sft_input(src, sys.stdin)
    if not samples:
        print("no samples in --sft-input", file=sys.stderr)
        return 1

    if not args.no_filter:
        samples = apply_filters(samples, args.max_tokens)
        samples = dedupe_prefix(samples)
    if not args.no_sanitize:
        samples = sanitize(samples, args.model_name, args.workspace_path)
    samples = decontaminate(samples, args.pack_root)

    write_samples(samples, args.out, args.stdout)
    if not args.stdout and args.out:
        print(f"wrote {len(samples)} samples -> {args.out}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path(__file__).resolve().parent
    p.add_argument("--sft-input", type=Path, default=None,
                   help="existing SFT JSONL (or '-' for stdin)")
    p.add_argument("--stdout", action="store_true",
                   help="write filtered JSONL to stdout")
    p.add_argument("--no-filter", action="store_true",
                   help="skip quality filters (decontam only)")
    p.add_argument("--pack-root", type=Path, default=PACK_ROOT,
                   help="agent-problem-pack root for n-gram decontamination")
    p.add_argument("--opencode", type=Path, default=base / "raw" / "opencode")
    p.add_argument("--cursor", type=Path, default=base / "raw" / "cursor")
    p.add_argument("--out", type=Path, default=base / "dataset" / "sft.jsonl")
    p.add_argument("--max-tokens", type=int, default=64000)
    p.add_argument("--balance", type=float, default=0)
    p.add_argument("--sources", default="")
    p.add_argument("--exclude-tasks", type=Path, default=None)
    p.add_argument("--eval-frac", type=float, default=0.0)
    p.add_argument("--model-name", default="local-model")
    p.add_argument("--workspace-path", default="/workspace")
    p.add_argument("--no-sanitize", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.sft_input is not None:
        return process_sft_input(args)

    exclude_ids: set[str] = set()
    if args.exclude_tasks and args.exclude_tasks.is_file():
        exclude_ids = {json.loads(l)["id"]
                       for l in args.exclude_tasks.read_text().splitlines()
                       if l.strip()}
        print(f"excluding traces from {len(exclude_ids)} held-out tasks "
              f"({args.exclude_tasks.name})", file=sys.stderr)

    print("loading traces...", file=sys.stderr)
    samples: list[dict] = []
    if args.opencode.is_dir():
        samples += load_opencode(args.opencode, exclude_ids)
    if args.cursor.is_dir():
        samples += load_cursor(args.cursor)
    if not samples:
        print("no samples found", file=sys.stderr)
        return 1

    if args.sources:
        allowed = {s.strip() for s in args.sources.split(",") if s.strip()}
        before = len(samples)
        samples = [s for s in samples if s["source"] in allowed]
        print(f"  sources: kept {len(samples)}/{before} ({sorted(allowed)})",
              file=sys.stderr)

    samples = apply_filters(samples, args.max_tokens)
    samples = balance(samples, args.balance, args.seed)
    if not args.no_sanitize:
        samples = sanitize(samples, args.model_name, args.workspace_path)
    samples = decontaminate(samples, args.pack_root)

    eval_samples: list[dict] = []
    if args.eval_frac > 0 and samples:
        def group_of(s: dict) -> str:
            return s.get("_group") or hashlib.sha256(
                json.dumps(s["messages"][:1], sort_keys=True).encode()
            ).hexdigest()

        def group_hash(g: str) -> int:
            return int.from_bytes(
                hashlib.sha256(f"{args.seed}:{g}".encode()).digest()[:8], "big")

        groups = sorted({group_of(s) for s in samples}, key=group_hash)
        n_eval = max(1, round(args.eval_frac * len(groups)))
        eval_groups = set(groups[:n_eval])
        eval_samples = [s for s in samples if group_of(s) in eval_groups]
        samples = [s for s in samples if group_of(s) not in eval_groups]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    eval_out = args.out.with_name(args.out.stem + "-eval.jsonl")
    for path, group in ((args.out, samples), (eval_out, eval_samples)):
        if path is eval_out and not eval_samples:
            continue
        with path.open("w", encoding="utf-8") as fh:
            for s in group:
                s = {k: v for k, v in s.items() if not k.startswith("_")}
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    by_src: dict[str, int] = {}
    for s in samples:
        by_src[s["source"]] = by_src.get(s["source"], 0) + 1
    total_tok = sum(approx_tokens(s["messages"]) for s in samples)
    print(f"wrote {len(samples)} samples ({by_src}) "
          f"~{total_tok/1e6:.2f}M tokens -> {args.out}", file=sys.stderr)
    if eval_samples:
        print(f"wrote {len(eval_samples)} eval samples -> {eval_out}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
