#!/usr/bin/env python3
"""Manufacture replay tasks from saved opencode proxy traces.

This is the opencode-trace counterpart to make_tasks.py (which mines Cursor
transcripts + git history). The proxy in raw/opencode/ logs one request/
response record per LLM call; each record's request carries the FULL
conversation so far and the working directory is baked into the system prompt's
<env> block. We collapse the records into sessions (maximal trajectory per
session, exactly like build_dataset.load_opencode) and turn the genuine
read/explain conversations into gradable `knowledge` tasks.

Only "still doable" sessions survive:

  - the session's working directory still exists on disk AND is a git repo, so
    collect/replay.py can spin up a worktree (at the chat-era commit) and grade.
  - the directory is NOT an ephemeral replay/temp worktree (those are gone or
    meaningless to re-run) — e.g. /var/folders/.../T/improver-replay-XXXX.

Only genuine read/explain Q&A becomes a task (same rationale as make_tasks.py):
edit/run/generate conversations can't be faithfully replayed because git
time-travel only restores tracked files and the snapshot often already contains
the finished work, yielding misleading false passes. The opening turn must be a
question and the conversation is trimmed at the first mutation turn. Those
skipped sessions are still captured verbatim as SFT trajectories by
build_dataset.load_opencode(), so filtering here loses no training data — it
only keeps the verifiable-eval set clean.

The output (tasks-opencode.jsonl) uses the SAME schema as tasks.jsonl, so it
drops straight into split_tasks.py / collect/replay.py / build_dataset.py's
--exclude-tasks.

Usage:
    python collect/make_tasks_opencode.py \
        --opencode raw/opencode --out tasks-opencode.jsonl \
        [--max-knowledge-per-repo 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# reuse the exact intent classifiers used for Cursor tasks so both task files
# share one definition of "question" vs "mutation"
from make_tasks import is_mutation, is_question  # noqa: E402

# Working directory line inside the opencode harness system prompt's <env> block:
#   <env>
#     Working directory: /Users/.../repo
WORKDIR_RE = re.compile(r"Working directory:\s*(.+)")

# Ephemeral dirs we never treat as "still doable": replay worktrees created by
# replay.py (improver-replay-*) and other transient temp dirs (mkdtemp under the
# system temp folder, /tmp, tmp.* names). Re-running these is impossible or
# pointless.
EPHEMERAL_RE = re.compile(
    r"(/var/folders/|^/tmp/|^/private/tmp/|/T/|-replay-|/tmp\.[A-Za-z0-9]+)"
)

# Degenerate "references" that aren't gradable answers: the agent failed to
# ground (wrong repo / missing file) or punted with a clarification request.
# These appear in replayed prompts whose @file mentions point at a different
# repo than the session's working directory.
NONANSWER_RE = re.compile(
    r"(could you (?:confirm|clarify|provide)|i could ?n'?t find|i (?:can'?t|do "
    r"not|don'?t) (?:find|see|locate)|does not (?:exist|appear)|doesn'?t "
    r"exist|no such file|i'?m not sure (?:what|which)|please (?:confirm|clarify"
    r"|specify)|which (?:file|repo|directory) (?:are|do) you)",
    re.IGNORECASE,
)

MAX_TURNS_PER_TASK = 4

_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")


def has_mutation_sentence(prompt: str) -> bool:
    """True if ANY sentence in the prompt is an imperative edit/run request.

    make_tasks.is_mutation only inspects the prompt's opening, which misses
    mid-prompt imperatives like "Pasted in an updated doc. Add a section ..."
    (an edit task). We apply the same start-anchored classifier per sentence so
    a request buried after some context is still recognised, while prose like
    "Does it make sense ..." (no sentence starts with an action verb) is not."""
    return any(is_mutation(s) for s in _SENTENCE_SPLIT_RE.split(prompt) if s.strip())


def is_question_prompt(prompt: str) -> bool:
    """Treat a turn as a read/explain question if make_tasks' strict opener
    matches OR — opencode prompts are more conversational than Cursor's — it
    poses an explicit question (contains '?') with no imperative edit/run
    sentence anywhere in it."""
    if has_mutation_sentence(prompt):
        return False
    if is_question(prompt):
        return True
    return "?" in prompt


# ------------------------------------------------------------- message helpers


def content_to_text(content) -> str:
    """Flatten OpenAI-style content (str or list-of-parts) to plain text."""
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


def session_key(messages: list[dict]) -> str:
    """Identify a session by its system prompt + first user message (same key
    build_dataset.py uses, so records from one conversation collapse together)."""
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


def strip_wrapping_quotes(text: str) -> str:
    """Replayed prompts arrive wrapped in a single pair of double quotes
    (opencode received them as a quoted CLI arg). Drop one outer pair."""
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].strip()
    return text


def working_dir(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "system":
            m_wd = WORKDIR_RE.search(content_to_text(m.get("content")))
            return m_wd.group(1).strip() if m_wd else None
    return None


def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def conversation_turns(messages: list[dict]) -> list[dict]:
    """(user query -> that turn's final assistant text) pairs. The reference is
    the LAST assistant text before the next user turn — the turn's synthesis,
    skipping intermediate tool-call-only assistant messages and tool results."""
    turns: list[dict] = []
    current: dict | None = None
    for m in messages:
        role = m.get("role")
        if role == "user":
            query = strip_wrapping_quotes(content_to_text(m.get("content")))
            if not query:
                continue
            if current and current.get("reference"):
                turns.append(current)
            current = {"prompt": query, "reference": None}
        elif role == "assistant" and current is not None:
            text = content_to_text(m.get("content")).strip()
            if text:
                current["reference"] = text
    if current and current.get("reference"):
        turns.append(current)
    return turns


# ------------------------------------------------------------- session load


def load_sessions(raw_dir: Path) -> list[dict]:
    """Collapse proxy records into maximal per-session trajectories (the latest
    request in a session is a superset of earlier ones; appending its response
    gives the full conversation)."""
    by_session: dict[str, dict] = {}
    for path in sorted(raw_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        req = rec.get("request") or {}
        resp = rec.get("response") or {}
        messages = req.get("messages")
        message = resp.get("message")
        if not isinstance(messages, list) or not isinstance(message, dict):
            continue
        convo = [m for m in messages if isinstance(m, dict)]
        convo.append(message)
        key = session_key(convo)
        prev = by_session.get(key)
        if prev is None or len(convo) > len(prev["messages"]):
            by_session[key] = {"messages": convo, "timestamp": rec.get("timestamp")}
    return list(by_session.values())


def parse_chat_date(timestamp: str | None) -> str | None:
    """opencode trace timestamps look like '20260609-201744'."""
    if not timestamp:
        return None
    try:
        dt = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc).isoformat()


# ------------------------------------------------------------- task synthesis


def knowledge_tasks(sessions: list[dict], per_repo_cap: int) -> list[dict]:
    tasks: dict[str, dict] = {}
    per_repo: dict[str, int] = {}
    for sess in sessions:
        wd = working_dir(sess["messages"])
        if not wd or EPHEMERAL_RE.search(wd):
            continue
        repo = Path(wd)
        if not repo.is_dir() or not is_git_repo(repo):
            continue
        if per_repo.get(str(repo), 0) >= per_repo_cap:
            continue

        turns = conversation_turns(sess["messages"])[:MAX_TURNS_PER_TASK]
        if not turns or not is_question_prompt(turns[0]["prompt"]):
            continue
        # the opening reference must be a real grounded answer, not a
        # wrong-repo "couldn't find it" / clarification punt
        if NONANSWER_RE.search(turns[0]["reference"]):
            continue
        # trim at the first edit/run/generate turn — not faithfully replayable
        kept: list[dict] = []
        for t in turns:
            if has_mutation_sentence(t["prompt"]):
                break
            kept.append(t)
        if not kept or not any(len(t["reference"]) >= 200 for t in kept):
            continue
        turns = kept
        query = turns[0]["prompt"]
        tid = hashlib.sha256(f"ok:{repo}:{query}".encode()).hexdigest()[:12]
        if tid in tasks:
            continue
        tasks[tid] = {
            "id": tid,
            "type": "knowledge",
            "repo": str(repo),
            "prompt": query,
            "reference": turns[0]["reference"],
            "turns": turns,
            "chat_date": parse_chat_date(sess.get("timestamp")),
            "source": "opencode",
        }
        per_repo[str(repo)] = per_repo.get(str(repo), 0) + 1
    return list(tasks.values())


# ------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path(__file__).resolve().parent.parent
    p.add_argument("--opencode", type=Path, default=base / "raw" / "opencode")
    p.add_argument("--out", type=Path, default=base / "tasks-opencode.jsonl")
    p.add_argument("--max-knowledge-per-repo", type=int, default=20)
    args = p.parse_args()

    if not args.opencode.is_dir():
        print(f"error: opencode dir not found: {args.opencode}")
        return 1

    sessions = load_sessions(args.opencode)
    tasks = knowledge_tasks(sessions, args.max_knowledge_per_repo)

    with args.out.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    by_repo: dict[str, int] = {}
    for t in tasks:
        name = Path(t["repo"]).name
        by_repo[name] = by_repo.get(name, 0) + 1
    print(f"scanned {len(sessions)} opencode sessions -> wrote {len(tasks)} "
          f"knowledge tasks across {len(by_repo)} repos -> {args.out}")
    for repo, n in sorted(by_repo.items(), key=lambda kv: -kv[1]):
        print(f"  {repo}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
