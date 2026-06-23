#!/usr/bin/env python3
"""Manufacture replay tasks from Cursor transcripts and repo git history.

Two task types are emitted to tasks.jsonl:

  knowledge — a real query you once asked Cursor, paired with the stronger
              model's final answer as a grading reference. Replayed against
              the repo (optionally time-traveled to the chat-era commit).
              Only genuine read/explain Q&A becomes a knowledge task: edit,
              run, and generation conversations can't be faithfully replayed
              (git restores only tracked files, and the chat-era snapshot
              frequently already contains the finished work, so the model
              "reproduces" an answer that was already in the repo — a
              misleading false pass). Those conversations are still captured
              as text-only distillation pairs by build_dataset.load_cursor(),
              so filtering them here loses no data; it only keeps the
              verifiable-replay set clean.

  code      — synthesized from git history: check out a commit's PARENT,
              prompt with the commit message, verify the attempt against the
              real changed files. Unlimited, automatically verifiable supply.

Usage:
    python collect/make_tasks.py --cursor raw/cursor --out tasks.jsonl \
        [--max-commits-per-repo 20] [--max-knowledge-per-repo 20]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)

# Optional polite/filler lead-in that may precede the real intent verb, e.g.
# "please", "can you", "hey!", "ok", "let's", "I would like you to". Both
# classifiers below anchor at the start AFTER this lead, so the distinction is
# the FIRST real word: "run X" (imperative -> mutation) vs "how do I run X"
# (informational -> question).
_LEAD = (
    r"(?:(?:please|kindly|can you|could you|would you|will you|hey+|hi|so|"
    r"ok(?:ay)?|yes|sure|great|now|next|also|then|first|quick question|"
    r"let'?s|i'?d?|i would|i want|i'?d? (?:like|want)(?: you)?(?: to)?|"
    r"i would (?:like|want)(?: you)?(?: to)?|"
    r"we(?:'?re| are| will| need)(?: going)?(?: to)?|go ahead(?: and)?)"
    r"[\s,!.:;-]+)*"
)

# Imperative request to CHANGE/RUN/PRODUCE something — cannot be faithfully
# replayed (see module docstring).
MUTATE_RE = re.compile(
    _LEAD +
    r"(make|create|add|implement|re-?implement|refactor|re-?write|write|build|"
    r"remove|delete|drop|rename|migrate|disable|enable|run|re-?run|execute|"
    r"move|finish|complete|fix|update|convert|replace|integrate|reconcile|wire|"
    r"rip|generate|host|deploy|set ?up|hook ?up|change|modify|edit|patch|tweak|"
    r"adjust|install|configure|optimi[sz]e|stress[- ]?test|do (?:this|it|that))"
    r"\b",
    re.IGNORECASE,
)

# Genuine question / explanation opener — the only thing we replay as knowledge.
QUESTION_RE = re.compile(
    _LEAD +
    r"(how|what|wha?t'?s|where|why|when|which|who|whose|whom|does|do|did|is|are|"
    r"was|were|should|would|could|show me|walk me|tell me|remind me|clarify|"
    r"explain|summari[sz]e|describe|"
    r"can (?:you )?(?:explain|summari[sz]e|describe|walk|tell|clarify|help)|"
    r"help me understand|i'?m (?:new|trying to understand|curious|confused)|"
    r"i want to understand|i'?m wondering|give me an?)\b",
    re.IGNORECASE,
)


def is_mutation(prompt: str) -> bool:
    """True if the turn asks the agent to change/run/produce something."""
    return bool(MUTATE_RE.match(prompt.lstrip()))


def is_question(prompt: str) -> bool:
    """True if the turn is a read/explain question (and not a mutation)."""
    p = prompt.lstrip()
    return bool(QUESTION_RE.match(p)) and not is_mutation(p)


# ------------------------------------------------------------- repo resolve


@functools.lru_cache(maxsize=None)
def resolve_project_path(project: str) -> Path | None:
    """Cursor project folders encode the path with '-' for '/'. Directory
    names may themselves contain dashes, so resolve by DFS over split points."""
    segments = project.split("-")

    def dfs(base: Path, idx: int) -> Path | None:
        if idx == len(segments):
            return base
        # greedily try the longest joined component first
        for end in range(len(segments), idx, -1):
            candidate = base / "-".join(segments[idx:end])
            if candidate.is_dir():
                found = dfs(candidate, end)
                if found is not None:
                    return found
        return None

    return dfs(Path("/"), 0)


def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=60,
    )
    return out.stdout.strip()


# ------------------------------------------------------------- transcripts


def conversation_turns(path: Path) -> list[dict]:
    """Extract (user query -> final assistant answer) pairs for every human
    turn in a transcript. Each turn's reference is the LAST assistant text
    before the next user query, i.e. the turn's final answer."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    turns: list[dict] = []
    current: dict | None = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("role")
        texts = [
            p.get("text", "") for p in
            ((obj.get("message") or {}).get("content") or [])
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        text = "\n".join(t for t in texts if t.strip())
        if not text.strip():
            continue
        if role == "user":
            m = USER_QUERY_RE.search(text)
            if m:
                query = m.group(1).strip()
            elif not turns and current is None:
                query = text.strip()  # first message may lack the tag
            else:
                continue  # injected context / reminders, not a human turn
            if current and current.get("reference"):
                turns.append(current)
            current = {"prompt": query, "reference": None}
        elif role == "assistant" and current is not None:
            current["reference"] = text.strip()
    if current and current.get("reference"):
        turns.append(current)
    return turns


MAX_TURNS_PER_TASK = 4


def knowledge_tasks(cursor_dir: Path, per_repo_cap: int) -> list[dict]:
    tasks: list[dict] = []
    per_repo: dict[str, int] = {}
    for project_dir in sorted(p for p in cursor_dir.iterdir() if p.is_dir()):
        repo = resolve_project_path(project_dir.name)
        if repo is None or not is_git_repo(repo):
            continue
        for jsonl in sorted(project_dir.rglob("*.jsonl")):
            if "subagents" in jsonl.parts:
                continue
            if per_repo.get(str(repo), 0) >= per_repo_cap:
                break
            # keep turns in order (replay needs the full conversation), but
            # require at least one substantive reference to grade against
            turns = conversation_turns(jsonl)[:MAX_TURNS_PER_TASK]
            if not turns:
                continue
            # keep only genuine read/explain Q&A. The opening turn must be a
            # question, and we trim the conversation at the first edit/run/
            # generation turn — those can't be faithfully replayed and survive
            # as text-only distillation (build_dataset.load_cursor) instead.
            if not is_question(turns[0]["prompt"]):
                continue
            kept: list[dict] = []
            for t in turns:
                if is_mutation(t["prompt"]):
                    break
                kept.append(t)
            if not kept or not any(len(t["reference"]) >= 200 for t in kept):
                continue
            turns = kept
            query = turns[0]["prompt"]
            chat_date = datetime.fromtimestamp(
                jsonl.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            tid = hashlib.sha256(f"k:{repo}:{query}".encode()).hexdigest()[:12]
            tasks.append({
                "id": tid,
                "type": "knowledge",
                "repo": str(repo),
                "prompt": query,
                "reference": turns[0]["reference"],
                "turns": turns,
                "chat_date": chat_date,
            })
            per_repo[str(repo)] = per_repo.get(str(repo), 0) + 1
    return tasks


# ------------------------------------------------------------- git commits


def code_tasks(repos: list[Path], per_repo_cap: int) -> list[dict]:
    tasks: list[dict] = []
    for repo in repos:
        log = git(repo, "log", "--no-merges", "--pretty=%H\x01%s\x01%b\x02",
                  "-n", "200")
        count = 0
        for entry in log.split("\x02"):
            if count >= per_repo_cap:
                break
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split("\x01")
            if len(parts) < 2:
                continue
            sha, subject = parts[0], parts[1]
            body = parts[2] if len(parts) > 2 else ""
            message = (subject + "\n" + body).strip()
            if len(subject) < 20 or subject.lower().startswith(
                    ("wip", "merge", "bump", "typo", "fmt", "lint")):
                continue
            # numstat gives untruncated paths (--stat abbreviates long ones
            # to ".../suffix", which silently broke replay grading)
            numstat = git(repo, "show", "--numstat", "--format=", sha)
            files, insertions = [], 0
            for l in numstat.splitlines():
                parts = l.split("\t")
                if len(parts) != 3:
                    continue
                added, _, path = parts
                insertions += int(added) if added.isdigit() else 0
                # renames: "dir/{old.py => new.py}" or "old.py => new.py"
                m = re.match(r"^(.*)\{(.*) => (.*)\}(.*)$", path)
                if m:
                    path = f"{m.group(1)}{m.group(3)}{m.group(4)}"
                elif " => " in path:
                    path = path.split(" => ")[1]
                files.append(path)
            if not (1 <= len(files) <= 10 and 5 <= insertions <= 500):
                continue
            parent = git(repo, "rev-parse", f"{sha}^")
            if not parent:
                continue
            tid = hashlib.sha256(f"c:{repo}:{sha}".encode()).hexdigest()[:12]
            tasks.append({
                "id": tid,
                "type": "code",
                "repo": str(repo),
                "commit": sha,
                "parent": parent,
                "prompt": (
                    "Implement the following change in this repository:\n\n"
                    + message
                ),
                "changed_files": files,
            })
            count += 1
    return tasks


# ------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path(__file__).resolve().parent.parent
    p.add_argument("--cursor", type=Path, default=base / "raw" / "cursor")
    p.add_argument("--out", type=Path, default=base / "tasks.jsonl")
    p.add_argument("--max-commits-per-repo", type=int, default=20)
    p.add_argument("--max-knowledge-per-repo", type=int, default=20)
    p.add_argument("--extra-repos", type=Path, nargs="*", default=[],
                   help="additional git repos to mine for code tasks")
    args = p.parse_args()

    ktasks = knowledge_tasks(args.cursor, args.max_knowledge_per_repo) \
        if args.cursor.is_dir() else []

    repos: dict[str, Path] = {}
    for t in ktasks:
        repos[t["repo"]] = Path(t["repo"])
    if args.cursor.is_dir():
        # also mine repos whose transcripts were all code-style asks
        for project_dir in sorted(p for p in args.cursor.iterdir() if p.is_dir()):
            r = resolve_project_path(project_dir.name)
            if r and is_git_repo(r):
                repos[str(r)] = r
    for r in args.extra_repos:
        r = r.expanduser().resolve()
        if is_git_repo(r):
            repos[str(r)] = r

    ctasks = code_tasks(list(repos.values()), args.max_commits_per_repo)

    tasks = ktasks + ctasks
    with args.out.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    by_repo: dict[str, int] = {}
    for t in tasks:
        by_repo[Path(t["repo"]).name] = by_repo.get(Path(t["repo"]).name, 0) + 1
    print(f"wrote {len(tasks)} tasks ({len(ktasks)} knowledge, "
          f"{len(ctasks)} code) across {len(repos)} repos -> {args.out}")
    for repo, n in sorted(by_repo.items(), key=lambda kv: -kv[1]):
        print(f"  {repo}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
