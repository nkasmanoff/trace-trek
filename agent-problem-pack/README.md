# Agent Problem Pack

A reproducible benchmark suite for evaluating coding agents on **repair** and **comprehension** tasks. Each problem ships with an isolated workspace workflow, automated verification, rubric-backed evaluation prompts, and normalized token-usage capture.

See [DESIGN.md](DESIGN.md) for methodology, taxonomy, and guidance on adding problems.

## Quick start

From the pack root:

```bash
cd agent-problem-pack
uv sync
uv run python scripts/pack_tools.py list
uv run python scripts/pack_tools.py validate
```

### Run one problem manually

Prepare an isolated workspace:

```bash
uv run python scripts/pack_tools.py prepare problem-01-tokenizer-regression run-1
```

The command prints the run directory and task prompt path. Open the workspace in your agent:

```bash
cd runs/problem-01-tokenizer-regression/run-1/workspace
```

Use the prompt at `../artifacts/task-prompt.txt`. The agent can verify locally with:

```bash
uv run pytest
```

After the agent finishes, capture artifacts from the pack root:

```bash
uv run python scripts/pack_tools.py capture runs/problem-01-tokenizer-regression/run-1
```

Open `runs/.../artifacts/evaluate-with-codex.md` for a rubric-backed judge prompt. Each run also records token usage at `artifacts/usage.json`.

Use a distinct run name per model or attempt (`codex-1`, `claude-1`, `qwen-1`, …).

## Problem catalog

| ID | Title | Kind | Difficulty | Verify |
| --- | --- | --- | --- | --- |
| `problem-01-tokenizer-regression` | Tokenizer regression on empty CSV fields | repair | easy | `uv run pytest` |
| `problem-02-shell-command-injection` | Remove shell injection from task runner | repair | medium | `uv run pytest` |
| `problem-03-cross-platform-task-path` | Anchor task file path to script location | repair | easy | `uv run pytest tests` |
| `problem-04-import-error-after-refactor` | Restore compatibility after module move | repair | easy | `uv run pytest tests` |
| `problem-05-mutable-default-cache` | Fix shared state from mutable default arg | repair | medium | `uv run pytest` |
| `problem-06-config-merge-priority` | Correct env > file > defaults precedence | repair | medium | `uv run pytest tests` |
| `problem-07-thread-safe-cache` | Fix TOCTOU race in concurrent cache | repair | hard | `uv run pytest tests` |
| `problem-08-pack-lifecycle` | Explain prepare → agent → capture lifecycle | comprehension | medium | `uv run pytest tests` |
| `problem-09-eval-pipeline-viewer` | Trace eval results into viewer UI | comprehension | medium | `uv run pytest tests` |
| `problem-10-flatten-depth` | Fix off-by-one in recursive list flatten | repair | medium | `uv run pytest tests` |
| `problem-11-eval-scoring-pipeline` | Repair multi-file eval scoring/aggregation contract | repair | hard | `uv run pytest tests tests_hidden` |
| `problem-12-merge-latest-property` | Satisfy dedup invariants under property tests | repair | hard | `uv run pytest tests tests_hidden` |
| `problem-13-persona-vector-extraction` | Explain persona-vector extraction, monitoring & steering | comprehension | hard | `uv run pytest tests` |
| `problem-14-agent-eval-suite` | Design an eval suite for a coding agent (meta) | comprehension | hard | `uv run pytest tests` |
| `problem-15-pipeline-bug-trace` | Weighted score bug spanning runner + aggregate | repair | hard | `uv run pytest tests tests_hidden` |
| `problem-16-grounded-audit` | Report values only obtainable by executing the audit | grounding | hard | `uv run pytest tests` |
| `problem-17-stage-localization` | Localize one faulty stage across 30+ modules | repair | hard | `uv run pytest tests tests_hidden` |
| `problem-18-edit-gauntlet` | Precisely fix 3 of 6 near-identical handlers | repair | hard | `uv run pytest tests tests_hidden` |
| `problem-19-follow-the-pattern` | Add a service following framework conventions | implement | hard | `uv run pytest tests tests_hidden` |
| `problem-20-limiter-follow-ups` | Fix limiter, then extend API over follow-up turns | repair | hard | `uv run pytest tests tests_hidden` |
| `problem-21-js-eval-aggregate` | Repair JS eval aggregation + flip detection | repair | hard | `node --test tests/*.test.js tests_hidden/*.test.js` |
| `problem-22-implement-jsonl-support` | Terse commit-style request: "jsonl support" | implement | hard | `uv run pytest tests tests_hidden` |
| `problem-23-implement-retry-backoff` | Terse commit-style request: retry with backoff | implement | hard | `uv run pytest tests tests_hidden` |
| `problem-24-implement-dry-run` | Terse commit-style request: "add --dry-run" | implement | hard | `uv run pytest tests tests_hidden` |
| `problem-25-implement-parse-lineno` | Terse request: line numbers in parse errors (exact contract in docstring) | implement | hard | `uv run pytest tests tests_hidden` |
| `problem-26-lru-pin-revision` | Repair pinnable LRU cache to its full documented contract | repair | hard | `uv run pytest tests tests_hidden` |
| `problem-27-implement-rates-section` | Terse request: add report rates section per docs spec | implement | hard | `uv run pytest tests tests_hidden` |
| `problem-28-spreadsheet-grounding` | Report values only obtainable by parsing a messy attached CSV | grounding | hard | `uv run pytest tests` |
| `problem-29-web-research-hackathon` | Answer a fact absent from the repo (web research or honest "don't know") | grounding | hard | `uv run pytest tests` |

### Task kinds

- **repair** — failing tests; diagnose and fix with a minimal change.
- **comprehension** — read source, explain in `AGENT_FINAL_ANSWER.md`; graded by concept checks.
- **implement** — a terse, commit-message-style change request. Visible tests pass at
  baseline; grading tests are hidden, so there is no feedback loop for the new
  behavior — the agent must recover the intended contract from repo conventions,
  docstrings, and READMEs (mirrors the "implement the following change" tasks in
  the opencode SFT traces).
- **grounding** — the answer must contain values (a hash token, counts) only
  obtainable by actually running code in the workspace. Discriminates agents that
  execute from agents that hallucinate plausible output.

Multi-turn problems (e.g. `problem-20`) declare follow-up `turns`; harnesses
deliver them with `opencode run --continue` (see `pipeline/eval/run_problem_pack.py`).
Final grading covers all turns.

### Hidden tests

Some harder problems ship a `tests_hidden/` directory alongside the visible `tests/`.
Hidden tests are **withheld from the agent during `prepare`** and only injected at
`capture` (and during `validate`) time. The agent gets a feedback loop from the visible
suite, but a fix that overfits the visible symptoms still fails grading. Hidden tests are
removed from the workspace after verification and never appear in `diff.patch`. Their
verify command includes both paths, e.g. `uv run pytest tests tests_hidden`.

Show full metadata and rubric for one problem:

```bash
uv run python scripts/pack_tools.py info problem-07-thread-safe-cache
```

## Headless agent harnesses

These CLIs support scripted benchmark runs:

| Agent | Headless command shape | Notes |
| --- | --- | --- |
| Codex | `codex exec --json "<prompt>"` | Extract `usage` from the final `turn.completed` event. |
| Claude Code | `ollama launch claude --model <model> -- --output-format json -p "<prompt>"` | Pass Claude flags after `--`. |
| Cline | `cline --json "<prompt>"` | JSON mode when available. |
| Qwen Code | `qwen --model <model> --output-format json -p "<prompt>"` | Confirm flags with `qwen --help`. |

For unattended runs in isolated workspaces, Claude may use `--permission-mode bypassPermissions` after `--`.

### Token usage schema

Write normalized usage to `artifacts/usage.json`:

```json
{
  "schema_version": 1,
  "harness": "claude",
  "model": "qwen3.6:35b",
  "source": "cli_json",
  "exact": true,
  "input_tokens": 24327,
  "output_tokens": 16,
  "total_tokens": 24343,
  "cached_input_tokens": 0,
  "reasoning_output_tokens": 0,
  "raw_usage": {},
  "notes": ""
}
```

If exact counts are unavailable, set `"exact": false` and explain in `"notes"`.

## Automated suite evaluation

For orchestrated runs, use the headless evaluator skill at `skills/headless-evaluator/SKILL.md`.

**Warning:** automated evaluation prompts agents to read and modify files. Run only in a sandbox or dedicated evaluation machine.

Replace CLI, model, and run name as needed:

```text
In <path-to>/agent-problem-pack

Evaluate this problem pack with a headless coding agent.

Work only from the agent-problem-pack folder. Do not use files outside this folder.

First read:
- README.md
- DESIGN.md
- skills/headless-evaluator/SKILL.md

Target agent:
- CLI: codex
- model: qwen3.6:35b
- run name: qwen36-run-1

Use the pack scripts:
- list: uv run python scripts/pack_tools.py list
- validate: uv run python scripts/pack_tools.py validate
- prepare one isolated workspace per problem
- run the target agent headlessly from each workspace using artifacts/task-prompt.txt
- write token usage to artifacts/usage.json
- capture: uv run python scripts/pack_tools.py capture runs/<problem-id>/<run-name>

For each problem, read workspace/AGENT_FINAL_ANSWER.md and artifacts/{diff.patch,git-status.txt,verification.txt,usage.json,evaluate-with-codex.md}.

Report pass/fail per problem, token totals, failure analysis, and whether edits were appropriately scoped.
```

## Development

Validate pack integrity (baseline failures + golden fixes):

```bash
uv run python scripts/pack_tools.py validate
uv run pytest scripts/test_pack_tools.py
```

Golden reference fixes live in `scripts/golden.py` for CI validation only — they are not copied into agent workspaces.
