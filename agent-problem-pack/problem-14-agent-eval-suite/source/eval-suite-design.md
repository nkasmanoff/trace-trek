# Eval-suite design notes (how the agent-problem-pack works)

These notes describe how trace-trek's benchmark for coding agents is built. Use them as the
reference design for an agent evaluation suite.

## Goals

The suite measures whether a coding agent can:

1. **Repair** — diagnose a failing test suite and apply a minimal, correct fix.
2. **Comprehend** — read real project artifacts and explain architecture, data flow, or a
   tooling lifecycle in writing, without editing code.

Problems are intentionally small and self-contained so results are attributable to a single
capability rather than to scaffolding luck.

## Task taxonomy and verification

| Kind | Verification | What "pass" means |
| --- | --- | --- |
| `repair` | `pytest` in an isolated workspace | All tests green after the agent's edits |
| `comprehension` | keyword/concept checks on `AGENT_FINAL_ANSWER.md` | Written answer covers required concepts |

Each problem declares a `verify_command` (e.g. `uv run pytest tests`) and an
evaluator-facing rubric (`expected_behavior`) that is **never shown to the agent**.

## Run lifecycle (isolation)

Every run is isolated under `runs/<problem>/<run-name>/` with an `artifacts/` and a
`workspace/` directory.

- **prepare**: copy the pristine problem template into `workspace/`, attach a shared
  `pyproject.toml`, write `artifacts/task-prompt.txt`, seed `usage.json`, then `git init` +
  a `baseline` commit so a later `git diff` is meaningful. The agent sees only the task
  prompt and whatever it discovers in the workspace.
- **agent work**: the agent edits files and/or writes `AGENT_FINAL_ANSWER.md`.
- **capture**: run the `verify_command`, save stdout/stderr to `verification.txt`, capture
  `diff.patch` and `git-status.txt`, and emit a rubric-backed `evaluate-with-codex.md`
  prompt for a separate judge model. Harnesses overwrite `usage.json` with normalized token
  counts before capture.

## Overfit / gaming resistance

A problem may ship a `tests_hidden/` directory beside the visible `tests/`. The harness:

1. **Omits** `tests_hidden/` during `prepare` (the agent never sees it).
2. **Injects** it just before running the verify command at `capture`/`validate`, then
   removes it again.
3. **Excludes** it from `diff.patch`, so a hidden test can never leak.

The visible suite gives the agent a feedback loop while grading uses extra checks. A fix
that overfits the visible tests (hard-coding an example, patching only the obvious symptom)
can pass `tests/` yet fail `tests_hidden/`. Property-based tests (hypothesis) catch
order-dependent or invariant-violating fixes.

## Pack integrity (validation)

Validation (`pack_tools.py validate`) guarantees each problem is well-formed:

- every problem directory exists;
- each repair **baseline must fail** (the unmodified template has failing tests, proving the
  bug is real);
- a registered **golden fix** makes all tests pass (including hidden tests).

This baseline-fails + golden-passes check is what keeps the suite honest over time.

## Scoring and aggregation

For repair tasks, **test pass rate is necessary but not sufficient**. The judge prompt also
asks whether the root cause was identified, edits were scoped to the relevant files, and the
fix is a minimal safe change rather than a brittle workaround. For comprehension tasks,
pytest checks required vocabulary/concepts and a judge assesses clarity.

Per-run results (pass/fail, score, token usage, agent steps) are aggregated into pass rate
per model so checkpoints can be compared. Token usage is normalized to a fixed schema
(input/output/total/cached/reasoning) so cost is comparable across harnesses.

## Reproducibility

Results are reproducible because each run is isolated (fresh workspace + git baseline), the
verify command is fixed and machine-checkable, problem templates are pristine and never
edited in place, and validation is run before publishing results.
