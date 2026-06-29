# Design and Evaluation Methodology

This document describes how the Agent Problem Pack is structured, why problems are designed the way they are, and how to interpret results from agent runs.

## Goals

The pack measures whether coding agents can:

1. **Repair** — diagnose a failing test suite and apply a minimal, correct fix.
2. **Comprehend** — read real project artifacts and explain architecture, data flow, or tooling lifecycle in writing.

Problems are intentionally small. Each repair task should be solvable with a localized change (often one file, often a few lines). Comprehension tasks use excerpted source from this repository so answers can be checked automatically against required concepts.

## Problem taxonomy

| Kind | Count | Verification | What “pass” means |
| --- | --- | --- | --- |
| `repair` | 10 | `pytest` in an isolated workspace | All tests green after the agent’s edits |
| `comprehension` | 4 | Keyword/concept checks on `AGENT_FINAL_ANSWER.md` | Written answer covers required lifecycle or pipeline concepts |

Repair problems span common failure modes seen in agent benchmarks and production codebases:

- Off-by-one and edge-case logic (`problem-01`, `problem-10`)
- Security footguns (`problem-02`)
- Path and import portability (`problem-03`, `problem-04`)
- Shared mutable state (`problem-05`, `problem-07`)
- Configuration precedence bugs (`problem-06`)
- Multi-file contract repair across a realistic package (`problem-11`)
- Invariant/property satisfaction beyond example tests (`problem-12`)

Harder problems raise difficulty along axes that match real agent work: fixes that
span multiple modules, bugs whose contract lives in docstrings rather than the visible
tests, and correctness that only holds under property-based fuzzing.

Comprehension problems (`problem-08`, `problem-09`, `problem-13`, `problem-14`) test whether an agent can navigate multi-file context and produce structured technical writing without modifying code. `problem-13` and `problem-14` use excerpted source from external research (Persona Vectors) and from this repo's own eval methodology (a "meta" eval-suite design task), respectively.

## Run lifecycle

Each evaluation run is isolated:

```
agent-problem-pack/
├── problem-XX-.../          # pristine template (never edited during eval)
└── runs/
    └── problem-XX-.../
        └── <run-name>/
            ├── metadata.json
            ├── artifacts/
            │   ├── task-prompt.txt      # agent-facing instruction
            │   ├── usage.json           # normalized token usage
            │   ├── diff.patch           # captured after agent work
            │   ├── git-status.txt
            │   ├── verification.txt     # pytest output
            │   └── evaluate-with-codex.md
            └── workspace/               # git init + baseline commit
                ├── (problem files)
                ├── pyproject.toml
                └── AGENT_FINAL_ANSWER.md
```

### `prepare`

1. Copy the problem template into `workspace/`.
2. Attach the pack-level `pyproject.toml` so `uv run pytest` works uniformly.
3. Write `artifacts/task-prompt.txt` and seed `usage.json`.
4. Initialize git with a `baseline` commit so diffs are meaningful.

### Agent work

The agent receives only the task prompt (plus whatever it discovers in the workspace). It should not see `expected_behavior` rubric text or golden solutions.

### `capture`

1. Run the problem’s verify command (`pytest`).
2. Record stdout/stderr to `verification.txt`.
3. Capture `git diff` and status.
4. Emit `evaluate-with-codex.md` — a rubric-backed prompt for a separate judge model.

Automated harnesses should overwrite `artifacts/usage.json` with normalized token counts before calling `capture`.

## Hidden tests and overfit resistance

A problem may include a `tests_hidden/` directory next to `tests/`. The harness:

1. Omits `tests_hidden/` from the workspace during `prepare` (the agent never sees it).
2. Injects it into the workspace immediately before running the verify command during
   `capture` and `validate`, then removes it again afterward.
3. Excludes it from `diff.patch`, so a hidden test can never leak into captured artifacts.

This lets the visible `tests/` give the agent a feedback loop while grading uses additional
checks. A fix that overfits the visible suite (for example, hard-coding an example or fixing
only the obvious symptom) can pass `tests/` yet still fail `tests_hidden/`. Property-based
tests (`hypothesis`) are used in `tests_hidden/` to catch order-dependent or invariant-violating
fixes. Problems using hidden tests set their verify command to include both directories, e.g.
`uv run pytest tests tests_hidden`.

When validating, the failing baseline is checked in the pristine problem directory (which
contains `tests_hidden/`), and the golden fix is checked against both suites — so a registered
golden fix must satisfy the hidden tests too.

## Scoring guidance

For repair tasks, **test pass rate is necessary but not sufficient**. The evaluation prompt asks a judge to check:

- Was the root cause identified correctly in `AGENT_FINAL_ANSWER.md`?
- Were edits scoped to the relevant files?
- Does the fix look like a minimal safe change vs. a brittle workaround?

For comprehension tasks, pytest checks required vocabulary and concepts. A human or judge model should still assess clarity and correctness of the explanation.

## Pack integrity

Run validation before publishing results or cutting a release:

```bash
uv run python scripts/pack_tools.py validate
```

This checks that every problem directory exists, repair baselines fail as expected, and golden reference fixes make all repair tests pass.

## Adding a new problem

1. Create `problem-NN-short-name/` with a minimal repro and focused tests. Optionally add a
   `tests_hidden/` directory for overfit-resistant grading, and a `conftest.py` if the layout
   needs `src/` on the path.
2. Register the problem in `scripts/pack_tools.py` (`PROBLEMS` dict) with `kind`, `difficulty`,
   `skills`, and rubric bullets. If you ship hidden tests, set `verify_command` to include both
   directories (e.g. `("uv", "run", "pytest", "tests", "tests_hidden")`).
3. Add a golden fix entry in `scripts/golden.py` if the kind is `repair`. The golden fix must pass
   both the visible and hidden suites. Multi-file fixes are supported (map several paths).
4. Extend `scripts/test_pack_tools.py` if the problem introduces new harness behavior.
5. Run `uv run python scripts/pack_tools.py validate`.

Keep prompts agent-facing (symptoms and goals), and keep rubric criteria evaluator-facing (`expected_behavior`).
