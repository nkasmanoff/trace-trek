#!/usr/bin/env python
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Problem:
    identifier: str
    title: str
    kind: str
    difficulty: str
    skills: tuple[str, ...]
    verify_command: tuple[str, ...]
    task_prompt: str
    expected_behavior: tuple[str, ...]
    # Follow-up user turns delivered after task_prompt (multi-turn problems).
    turns: tuple[str, ...] = ()


# Kinds that must have a failing baseline and a registered golden fix.
# - repair:    diagnose failing tests, fix the code
# - implement: terse change request (commit-message style); grading is done by
#              hidden tests only, so the agent gets no visible feedback loop
#              for the new behavior and must infer intent from repo conventions
REPAIR_LIKE_KINDS = {"repair", "implement"}

# Kinds graded by checks on the agent's written/reported output.
# - comprehension: concept checks on AGENT_FINAL_ANSWER.md
# - grounding:     answer must contain values only obtainable by actually
#                  executing code in the workspace (anti-hallucination)
ANSWER_KINDS = {"comprehension", "grounding"}


PROBLEMS = {
    "problem-01-tokenizer-regression": Problem(
        identifier="problem-01-tokenizer-regression",
        title="Tokenizer Regression",
        kind="repair",
        difficulty="easy",
        skills=("debugging", "string-processing", "edge-cases"),
        verify_command=("uv", "run", "pytest"),
        task_prompt=(
            "A tokenizer regression test fails. Diagnose the root cause and make the smallest "
            "safe code change so the tests pass. Explain the fix briefly after editing."
        ),
        expected_behavior=(
            "The agent should identify that splitting stripped empty input can create an empty-string token.",
            "The fix should filter empty parts produced by split.",
            "The tests should pass with uv run pytest.",
            "The change should stay scoped to tokenizer.py unless there is a clear reason.",
        ),
    ),
    "problem-02-shell-command-injection": Problem(
        identifier="problem-02-shell-command-injection",
        title="Shell Command Injection",
        kind="repair",
        difficulty="medium",
        skills=("security", "subprocess", "api-design"),
        verify_command=("uv", "run", "pytest"),
        task_prompt=(
            "Review and fix the command runner. The command comes from a JSON task file that readers may edit. "
            "Make the smallest safe change that avoids command injection risk while preserving support for "
            "explicit argument lists. Explain the risk and the safer direction after editing."
        ),
        expected_behavior=(
            "The agent should identify shell=True as the command-injection risk.",
            "The implementation should avoid shell=True.",
            "The implementation should accept an explicit argument list.",
            "String commands from editable task data should be rejected or otherwise not executed through a shell.",
            "The tests should pass with uv run pytest.",
        ),
    ),
    "problem-03-cross-platform-task-path": Problem(
        identifier="problem-03-cross-platform-task-path",
        title="Cross-Platform Task Path",
        kind="repair",
        difficulty="easy",
        skills=("path-handling", "portability", "debugging"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "The benchmark should find its JSONL task file whether it is run from the project root or from "
            "its own script directory. Make the smallest code change that fixes the path handling. "
            "Explain the change briefly."
        ),
        expected_behavior=(
            "The agent should anchor the task path to the script file, not the current working directory.",
            'The expected pattern is Path(__file__).with_name("personal_tool_reasoning_tasks.jsonl") or an equivalently robust file-relative path.',
            "The fix should be in code/tool-reasoning-benchmark/ollama_tool_reasoning_bench.py.",
            "The tests should pass with uv run pytest tests.",
        ),
    ),
    "problem-04-import-error-after-refactor": Problem(
        identifier="problem-04-import-error-after-refactor",
        title="Import Error After Refactor",
        kind="repair",
        difficulty="easy",
        skills=("imports", "refactoring", "compatibility"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "The test suite fails after a file move from config.py to settings.py. Inspect the failing import "
            "and make the smallest compatibility-preserving fix so existing imports keep working. "
            "Explain what you changed."
        ),
        expected_behavior=(
            "The agent should inspect the failing test/import before editing.",
            "The fix should preserve the old project.config import path.",
            "A small compatibility module src/project/config.py that re-exports from settings.py is the expected minimal fix.",
            "The tests should pass with uv run pytest tests.",
        ),
    ),
    "problem-05-mutable-default-cache": Problem(
        identifier="problem-05-mutable-default-cache",
        title="Mutable Default Cache Leak",
        kind="repair",
        difficulty="medium",
        skills=("python-semantics", "debugging", "test-isolation"),
        verify_command=("uv", "run", "pytest"),
        task_prompt=(
            "A unit test fails only when the whole file is run, but passes in isolation. Diagnose the root "
            "cause and make the smallest safe fix. Explain why the failure only appears when both tests run."
        ),
        expected_behavior=(
            "The agent should identify the mutable default argument as the root cause.",
            "The fix should use None as the default and create a new dict inside the function.",
            "The tests should pass with uv run pytest.",
            "The explanation should mention state leaking across calls/tests.",
        ),
    ),
    "problem-06-config-merge-priority": Problem(
        identifier="problem-06-config-merge-priority",
        title="Config Cascade Merge Priority",
        kind="repair",
        difficulty="medium",
        skills=("configuration", "precedence", "debugging"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "A configuration system loads settings from three sources: hard-coded defaults, a JSON config file, "
            "and environment variables. The intended priority is: env vars > config file > defaults. "
            "Find the bug where env vars do not take priority over the config file and fix it. "
            "Explain why the current merge order is wrong and how your fix restores the intended cascade."
        ),
        expected_behavior=(
            "The agent should load and read the source files in src/ before editing.",
            "The fix should change the merge order: defaults first, then file values, then env vars on top.",
            "The env-var-overrides-file test should pass after the fix.",
            "All four tests in tests/test_merge.py should pass with uv run pytest tests.",
        ),
    ),
    "problem-07-thread-safe-cache": Problem(
        identifier="problem-07-thread-safe-cache",
        title="Thread-Safe Cache Race",
        kind="repair",
        difficulty="hard",
        skills=("concurrency", "locking", "toctou"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "A caching class has a get_or_compute method intended for concurrent use. It has a TOCTOU "
            "(time-of-check-to-time-of-use) race condition that causes the factory function to run more than "
            "once for the same key under concurrent access. Find and fix the race condition. "
            "Explain why the test_concurrent_computes_do_not_duplicate test fails and how your fix prevents "
            "the double computation."
        ),
        expected_behavior=(
            "The agent should identify the race window between the if-check and the cache assignment.",
            "The fix should use a threading.Lock to guard the check-and-compute as an atomic section.",
            "The factory should run at most once per key even under concurrent calls.",
            "All tests in tests/test_cache.py should pass with uv run pytest tests.",
        ),
    ),
    "problem-08-pack-lifecycle": Problem(
        identifier="problem-08-pack-lifecycle",
        title="Agent Problem Pack Lifecycle",
        kind="comprehension",
        difficulty="medium",
        skills=("code-reading", "tooling", "technical-writing"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "Read the files in the source/ directory (pack_tools.py and SKILL.md) and describe the complete "
            "lifecycle of a problem in the agent-problem-pack: what does pack_tools.py prepare set up, "
            "what happens when an agent works on the problem, and how does pack_tools.py capture evaluate "
            "the result? Include the directory structure and key files at each stage. "
            "Write your answer to AGENT_FINAL_ANSWER.md."
        ),
        expected_behavior=(
            "The agent should describe the prepare step: workspace creation, git init, task-prompt.txt.",
            "The agent should describe the agent work step: modifying workspace files, writing AGENT_FINAL_ANSWER.md.",
            "The agent should describe the capture step: running verify_command, capturing diff and status.",
            "The agent should mention metadata.json and usage.json.",
            "All tests in tests/test_answer.py should pass with uv run pytest tests.",
        ),
    ),
    "problem-09-eval-pipeline-viewer": Problem(
        identifier="problem-09-eval-pipeline-viewer",
        title="Eval Pipeline to Viewer",
        kind="comprehension",
        difficulty="medium",
        skills=("code-reading", "data-flow", "technical-writing"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "Read the files in the source/ directory (eval-results.json, eval-tasks.jsonl, eval.js, EvalView.jsx) "
            "and trace how an eval result makes its way from the pipeline evaluation runner to the viewer's "
            "EvalView component. Describe the data schema, the aggregation step, and how the viewer displays "
            "results including the run comparison (flip) feature. "
            "Write your answer to AGENT_FINAL_ANSWER.md."
        ),
        expected_behavior=(
            "The agent should describe the eval result schema: task_id, type, run, passed, score.",
            "The agent should describe the aggregateEval function and how it groups by run and task type.",
            "The agent should describe the EvalView component and how it renders pass rates, scores, and the flip comparison.",
            "All tests in tests/test_answer.py should pass with uv run pytest tests.",
        ),
    ),
    "problem-11-eval-scoring-pipeline": Problem(
        identifier="problem-11-eval-scoring-pipeline",
        title="Eval Scoring Pipeline Repair",
        kind="repair",
        difficulty="hard",
        skills=("ml-eval", "multi-file", "data-pipeline", "edge-cases"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "The evalkit package (src/evalkit/) turns raw runner records into per-(run, type) "
            "summaries. Several tests fail. Read the module docstrings in scoring.py and aggregate.py, "
            "which specify the intended contract, then make the smallest set of changes across the "
            "package so all tests pass. Pay attention to how skipped tasks and out-of-range scores "
            "must be handled. Explain the root causes you found and where you fixed each one."
        ),
        expected_behavior=(
            "The agent should read the scoring.py and aggregate.py docstrings to recover the intended contract.",
            "scoring.normalize_result should clamp score into the [0, 1] range.",
            "aggregate_results should exclude skipped tasks (passed is None) from total, passed, pass_rate, and mean_score.",
            "The fix should span both scoring.py and aggregate.py, not just patch one symptom.",
            "All tests pass with uv run pytest tests tests_hidden, including hidden tests not visible during the run.",
        ),
    ),
    "problem-12-merge-latest-property": Problem(
        identifier="problem-12-merge-latest-property",
        title="Merge-Latest Property Repair",
        kind="repair",
        difficulty="hard",
        skills=("property-testing", "invariants", "ordering", "debugging"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "src/merge.py deduplicates eval records, keeping the latest attempt per task_id. "
            "The visible tests are not enough to catch the bug; read the module docstring carefully, "
            "it specifies the exact contract (highest attempt wins, later-seen breaks ties, result "
            "sorted by task_id, input not mutated). Make the smallest change so the function satisfies "
            "the full contract for any input order. Explain which invariants were being violated."
        ),
        expected_behavior=(
            "The agent should derive the contract from the docstring, not just the visible tests.",
            "The fix should sort the output by task_id ascending.",
            "The fix should break attempt ties in favor of the record seen later in the input.",
            "The fix should preserve one-record-per-task and not mutate the input.",
            "All tests pass with uv run pytest tests tests_hidden, including the hidden hypothesis property tests.",
        ),
    ),
    "problem-10-flatten-depth": Problem(
        identifier="problem-10-flatten-depth",
        title="List Flatten Depth Off-by-One",
        kind="repair",
        difficulty="medium",
        skills=("recursion", "edge-cases", "debugging"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "A recursive list-flattening function supports a depth parameter to limit how many levels of "
            "nesting are flattened. The tests show a subtle off-by-one error: depth=0 should disable "
            "flattening entirely, and depth=1 should flatten exactly one level. Find the off-by-one bug "
            "and fix it with the smallest change. Explain why the bug occurs and how your fix is correct."
        ),
        expected_behavior=(
            "The agent should read and understand the recursive flatten logic.",
            "The fix should correctly guard the recursion so depth is decremented after the check, not before.",
            "The fix should preserve behavior for depth=None (full flatten) and positive depths.",
            "All tests in tests/test_flatten.py should pass with uv run pytest tests.",
        ),
    ),
    "problem-13-persona-vector-extraction": Problem(
        identifier="problem-13-persona-vector-extraction",
        title="Persona Vector Extraction & Steering",
        kind="comprehension",
        difficulty="hard",
        skills=("code-reading", "ml-interpretability", "activation-steering", "technical-writing"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "Read the files in the source/ directory (README.md, generate_vec.py, activation_steer.py) "
            "and explain the Persona Vectors method end to end. Cover: (1) extraction — how a persona "
            "vector is computed as the per-layer mean difference between activations under positive vs "
            "negative (contrastive) prompts, including how effective examples are filtered and the "
            "prompt_avg / response_avg / prompt_last summaries (note which one the paper uses); "
            "(2) monitoring — projecting activations onto the vector to measure a trait; and "
            "(3) control — how ActivationSteerer adds coeff * vector at a chosen layer via a forward "
            "hook, the positions option (all/prompt/response), and the difference between inference-time "
            "steering and training-time preventative steering (steer vs ablate/CAFT). "
            "Write your answer to AGENT_FINAL_ANSWER.md."
        ),
        expected_behavior=(
            "The agent should explain extraction as the per-layer mean of positive activations minus mean of negative activations.",
            "The agent should mention filtering effective contrastive pairs by trait threshold and coherence.",
            "The agent should distinguish prompt_avg, response_avg, and prompt_last, noting response_avg_diff is the one used in the paper.",
            "The agent should describe monitoring via projection of activations onto the vector.",
            "The agent should describe ActivationSteerer adding coeff*vector at a layer via a forward hook, with the positions option.",
            "The agent should distinguish inference-time steering from training-time preventative steering (steer vs ablate/CAFT).",
            "All tests in tests/test_answer.py should pass with uv run pytest tests.",
        ),
    ),
    "problem-15-pipeline-bug-trace": Problem(
        identifier="problem-15-pipeline-bug-trace",
        title="Weighted Pipeline Score Repair",
        kind="repair",
        difficulty="hard",
        skills=("multi-file", "data-pipeline", "debugging", "edge-cases"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "The eval pipeline in src/pipeline/ (dataset -> runner -> aggregate -> report) "
            "reports a wrong weighted suite score. Read the module docstrings, which specify "
            "the weighting contract, then make the smallest set of changes so the whole "
            "pipeline honors it. The bug is not confined to one stage. Explain each root "
            "cause and where you fixed it."
        ),
        expected_behavior=(
            "The agent should recover the weight contract from the docstrings: weighted mean over positive-weight tasks, weight 0.0 excluded but recorded.",
            "The runner must stop coercing an explicit weight of 0.0 to 1.0 (task.get('weight') or 1.0).",
            "The aggregate must divide by the sum of positive weights, not the record count, and exclude zero-weight tasks from the numerator.",
            "The fix should span runner.py and aggregate.py; patching only one stage still fails hidden tests.",
            "All tests pass with uv run pytest tests tests_hidden, including hidden property tests.",
        ),
    ),
    "problem-16-grounded-audit": Problem(
        identifier="problem-16-grounded-audit",
        title="Grounded Execution Audit",
        kind="grounding",
        difficulty="hard",
        skills=("execution", "cli", "grounding", "reporting"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "Run the event-log audit at source/audit.py in STRICT mode and report its "
            "findings. Your final answer must state exactly: the AUDIT_TOKEN, the number "
            "of valid records, the top category, and the mean latency in ms rounded to "
            "one decimal place — all as produced by the strict-mode audit. Do not guess "
            "or estimate any value; every number must come from actually running the audit. "
            "Write your answer to AGENT_FINAL_ANSWER.md."
        ),
        expected_behavior=(
            "The agent should execute source/audit.py with the --strict flag (not just read the code or run the default mode).",
            "The answer must contain the strict-mode AUDIT_TOKEN, which is a hash over the selected record ids and cannot be produced without execution.",
            "The answer must contain the strict-mode valid record count, top category, and mean latency (one decimal).",
            "Reporting default-mode numbers indicates the agent ran the wrong command; fabricated numbers indicate hallucination.",
            "All tests in tests/test_answer.py should pass with uv run pytest tests.",
        ),
    ),
    "problem-17-stage-localization": Problem(
        identifier="problem-17-stage-localization",
        title="Pipeline Stage Fault Localization",
        kind="repair",
        difficulty="hard",
        skills=("fault-localization", "code-search", "multi-file", "debugging"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "An integration test for the flowkit pipeline fails: the 'smooth' stage "
            "produces wrong output. The package has many modules and more than one "
            "windowing implementation; find the stage implementation the pipeline "
            "actually uses, fix the real bug with the smallest change, and do not "
            "touch code that is working. Explain how you located the faulty function."
        ),
        expected_behavior=(
            "The agent should trace the stage registry to src/flowkit/transforms/windows.py rather than guessing from file names.",
            "The fix should change rolling_mean in transforms/windows.py to the documented trailing window (mean of values[max(0, i-window+1):i+1]).",
            "The correct rolling_mean in stats/window_stats.py must not be modified or rewired into the registry.",
            "The integration test must pass without editing tests.",
            "All tests pass with uv run pytest tests tests_hidden, including hidden unit tests pinning the fix to the right module.",
        ),
    ),
    "problem-18-edit-gauntlet": Problem(
        identifier="problem-18-edit-gauntlet",
        title="Handler Table Precision Edits",
        kind="repair",
        difficulty="hard",
        skills=("precise-editing", "code-reading", "edge-cases"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "Three of the six event handlers in src/handlers.py violate the contract "
            "documented at the top of the file; the other three are correct. The "
            "handlers are nearly identical, so be precise: fix exactly the broken "
            "ones and leave the correct ones byte-for-byte untouched. State which "
            "handlers were broken and why."
        ),
        expected_behavior=(
            "The agent should identify the three broken handlers: log (wrong payload key), trace (boundary excludes the limit), alert (ttl wrongly decremented).",
            "Each fix should be a minimal targeted edit in the right handler.",
            "The three correct handlers (metric, audit, heartbeat) must remain unchanged; blanket find/replace edits break them.",
            "All tests pass with uv run pytest tests tests_hidden, including hidden collateral-damage tests.",
        ),
    ),
    "problem-19-follow-the-pattern": Problem(
        identifier="problem-19-follow-the-pattern",
        title="Implement a Service Following Conventions",
        kind="implement",
        difficulty="hard",
        skills=("pattern-following", "code-reading", "implementation"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "Implement the following change in this repository: add a line-stats "
            "service. Given {\"text\": str} it reports the number of non-empty lines, "
            "the number of whitespace-separated words, and the number of distinct "
            "words compared case-insensitively, as {\"lines\": int, \"words\": int, "
            "\"unique_words\": int}. Its public name is \"line-stats\". Follow the "
            "existing service conventions exactly — the framework and the current "
            "services define everything else you need to know."
        ),
        expected_behavior=(
            "The agent should read base.py and the existing services to recover the conventions before writing code.",
            "The new service should subclass BaseService, declare name 'line-stats' and schema {'text': str}, and implement only _process.",
            "The service module must be imported from src/services/__init__.py or it never registers.",
            "Validation errors must surface as ServiceInputError via the base class, not ad-hoc raises.",
            "All tests pass with uv run pytest tests tests_hidden; the grading tests are hidden, so conventions must be followed without a feedback loop.",
        ),
    ),
    "problem-20-limiter-follow-ups": Problem(
        identifier="problem-20-limiter-follow-ups",
        title="Rate Limiter Fix with Follow-Up Turns",
        kind="repair",
        difficulty="hard",
        skills=("debugging", "multi-turn", "api-design"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "A token-bucket rate limiter test fails: an idle bucket accumulates "
            "tokens past its capacity. Diagnose and fix the bug in src/limiter.py "
            "with the smallest safe change."
        ),
        turns=(
            "Now extend the limiter API: allow() should accept a cost parameter "
            "(defaulting to 1) so a single request can spend several tokens — a "
            "request that cannot be paid in full is denied and spends nothing — and "
            "add a remaining() method that refills and then returns the current "
            "token count.",
            "Finally, in AGENT_FINAL_ANSWER.md, explain the original bug in one "
            "or two sentences and document the final public API of TokenBucket.",
        ),
        expected_behavior=(
            "Turn 1: the refill must clamp tokens at capacity (min(capacity, tokens + elapsed * rate)).",
            "Turn 2: allow(cost=1) spends cost tokens atomically; a denied request spends nothing; remaining() refills then reports tokens.",
            "The follow-up API must not break the original single-token behavior.",
            "Turn 3: AGENT_FINAL_ANSWER.md explains the bug and documents the API.",
            "All tests pass with uv run pytest tests tests_hidden; the follow-up API is graded by hidden tests.",
        ),
    ),
    "problem-21-js-eval-aggregate": Problem(
        identifier="problem-21-js-eval-aggregate",
        title="JS Eval Aggregation Repair",
        kind="repair",
        difficulty="hard",
        skills=("javascript", "data-aggregation", "debugging", "edge-cases"),
        verify_command=("node", "--test", "tests/*.test.js", "tests_hidden/*.test.js"),
        task_prompt=(
            "This JavaScript module (src/aggregate.js) aggregates eval results and "
            "detects run-to-run regressions. Tests fail (run them with: node --test "
            "tests). Read the contract in the module's JSDoc header — skipped "
            "records (passed === null) have precise semantics — and fix the module "
            "so the full contract holds. Explain the root causes."
        ),
        expected_behavior=(
            "The agent should run node --test to see the failures and read the JSDoc contract.",
            "aggregateEval must exclude passed === null records from total, passed, passRate, and meanScore.",
            "findFlips must require an actual pass in base and an explicit fail (passed === false) in candidate; skipped or missing records are not flips.",
            "All tests pass with node --test tests/*.test.js tests_hidden/*.test.js, including hidden edge cases.",
        ),
    ),
    "problem-22-implement-jsonl-support": Problem(
        identifier="problem-22-implement-jsonl-support",
        title="Implement: JSONL Support",
        kind="implement",
        difficulty="hard",
        skills=("implementation", "conventions", "error-handling"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "Implement the following change in this repository: jsonl support"
        ),
        expected_behavior=(
            "The agent should read src/datakit/loaders.py and errors.py to recover the loader conventions (EXT_LOADERS registry, RecordParseError semantics).",
            "A .jsonl loader must be registered in EXT_LOADERS and parse one JSON record per line.",
            "Blank lines must be skipped; a malformed line must raise RecordParseError whose message includes the path and the 1-based line number.",
            "Existing .json behavior must be unchanged.",
            "All tests pass with uv run pytest tests tests_hidden; the grading tests are hidden.",
        ),
    ),
    "problem-23-implement-retry-backoff": Problem(
        identifier="problem-23-implement-retry-backoff",
        title="Implement: Retry with Exponential Backoff",
        kind="implement",
        difficulty="hard",
        skills=("implementation", "conventions", "resilience"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "Implement the following change in this repository: retry transient "
            "failures with exponential backoff"
        ),
        expected_behavior=(
            "The agent should recover the contract from config.py (MAX_ATTEMPTS is total attempts, BACKOFF_BASE doubles per retry) and errors.py (only TransientError is retryable).",
            "ApiClient.send must retry TransientError up to MAX_ATTEMPTS total attempts, sleeping BACKOFF_BASE then doubling between attempts.",
            "Non-transient ApiError must not be retried; the success path must not sleep.",
            "After MAX_ATTEMPTS transient failures the last TransientError propagates.",
            "All tests pass with uv run pytest tests tests_hidden; the grading tests are hidden and measure real call counts and delays.",
        ),
    ),
    "problem-24-implement-dry-run": Problem(
        identifier="problem-24-implement-dry-run",
        title="Implement: Cleanup --dry-run Flag",
        kind="implement",
        difficulty="hard",
        skills=("implementation", "cli", "conventions"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "Implement the following change in this repository: add --dry-run"
        ),
        expected_behavior=(
            "The agent should find the roadmap contract in README.md: --dry-run prints 'would delete <path>' per candidate, deletes nothing, exits 0.",
            "Selection and ordering must match a real run (the policy's sorted candidates).",
            "The real (non-dry-run) path must keep deleting and printing 'deleted <path>'.",
            "The flag must be wired through the argparse CLI in src/janitor/cli.py.",
            "All tests pass with uv run pytest tests tests_hidden; the grading tests are hidden.",
        ),
    ),
    "problem-25-implement-parse-lineno": Problem(
        identifier="problem-25-implement-parse-lineno",
        title="Implement: Parse Errors with Line Numbers",
        kind="implement",
        difficulty="hard",
        skills=("implementation", "conventions", "error-handling"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "Implement the following change in this repository: report line "
            "numbers in parse errors"
        ),
        expected_behavior=(
            "The agent should recover the contract from src/logkit/errors.py: 1-based line numbers counting every physical line, str(error) exactly 'line {lineno}: {reason}', int lineno attribute, bare reason attribute, first invalid line wins.",
            "ParseError construction and the parser's raise sites must both change consistently.",
            "Existing parsing behavior (blank/comment skipping, last-assignment-wins, value whitespace handling) must not regress.",
            "All tests pass with uv run pytest tests tests_hidden; the grading tests are hidden and check the exact contract.",
        ),
    ),
    "problem-26-lru-pin-revision": Problem(
        identifier="problem-26-lru-pin-revision",
        title="LRU Cache Pinning Contract Repair",
        kind="repair",
        difficulty="hard",
        skills=("debugging", "invariants", "data-structures"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "The pinnable LRU cache in src/lrupin.py violates its documented "
            "contract: a visible test shows a pinned entry being evicted. Read "
            "the class docstring, find every place the implementation deviates "
            "from the contract, and fix them with minimal changes."
        ),
        expected_behavior=(
            "The agent should fix eviction to skip pinned keys and take the least recently used unpinned key.",
            "put() overwriting an existing key must never evict (the buggy code evicts before checking membership).",
            "Inserting into a fully pinned cache must raise RuntimeError('all entries pinned') and leave the cache unchanged.",
            "Recency semantics (get refreshes, pin/unpin do not) must be preserved.",
            "All tests pass with uv run pytest tests tests_hidden; hidden tests grade the full contract.",
        ),
    ),
    "problem-27-implement-rates-section": Problem(
        identifier="problem-27-implement-rates-section",
        title="Implement: Report Rates Section",
        kind="implement",
        difficulty="hard",
        skills=("implementation", "conventions", "formatting"),
        verify_command=("uv", "run", "pytest", "tests", "tests_hidden"),
        task_prompt=(
            "Implement the following change in this repository: add the rates "
            "section to the summary report"
        ),
        expected_behavior=(
            "The agent should find the spec in docs/REPORT_FORMAT.md and follow the section conventions in src/reportkit/sections.py.",
            "A section_rates function must be registered in SECTIONS immediately after counts.",
            "Rates are computed over non-skip records; unknown statuses count as errors; formatting is exactly one decimal plus '%'.",
            "With zero attempted records both lines must read 'n/a'.",
            "All tests pass with uv run pytest tests tests_hidden; hidden tests grade exact output strings and registration order.",
        ),
    ),
    "problem-14-agent-eval-suite": Problem(
        identifier="problem-14-agent-eval-suite",
        title="Meta: Design an Agent Eval Suite",
        kind="comprehension",
        difficulty="hard",
        skills=("evaluation-design", "benchmarking", "code-reading", "technical-writing"),
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "Using the trace-trek overview and the eval-suite design notes in the source/ "
            "directory, design an evaluation suite that measures a coding agent. This is meta: "
            "you are describing the harness that would evaluate an agent like you, as the eval "
            "stage of trace-trek's collect -> train -> eval -> deploy loop. Your answer must cover: "
            "(1) the task taxonomy (repair vs comprehension) and how each kind is verified "
            "(pytest vs concept checks), plus rubric/judge scoring beyond raw test counts; "
            "(2) the isolated per-run lifecycle (prepare -> agent -> capture) including the git "
            "baseline and captured diff; (3) how the suite resists overfitting/gaming via hidden "
            "tests; (4) integrity validation (failing baseline + golden fix); and (5) scoring, "
            "aggregation into a pass rate, normalized token-usage capture, and what makes the "
            "suite reproducible. Write your answer to AGENT_FINAL_ANSWER.md."
        ),
        expected_behavior=(
            "The agent should define the repair and comprehension task kinds and their verification (pytest vs concept checks on AGENT_FINAL_ANSWER.md).",
            "The agent should describe the isolated prepare -> agent -> capture lifecycle with a git baseline commit and captured diff.",
            "The agent should explain hidden tests as overfit/gaming resistance (withheld during prepare, injected at grading, excluded from the diff).",
            "The agent should describe integrity validation: repair baselines must fail and a golden fix must pass.",
            "The agent should cover scoring beyond pass rate (rubric/judge), aggregation into pass rate, normalized token usage, and reproducibility.",
            "All tests in tests/test_answer.py should pass with uv run pytest tests.",
        ),
    ),
}


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    if not slug:
        raise ValueError("run name must contain at least one alphanumeric character")
    return slug


def run_command(command, cwd):
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env)


def checked_run(command, cwd):
    result = run_command(command, cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed in {cwd}: {' '.join(command)}\n{result.stdout}{result.stderr}"
        )
    return result


HIDDEN_TESTS_DIRNAME = "tests_hidden"


def copy_problem(source, destination, include_hidden_tests=True):
    def ignore(_directory, names):
        ignored = {".DS_Store", ".git", ".pytest_cache", ".venv", "__pycache__"}
        if not include_hidden_tests:
            ignored.add(HIDDEN_TESTS_DIRNAME)
        return {name for name in names if name in ignored}

    shutil.copytree(source, destination, ignore=ignore)


def inject_hidden_tests(source_problem_dir, workspace):
    """Copy a problem's hidden tests into a workspace for grading.

    Hidden tests are withheld from the agent during prepare and only added at
    capture/validate time so a fix cannot be overfit to the visible suite.
    Returns the injected destination path, or None when no hidden tests exist.
    """
    hidden_source = source_problem_dir / HIDDEN_TESTS_DIRNAME
    if not hidden_source.is_dir():
        return None
    destination = workspace / HIDDEN_TESTS_DIRNAME
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(hidden_source, destination)
    return destination


def task_prompt_text(problem):
    return (
        problem.task_prompt
        + "\n\nAt the end, write your concise final answer to AGENT_FINAL_ANSWER.md in this workspace."
    )


def default_usage_record():
    return {
        "schema_version": 1,
        "harness": None,
        "model": None,
        "source": "not_recorded",
        "exact": False,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cached_input_tokens": None,
        "reasoning_output_tokens": None,
        "raw_usage": None,
        "notes": "Overwrite this file with exact CLI usage when running the harness.",
    }


def ensure_usage_file(artifacts):
    usage_path = artifacts / "usage.json"
    if not usage_path.exists():
        usage_path.write_text(json.dumps(default_usage_record(), indent=2) + "\n", encoding="utf-8")


def write_run_metadata(run_dir, problem, run_name):
    metadata = {
        "problem": problem.identifier,
        "title": problem.title,
        "kind": problem.kind,
        "difficulty": problem.difficulty,
        "skills": list(problem.skills),
        "run_name": run_name,
        "verify_command": list(problem.verify_command),
        "turns": list(problem.turns),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def prepare_run(root, problem_id, run_name):
    problem = PROBLEMS[problem_id]
    safe_run_name = slugify(run_name)
    source = root / problem.identifier
    if not source.exists():
        raise FileNotFoundError(f"problem directory not found: {source}")

    run_dir = root / "runs" / problem.identifier / safe_run_name
    workspace = run_dir / "workspace"
    artifacts = run_dir / "artifacts"
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")

    artifacts.mkdir(parents=True)
    copy_problem(source, workspace, include_hidden_tests=False)
    shutil.copy2(root / "pyproject.toml", workspace / "pyproject.toml")
    (artifacts / "task-prompt.txt").write_text(task_prompt_text(problem) + "\n", encoding="utf-8")
    ensure_usage_file(artifacts)
    (workspace / "AGENT_FINAL_ANSWER.md").write_text(
        "Write the final answer for this run here.\n",
        encoding="utf-8",
    )
    write_run_metadata(run_dir, problem, safe_run_name)

    checked_run(["git", "init", "--quiet"], workspace)
    checked_run(["git", "add", "."], workspace)
    checked_run(
        [
            "git",
            "-c",
            "user.name=agent-problem-pack",
            "-c",
            "user.email=agent-problem-pack@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "baseline",
        ],
        workspace,
    )
    return run_dir


def load_run_problem(run_dir):
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return PROBLEMS[metadata["problem"]]


def resolve_run_dir(run_dir, root):
    run_dir = Path(run_dir)
    if run_dir.is_absolute():
        return run_dir
    cwd_relative = run_dir.resolve()
    if (cwd_relative / "metadata.json").exists():
        return cwd_relative
    return (root / run_dir).resolve()


def write_combined_output(path, command, result):
    text = [
        f"$ {' '.join(command)}",
        f"exit_code={result.returncode}",
        "",
        "[stdout]",
        result.stdout.rstrip(),
        "",
        "[stderr]",
        result.stderr.rstrip(),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def build_evaluation_prompt(run_dir, problem):
    workspace = run_dir / "workspace"
    artifacts = run_dir / "artifacts"
    expected = "\n".join(f"- {item}" for item in problem.expected_behavior)
    return f"""Evaluate the agent run for {problem.identifier}.

Read these files directly:
- Final answer: {workspace / "AGENT_FINAL_ANSWER.md"}
- Diff: {artifacts / "diff.patch"}
- Git status: {artifacts / "git-status.txt"}
- Verification output: {artifacts / "verification.txt"}
- Token usage: {artifacts / "usage.json"}

Expected behavior:
{expected}

Please return:
- pass/fail
- concise reasoning
- token usage summary if available
- any partial credit notes
- any concerns about unnecessary edits
"""


def capture_run(run_dir, root=PACK_ROOT):
    run_dir = resolve_run_dir(run_dir, Path(root).resolve())
    workspace = run_dir / "workspace"
    artifacts = run_dir / "artifacts"
    problem = load_run_problem(run_dir)

    artifacts.mkdir(exist_ok=True)
    ensure_usage_file(artifacts)
    source = Path(root).resolve() / problem.identifier
    injected_hidden = inject_hidden_tests(source, workspace)
    try:
        verification = run_command(problem.verify_command, workspace)
        write_combined_output(artifacts / "verification.txt", problem.verify_command, verification)
    finally:
        if injected_hidden is not None and injected_hidden.exists():
            shutil.rmtree(injected_hidden)

    from failure_analysis import summarize_verification

    answer_text = ""
    answer_path = workspace / "AGENT_FINAL_ANSWER.md"
    if answer_path.exists():
        answer_text = answer_path.read_text(encoding="utf-8")
    diff_text = ""
    try:
        checked_run(["git", "add", "-N", "."], workspace)
        diff = run_command(["git", "diff", "--no-ext-diff", "--", "."], workspace)
        diff_text = diff.stdout
    except Exception:
        diff_text = ""

    failure_summary = summarize_verification(
        (artifacts / "verification.txt").read_text(encoding="utf-8"),
        answer_text=answer_text,
        diff_text=diff_text,
    )
    (artifacts / "failure-summary.json").write_text(
        json.dumps(failure_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    checked_run(["git", "add", "-N", "."], workspace)
    diff = checked_run(["git", "diff", "--no-ext-diff", "--", "."], workspace)
    status = checked_run(["git", "status", "--short"], workspace)
    (artifacts / "diff.patch").write_text(diff.stdout, encoding="utf-8")
    (artifacts / "git-status.txt").write_text(status.stdout, encoding="utf-8")
    (artifacts / "evaluate-with-codex.md").write_text(
        build_evaluation_prompt(run_dir, problem),
        encoding="utf-8",
    )
    return verification


def catalog_problems():
    items = []
    for problem in PROBLEMS.values():
        number = int(re.search(r"problem-(\d+)", problem.identifier).group(1))
        items.append(
            {
                "id": problem.identifier,
                "number": number,
                "name": problem.title,
                "slug": problem.identifier.replace(f"problem-{number:02d}-", "").replace("-", " "),
                "kind": problem.kind,
                "difficulty": problem.difficulty,
                "skills": list(problem.skills),
                "verify_command": list(problem.verify_command),
                "task_prompt": problem.task_prompt,
                "turns": list(problem.turns),
            }
        )
    return sorted(items, key=lambda item: item["number"])


def list_filesystem_runs(root):
    runs_root = root / "runs"
    if not runs_root.is_dir():
        return []

    discovered = []
    for problem_dir in sorted(runs_root.iterdir()):
        if not problem_dir.is_dir():
            continue
        for run_dir in sorted(problem_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            verification_path = run_dir / "artifacts" / "verification.txt"
            failure_summary_path = run_dir / "artifacts" / "failure-summary.json"
            passed = None
            failure_summary = None
            if failure_summary_path.exists():
                failure_summary = json.loads(failure_summary_path.read_text(encoding="utf-8"))
                passed = failure_summary.get("passed")
            elif verification_path.exists():
                from failure_analysis import summarize_verification

                verification = verification_path.read_text(encoding="utf-8")
                answer_path = run_dir / "workspace" / "AGENT_FINAL_ANSWER.md"
                diff_path = run_dir / "artifacts" / "diff.patch"
                answer_text = answer_path.read_text(encoding="utf-8") if answer_path.exists() else ""
                diff_text = diff_path.read_text(encoding="utf-8") if diff_path.exists() else ""
                failure_summary = summarize_verification(
                    verification,
                    answer_text=answer_text,
                    diff_text=diff_text,
                )
                passed = failure_summary.get("passed")
            discovered.append(
                {
                    **metadata,
                    "run_dir": str(run_dir),
                    "passed": passed,
                    "failure_summary": failure_summary,
                    "source": "pack_tools",
                }
            )
    return discovered


def list_problems():
    for item in catalog_problems():
        problem = PROBLEMS[item["id"]]
        print(
            f"{problem.identifier}: {problem.title} "
            f"[{problem.kind}, {problem.difficulty}]"
        )


def show_problem_info(problem_id):
    problem = PROBLEMS[problem_id]
    print(f"id:         {problem.identifier}")
    print(f"title:      {problem.title}")
    print(f"kind:       {problem.kind}")
    print(f"difficulty: {problem.difficulty}")
    print(f"skills:     {', '.join(problem.skills)}")
    print(f"verify:     {' '.join(problem.verify_command)}")
    print(f"directory:  {PACK_ROOT / problem.identifier}")
    print("\nTask prompt:")
    print(task_prompt_text(problem))
    print("\nExpected behavior:")
    for item in problem.expected_behavior:
        print(f"- {item}")


def count_pytest_failures(result):
    match = re.search(r"(\d+) failed", result.stdout + result.stderr)
    return int(match.group(1)) if match else 0


def validate_problem(root, problem):
    from golden import GOLDEN_FILES, apply_golden_fix

    source = root / problem.identifier
    if not source.is_dir():
        raise FileNotFoundError(f"missing problem directory: {source}")

    if problem.kind in ANSWER_KINDS:
        required = ["source", "tests"]
        missing = [name for name in required if not (source / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"{problem.identifier} missing: {', '.join(missing)}")
        return

    baseline = run_command(problem.verify_command, source)
    # Non-pytest verifiers (e.g. node --test) don't print "N failed"; a
    # failing baseline is a nonzero exit either way.
    if baseline.returncode == 0 and count_pytest_failures(baseline) == 0:
        raise AssertionError(f"{problem.identifier}: expected failing baseline tests, got 0 failures")

    if problem.identifier not in GOLDEN_FILES:
        raise KeyError(f"{problem.identifier}: repair problem has no golden fix")

    workspace = source / ".validate-tmp"
    if workspace.exists():
        shutil.rmtree(workspace)
    copy_problem(source, workspace)
    shutil.copy2(root / "pyproject.toml", workspace / "pyproject.toml")
    try:
        apply_golden_fix(problem.identifier, workspace)
        fixed = run_command(problem.verify_command, workspace)
        if fixed.returncode != 0:
            raise AssertionError(
                f"{problem.identifier}: golden fix did not pass tests\n"
                f"{fixed.stdout}\n{fixed.stderr}"
            )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def validate_pack(root):
    errors = []
    for problem in PROBLEMS.values():
        try:
            validate_problem(root, problem)
            print(f"ok  {problem.identifier}")
        except Exception as exc:
            errors.append(f"{problem.identifier}: {exc}")
            print(f"FAIL {problem.identifier}: {exc}", file=sys.stderr)
    if errors:
        raise RuntimeError(f"validation failed for {len(errors)} problem(s)")


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare and capture agent problem-pack runs.")
    parser.add_argument("--root", type=Path, default=PACK_ROOT, help=f"Pack root. Default: {PACK_ROOT}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available problem ids.")

    info_parser = subparsers.add_parser("info", help="Show metadata and rubric for one problem.")
    info_parser.add_argument("problem", choices=sorted(PROBLEMS))

    subparsers.add_parser("validate", help="Verify problem baselines and golden fixes.")

    catalog_parser = subparsers.add_parser("catalog", help="Print problem metadata as JSON.")
    catalog_parser.add_argument(
        "--filesystem-runs",
        action="store_true",
        help="Include captured runs from runs/ in the JSON output.",
    )

    prepare_parser = subparsers.add_parser("prepare", help="Create an isolated run workspace.")
    prepare_parser.add_argument("problem", choices=sorted(PROBLEMS))
    prepare_parser.add_argument("run_name")

    capture_parser = subparsers.add_parser("capture", help="Capture diff, verification output, and evaluation prompt.")
    capture_parser.add_argument("run_dir", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            list_problems()
            return 0
        if args.command == "info":
            show_problem_info(args.problem)
            return 0
        if args.command == "validate":
            validate_pack(args.root.resolve())
            return 0
        if args.command == "catalog":
            payload = {"problems": catalog_problems()}
            if args.filesystem_runs:
                payload["filesystem_runs"] = list_filesystem_runs(args.root.resolve())
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "prepare":
            run_dir = prepare_run(args.root.resolve(), args.problem, args.run_name)
            print(run_dir)
            print(run_dir / "artifacts" / "task-prompt.txt")
            return 0
        if args.command == "capture":
            result = capture_run(args.run_dir, args.root.resolve())
            print(args.run_dir / "artifacts" / "evaluate-with-codex.md")
            return result.returncode
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
