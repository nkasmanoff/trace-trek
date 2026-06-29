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

    if problem.kind == "comprehension":
        required = ["source", "tests"]
        missing = [name for name in required if not (source / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"{problem.identifier} missing: {', '.join(missing)}")
        return

    baseline = run_command(problem.verify_command, source)
    failures = count_pytest_failures(baseline)
    if failures == 0:
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
