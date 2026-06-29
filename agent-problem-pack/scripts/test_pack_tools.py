import importlib.util
import pathlib
import shutil
import subprocess


SCRIPT_PATH = pathlib.Path(__file__).with_name("pack_tools.py")
PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_pack_tools():
    spec = importlib.util.spec_from_file_location("pack_tools", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_pack_without_runs(destination):
    def ignore(directory, names):
        ignored = {"runs", ".DS_Store", ".pytest_cache", ".venv", "__pycache__"}
        return {name for name in names if name in ignored}

    shutil.copytree(PACK_ROOT, destination, ignore=ignore)


def test_prepare_run_creates_isolated_workspace_and_prompt_file(tmp_path):
    tools = load_pack_tools()
    root = tmp_path / "agent-problem-pack"
    copy_pack_without_runs(root)

    run_dir = tools.prepare_run(root, "problem-01-tokenizer-regression", "codex-test")

    assert (run_dir / "workspace" / "tokenizer.py").exists()
    assert (run_dir / "artifacts" / "task-prompt.txt").exists()
    assert (run_dir / "artifacts" / "usage.json").exists()
    assert (run_dir / "workspace" / "pyproject.toml").exists()
    assert (run_dir / "workspace" / "AGENT_FINAL_ANSWER.md").exists()
    assert (run_dir / "workspace" / ".git").exists()

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=run_dir / "workspace",
        text=True,
        capture_output=True,
        check=True,
    )
    assert status.stdout.strip() == ""


def test_prepare_excludes_hidden_tests_but_capture_injects_them(tmp_path):
    tools = load_pack_tools()
    root = tmp_path / "agent-problem-pack"
    copy_pack_without_runs(root)

    run_dir = tools.prepare_run(root, "problem-11-eval-scoring-pipeline", "hidden-test")
    workspace = run_dir / "workspace"

    assert (workspace / "tests").is_dir()
    assert not (workspace / "tests_hidden").exists()

    result = tools.capture_run(run_dir, root)

    assert result.returncode != 0
    assert not (workspace / "tests_hidden").exists()
    diff = (run_dir / "artifacts" / "diff.patch").read_text(encoding="utf-8")
    assert "tests_hidden" not in diff
    verification = (run_dir / "artifacts" / "verification.txt").read_text(encoding="utf-8")
    assert "tests_hidden" in verification


def test_verify_commands_use_uv_run_pytest():
    tools = load_pack_tools()

    for problem in tools.PROBLEMS.values():
        assert problem.verify_command[:3] == ("uv", "run", "pytest")
        for expected in problem.expected_behavior:
            assert "unit" + "test" not in expected
            assert "python" + "3" not in expected


def test_capture_run_writes_diff_verification_and_evaluation_prompt(tmp_path):
    tools = load_pack_tools()
    root = tmp_path / "agent-problem-pack"
    copy_pack_without_runs(root)
    run_dir = tools.prepare_run(root, "problem-01-tokenizer-regression", "claude-test")
    workspace = run_dir / "workspace"

    (workspace / "tokenizer.py").write_text(
        "def tokenize(text):\n"
        "    return [part.lower() for part in text.strip().split(',') if part]\n",
        encoding="utf-8",
    )
    (workspace / "AGENT_FINAL_ANSWER.md").write_text(
        "Filtered empty split parts so blank input returns no tokens.\n",
        encoding="utf-8",
    )

    result = tools.capture_run(run_dir)

    assert result.returncode == 0
    assert "tokenizer.py" in (run_dir / "artifacts" / "diff.patch").read_text(encoding="utf-8")
    assert "2 passed" in (run_dir / "artifacts" / "verification.txt").read_text(encoding="utf-8")
    evaluation_prompt = (run_dir / "artifacts" / "evaluate-with-codex.md").read_text(encoding="utf-8")
    assert "Evaluate the agent run" in evaluation_prompt
    assert "AGENT_FINAL_ANSWER.md" in evaluation_prompt
    assert "diff.patch" in evaluation_prompt
    assert "usage.json" in evaluation_prompt


def test_capture_run_accepts_pack_root_relative_run_dir(tmp_path):
    tools = load_pack_tools()
    root = tmp_path / "agent-problem-pack"
    copy_pack_without_runs(root)
    run_dir = tools.prepare_run(root, "problem-01-tokenizer-regression", "qwen-test")
    workspace = run_dir / "workspace"

    (workspace / "tokenizer.py").write_text(
        "def tokenize(text):\n"
        "    return [part.lower() for part in text.strip().split(',') if part]\n",
        encoding="utf-8",
    )

    relative_run_dir = pathlib.Path("runs/problem-01-tokenizer-regression/qwen-test")

    result = tools.capture_run(relative_run_dir, root)

    assert result.returncode == 0
    assert (run_dir / "artifacts" / "evaluate-with-codex.md").exists()


def test_all_problems_registered_with_metadata():
    tools = load_pack_tools()

    assert len(tools.PROBLEMS) == 14
    for problem in tools.PROBLEMS.values():
        assert problem.kind in {"repair", "comprehension"}
        assert problem.difficulty in {"easy", "medium", "hard"}
        assert problem.skills
        assert (tools.PACK_ROOT / problem.identifier).is_dir()


def test_validate_pack_passes():
    tools = load_pack_tools()
    root = pathlib.Path(tools.PACK_ROOT)

    tools.validate_pack(root)


def test_golden_fixes_registered_for_all_repair_problems():
    tools = load_pack_tools()
    golden = load_golden()

    repair_ids = {
        pid for pid, problem in tools.PROBLEMS.items() if problem.kind == "repair"
    }
    assert repair_ids == set(golden.GOLDEN_FILES)


def load_golden():
    spec = importlib.util.spec_from_file_location(
        "golden", SCRIPT_PATH.with_name("golden.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_lists_all_problems():
    tools = load_pack_tools()

    items = tools.catalog_problems()
    assert len(items) == 14
    assert items[0]["number"] == 1
    assert items[-1]["number"] == 14
    assert all(item["kind"] in {"repair", "comprehension"} for item in items)


def test_failure_summary_from_verification():
    tools = load_pack_tools()
    failure = load_failure_analysis()

    text = """$ uv run pytest tests
exit_code=1

[stdout]
FAILED tests/test_answer.py::test_covers_prepare_stage - AssertionError: Answer should describe the prepare step
"""
    summary = failure.summarize_verification(
        text,
        answer_text="Write the final answer for this run here.\n",
        diff_text="",
        passed=False,
    )
    assert summary["passed"] is False
    assert "prepare step" in summary["headline"].lower() or "final answer" in summary["headline"].lower()
    assert summary["failure_count"] >= 1

    pass_text = """$ uv run pytest
exit_code=0

[stdout]
5 passed in 0.01s
"""
    pass_summary = failure.summarize_verification(pass_text)
    assert pass_summary["passed"] is True
    assert pass_summary["headline"] is None


def load_failure_analysis():
    spec = importlib.util.spec_from_file_location(
        "failure_analysis", SCRIPT_PATH.with_name("failure_analysis.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pyproject_declares_pytest_dependency():
    pyproject = (PACK_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "pytest" in pyproject
