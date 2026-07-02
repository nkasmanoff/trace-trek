"""Hidden: grade the --dry-run flag against the roadmap contract."""

import os
import time

from src.janitor.cli import main


def make_file(path, age_seconds):
    path.write_text("x", encoding="utf-8")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))


def test_dry_run_deletes_nothing(tmp_path, capsys):
    stale_one = tmp_path / "a.tmp"
    stale_two = tmp_path / "b.log"
    make_file(stale_one, age_seconds=100)
    make_file(stale_two, age_seconds=100)

    exit_code = main([str(tmp_path), "--max-age-seconds", "50", "--dry-run"])

    assert exit_code == 0
    assert stale_one.exists()
    assert stale_two.exists()
    out = capsys.readouterr().out
    assert f"would delete {stale_one}" in out
    assert f"would delete {stale_two}" in out
    assert "deleted " not in out.replace("would delete", "")


def test_dry_run_lists_in_sorted_order(tmp_path, capsys):
    second = tmp_path / "b.tmp"
    first = tmp_path / "a.tmp"
    make_file(second, age_seconds=100)
    make_file(first, age_seconds=100)

    main([str(tmp_path), "--max-age-seconds", "50", "--dry-run"])

    out_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("would delete")
    ]
    assert out_lines == [f"would delete {first}", f"would delete {second}"]


def test_real_run_still_deletes(tmp_path, capsys):
    stale = tmp_path / "a.tmp"
    make_file(stale, age_seconds=100)

    exit_code = main([str(tmp_path), "--max-age-seconds", "50"])

    assert exit_code == 0
    assert not stale.exists()
    assert f"deleted {stale}" in capsys.readouterr().out
