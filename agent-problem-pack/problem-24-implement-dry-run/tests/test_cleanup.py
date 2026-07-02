import os
import time

from src.janitor.cleanup import run_cleanup


def make_file(path, age_seconds):
    path.write_text("x", encoding="utf-8")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))


def test_deletes_only_stale_matching_files(tmp_path, capsys):
    stale = tmp_path / "a.tmp"
    fresh = tmp_path / "b.tmp"
    other = tmp_path / "c.txt"
    make_file(stale, age_seconds=100)
    make_file(fresh, age_seconds=1)
    make_file(other, age_seconds=100)

    removed = run_cleanup(tmp_path, max_age_seconds=50)

    assert removed == [stale]
    assert not stale.exists()
    assert fresh.exists()
    assert other.exists()
    assert f"deleted {stale}" in capsys.readouterr().out
