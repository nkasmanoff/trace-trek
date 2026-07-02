"""Deletion pass over the policy's candidates."""

from src.janitor.policy import candidates


def run_cleanup(root, max_age_seconds, emit=print):
    removed = []
    for path in candidates(root, max_age_seconds):
        path.unlink()
        emit(f"deleted {path}")
        removed.append(path)
    return removed
