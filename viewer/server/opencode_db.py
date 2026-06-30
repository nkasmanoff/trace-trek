#!/usr/bin/env python3
"""Read-only queries against opencode.db for the trace viewer.

Uses Python's sqlite3 with mode=ro + busy_timeout so large DB reads don't
fight the sqlite3 CLI's zero busy_timeout or block the Node event loop
indefinitely when opencode is writing.
"""
from __future__ import annotations

import json
import sqlite3
import sys


def query(db_path: str, sql: str) -> list[dict]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=60000")
    try:
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db_path")
    ap.add_argument("--out", help="write JSON result to this file instead of stdout")
    args = ap.parse_args()
    sql = sys.stdin.read()
    payload = json.dumps(query(args.db_path, sql)) if sql.strip() else "[]"
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)
