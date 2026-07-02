"""Event-log audit.

Reads data/events.jsonl (relative to this file) and reports audit statistics.

A record is VALID when:
- "id" is a non-empty string
- "category" is a string
- "latency_ms" is an int/float with 0 <= latency_ms <= 10000

Strict mode (--strict) additionally:
- drops records whose "flags" list contains "shadow"
- deduplicates by id, keeping the FIRST occurrence in file order

The report includes an AUDIT_TOKEN derived from the ids of the records that
survived filtering, so two audits agree only if they selected exactly the
same records.
"""

import argparse
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "events.jsonl"


def load_events(path=DATA):
    events = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def is_valid(record):
    identifier = record.get("id")
    category = record.get("category")
    latency = record.get("latency_ms")
    if not isinstance(identifier, str) or not identifier:
        return False
    if not isinstance(category, str):
        return False
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        return False
    return 0 <= latency <= 10000


def select(events, strict=False):
    selected = [record for record in events if is_valid(record)]
    if strict:
        selected = [
            record
            for record in selected
            if "shadow" not in (record.get("flags") or [])
        ]
        seen = set()
        deduped = []
        for record in selected:
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            deduped.append(record)
        selected = deduped
    return selected


def audit_token(selected, strict=False):
    mode = "strict" if strict else "default"
    payload = mode + ":" + ",".join(sorted(record["id"] for record in selected))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def top_category(selected):
    counts = {}
    for record in selected:
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def mean_latency(selected):
    if not selected:
        return 0.0
    return sum(record["latency_ms"] for record in selected) / len(selected)


def build_report(strict=False, path=DATA):
    selected = select(load_events(path), strict=strict)
    return {
        "mode": "strict" if strict else "default",
        "token": audit_token(selected, strict=strict),
        "valid_records": len(selected),
        "top_category": top_category(selected),
        "mean_latency_ms": round(mean_latency(selected), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit the event log.")
    parser.add_argument("--strict", action="store_true",
                        help="drop shadow-flagged records and dedupe by id")
    args = parser.parse_args()
    report = build_report(strict=args.strict)
    print(f"mode:            {report['mode']}")
    print(f"AUDIT_TOKEN:     {report['token']}")
    print(f"valid records:   {report['valid_records']}")
    print(f"top category:    {report['top_category']}")
    print(f"mean latency ms: {report['mean_latency_ms']}")


if __name__ == "__main__":
    main()
