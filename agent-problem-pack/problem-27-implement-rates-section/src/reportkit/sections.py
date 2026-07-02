"""Report sections.

Every section is a function ``(suite, records) -> list[str]`` returning the
section's body lines (without the banner). Sections are registered in
``SECTIONS`` as ``(name, function)`` pairs; the renderer emits them in list
order. See docs/REPORT_FORMAT.md for the exact output contract.
"""

from collections import Counter

KNOWN_STATUSES = ("pass", "fail", "error", "skip")


def section_header(suite, records):
    return [f"suite: {suite}", f"records: {len(records)}"]


def section_counts(suite, records):
    counts = Counter()
    for record in records:
        status = record.get("status")
        counts[status if status in KNOWN_STATUSES else "error"] += 1
    return [f"{status}: {counts[status]}" for status in KNOWN_STATUSES]


SECTIONS = [
    ("header", section_header),
    ("counts", section_counts),
]
