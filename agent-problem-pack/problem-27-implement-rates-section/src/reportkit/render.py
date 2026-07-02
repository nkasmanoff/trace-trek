"""Assemble the plain-text summary report from registered sections."""

from src.reportkit.sections import SECTIONS


def render_report(suite, records):
    lines = []
    for name, section in SECTIONS:
        lines.append(f"== {name} ==")
        lines.extend(section(suite, records))
    return "\n".join(lines) + "\n"
