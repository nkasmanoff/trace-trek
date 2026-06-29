"""Render aggregated buckets into a stable, human-readable summary."""


def format_summary(summaries):
    lines = []
    for bucket in summaries:
        lines.append(
            f"{bucket['run']}/{bucket['type']}: "
            f"{bucket['passed']}/{bucket['total']} passed "
            f"({bucket['pass_rate'] * 100:.1f}%), "
            f"mean_score={bucket['mean_score']:.3f}"
        )
    return "\n".join(lines)
