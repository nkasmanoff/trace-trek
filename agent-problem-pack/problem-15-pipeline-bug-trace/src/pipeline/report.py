"""Stage 4: render a one-line summary and flag regressions.

A run regresses when its weighted score drops more than `tolerance` below
the baseline's weighted score.
"""


def render_summary(summary):
    return (
        f"scored {summary['scored']}/{summary['recorded']} tasks, "
        f"weighted score {summary['weighted_score']:.3f}"
    )


def regressed(current, baseline, tolerance=0.02):
    return (baseline["weighted_score"] - current["weighted_score"]) > tolerance
