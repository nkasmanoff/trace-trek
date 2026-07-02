"""Stage 3: aggregate raw results into a weighted suite score.

The suite score is the WEIGHTED mean of task scores:

    weighted_score = sum(score_i * weight_i) / sum(weight_i)

computed over tasks whose weight is strictly positive. Tasks with weight 0.0
are excluded from BOTH the numerator and the denominator, but still counted
in the `recorded` field. If no task has positive weight, weighted_score is
0.0.

Returned summary:

    {"recorded": <all results>, "scored": <positive-weight results>,
     "weighted_score": <float>}
"""


def aggregate(results):
    total = 0.0
    for result in results:
        total += result["score"] * result["weight"]
    weighted_score = total / len(results) if results else 0.0
    scored = [r for r in results if r["weight"] > 0]
    return {
        "recorded": len(results),
        "scored": len(scored),
        "weighted_score": weighted_score,
    }
