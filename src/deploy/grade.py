"""Grade the shadow scores on what actually happened (Stage 7, D-045).

The input is ``shadow_outcomes`` (``src/deploy/views.py``): each score joined to the latest
snapshot and called ``on_time``, ``late`` or ``undecided``.

**Outcomes don't arrive evenly.** At any later snapshot a scored request has an outcome only
if it closed or passed its deadline. On-time outcomes arrive early (closed inside the
deadline); late ones only once the deadline has passed. Grading every decided row would
count the early on-time closures and miss the late ones still to come, so the late rate
would look lower than it is. So the grade uses only the **cohort whose deadline passed by
the outcome snapshot**. Inside it every row is decided (closed on time, closed late, or open
past the deadline), whatever its arrival order. The price: the cohort leans towards
short-deadline request types, and the report says how far.

A request still open can be scored by more than one run; it's graded once, on its first
score. The ratios use the gate's definition (min recall / max recall across groups,
``fairness.group_metrics``), but this is a report, not the gate: a grade never promotes
the shadow model (D-042, D-044).
"""

from __future__ import annotations

import math

import pandas as pd

from src.deploy.score import USAGE
from src.promote import fairness

DIMENSIONS = ("sector", "srg")
# Smaller groups are listed but left out of the ratio: a cohort of a few weeks is far
# smaller than a test year, so the gate's 1,000-row minimum would compare nothing.
MIN_COMPARED_POSITIVES = 30


def _r(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 4) if isinstance(v, float) else v


def first_scores(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Each request once, on its earliest score."""
    return (
        outcomes.sort_values(["scored_asof", "service_request_id"])
        .drop_duplicates("service_request_id", keep="first")
        .reset_index(drop=True)
    )


def deadline_passed(outcomes: pd.DataFrame) -> pd.DataFrame:
    """The rows whose deadline passed before the outcome snapshot: all of them decided."""
    deadline = pd.to_datetime(outcomes["deadline"], utc=True)
    taken = pd.to_datetime(outcomes["outcome_snapshot"], utc=True)
    cohort = outcomes.loc[deadline < taken]
    undecided = int((cohort["outcome"] == "undecided").sum())
    if undecided:
        raise ValueError(f"{undecided} rows past their deadline are still undecided")
    return cohort.reset_index(drop=True)


def _rates(frame: pd.DataFrame) -> dict:
    late = frame["outcome"] == "late"
    flagged = frame["flagged"].astype(bool)
    base = late.mean() if len(frame) else float("nan")
    precision = late[flagged].mean() if flagged.any() else float("nan")
    return {
        "rows": len(frame),
        "late_rate": _r(base),
        "flagged": int(flagged.sum()),
        "late_rate_flagged": _r(precision),
        "late_rate_unflagged": _r(late[~flagged].mean() if (~flagged).any() else float("nan")),
        "lift": _r(precision / base if base else float("nan")),
        "recall": _r((late & flagged).sum() / late.sum() if late.any() else float("nan")),
    }


def _groups(frame: pd.DataFrame, dim: str) -> dict:
    df = frame.assign(breach=(frame["outcome"] == "late").astype(int))
    g = fairness.group_metrics(df, dim, df["flagged"].astype(bool))
    compared = g.loc[
        (g["positives"] >= MIN_COMPARED_POSITIVES)
        & (g["group"] != fairness.UNKNOWN)
        & ~g["group"].isin(fairness.REPORT_ONLY.get(dim, frozenset()))
    ]
    ratio = worst = None
    if len(compared) >= 2 and compared["recall"].max() > 0:
        ratio = compared["recall"].min() / compared["recall"].max()
        worst = str(compared.loc[compared["recall"].idxmin(), "group"])
    g["compared"] = g["group"].isin(compared["group"])
    return {
        "recall_ratio": _r(ratio),
        "lowest_recall_group": worst,
        "groups": [{k: _r(v) for k, v in row.items()} for row in g.to_dict(orient="records")],
    }


def grade(outcomes: pd.DataFrame) -> dict:
    """The shadow grade: coverage, rates in the deadline-passed cohort, and per-group recall."""
    scored = first_scores(outcomes)
    cohort = deadline_passed(scored)
    seen = cohort["category_seen"].astype(bool)
    days = pd.to_numeric(scored["threshold_days"])
    cohort_days = pd.to_numeric(cohort["threshold_days"])
    return {
        "usage": USAGE,
        "scored_asof": sorted(scored["scored_asof"].astype(str).unique().tolist()),
        "outcome_snapshot": str(pd.to_datetime(scored["outcome_snapshot"]).max().date()),
        "rows_scored": len(scored),
        "rows_graded": len(cohort),
        "coverage": _r(len(cohort) / len(scored) if len(scored) else float("nan")),
        "outcomes_all_scored": scored["outcome"].value_counts().to_dict(),
        "deadline_days_median": {
            "all_scored": _r(float(days.median())),
            "graded": _r(float(cohort_days.median())) if len(cohort) else None,
        },
        "graded": _rates(cohort.loc[seen]),
        "graded_unseen_categories": _rates(cohort.loc[~seen]),
        "dimensions": {dim: _groups(cohort.loc[seen], dim) for dim in DIMENSIONS},
        "top_graded_types": cohort["service_name"].value_counts().head(10).to_dict(),
    }


def summary_lines(report: dict) -> list[str]:
    g = report["graded"]
    lines = [
        f"usage={report['usage']}  scored {report['scored_asof']}  "
        f"outcomes as of {report['outcome_snapshot']}",
        f"graded {report['rows_graded']:,} of {report['rows_scored']:,} "
        f"({report['coverage']:.0%}); median deadline {report['deadline_days_median']}",
        f"late: flagged {g['late_rate_flagged']}, unflagged {g['late_rate_unflagged']}, "
        f"all {g['late_rate']}; lift {g['lift']}; recall {g['recall']}",
    ]
    for dim, d in report["dimensions"].items():
        worst = d["lowest_recall_group"]
        lines.append(f"  {dim}: recall ratio {d['recall_ratio']} (lowest {worst})")
        for row in d["groups"]:
            mark = "" if row["compared"] else "  (too few late to compare)"
            lines.append(
                f"    {row['group']:<13} rows {row['rows']:>6,}  late {row['positives']:>5,}  "
                f"recall {row['recall']}  lift {row['lift']}{mark}"
            )
    return lines
