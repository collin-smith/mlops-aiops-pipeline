"""Stage 7 shadow grading (D-045): first score per request, the deadline-passed cohort, and
per-group recall, checked by hand on a small frame. No AWS calls."""

from __future__ import annotations

import pandas as pd
import pytest

from src.deploy import grade

SNAP = "2026-10-20"


def _row(rid, outcome, flagged, sector="EAST", deadline="2026-10-10", scored="2026-10-01"):
    return {
        "usage": "shadow-not-for-use",
        "scored_asof": scored,
        "service_request_id": rid,
        "service_name": "Pothole",
        "sector": sector,
        "srg": "ESTABLISHED",
        "flagged": flagged,
        "deadline": deadline,
        "threshold_days": 9.0,
        "category_seen": True,
        "outcome": outcome,
        "outcome_snapshot": SNAP,
    }


def test_first_score_wins_when_a_request_is_scored_twice():
    rows = pd.DataFrame([_row("A", "late", False, scored="2026-10-15"), _row("A", "late", True)])
    first = grade.first_scores(rows)
    assert len(first) == 1 and first.loc[0, "flagged"]


def test_cohort_is_deadline_passed_only_and_fully_decided():
    rows = pd.DataFrame(
        [
            _row("A", "late", True),
            _row("B", "on_time", False),
            # closed on time, deadline still ahead: decided, but outside the cohort
            _row("C", "on_time", False, deadline="2026-12-01"),
            _row("D", "undecided", True, deadline="2026-12-01"),
        ]
    )
    assert grade.deadline_passed(rows)["service_request_id"].tolist() == ["A", "B"]
    broken = pd.DataFrame([_row("E", "undecided", False)])
    with pytest.raises(ValueError, match="undecided"):
        grade.deadline_passed(broken)


def test_grade_rates_and_recall_ratio_by_hand():
    rows = []
    # EAST: 40 late (30 flagged), 60 on time (10 flagged)
    rows += [_row(f"E{i}", "late", i < 30, "EAST") for i in range(40)]
    rows += [_row(f"e{i}", "on_time", i < 10, "EAST") for i in range(60)]
    # WEST: 40 late (20 flagged), 60 on time (0 flagged)
    rows += [_row(f"W{i}", "late", i < 20, "WEST") for i in range(40)]
    rows += [_row(f"w{i}", "on_time", False, "WEST") for i in range(60)]
    # outside the cohort: deadline ahead
    rows += [_row("X", "undecided", True, "WEST", deadline="2027-01-01")]
    report = grade.grade(pd.DataFrame(rows))

    assert report["rows_scored"] == 201 and report["rows_graded"] == 200
    g = report["graded"]
    assert g["late_rate"] == 0.4
    assert g["late_rate_flagged"] == round(50 / 60, 4)
    assert g["recall"] == round(50 / 80, 4)
    assert g["lift"] == round((50 / 60) / 0.4, 4)
    sector = report["dimensions"]["sector"]
    assert sector["recall_ratio"] == round(0.5 / 0.75, 4)
    assert sector["lowest_recall_group"] == "WEST"
    assert report["usage"] == "shadow-not-for-use"
    assert any("recall ratio" in line for line in grade.summary_lines(report))


def test_small_groups_are_listed_but_not_compared():
    rows = [_row(f"E{i}", "late", i % 2 == 0, "EAST") for i in range(40)]
    rows += [_row(f"C{i}", "late", True, "CENTRE") for i in range(5)]
    report = grade.grade(pd.DataFrame(rows))
    sector = report["dimensions"]["sector"]
    assert sector["recall_ratio"] is None  # only EAST has enough late requests
    assert {g["group"]: g["compared"] for g in sector["groups"]} == {
        "CENTRE": False,
        "EAST": True,
    }
