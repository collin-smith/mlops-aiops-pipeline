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


def _scored(rid, outcome, score, sector, srg, scored="2026-10-01", deadline="2026-10-10"):
    row = _row(rid, outcome, False, sector, deadline=deadline, scored=scored)
    return {**row, "srg": srg, "score": score}


def _with_sector_flags(rows):
    """Set ``flagged`` the way the scoring job does: top 10% within each run's sector."""
    df = pd.DataFrame(rows)
    df["flagged"] = grade.reflag(df, ("sector",))
    return df


def test_reflag_is_per_run_and_per_cell():
    # one run, one sector, two growth classes of 10 rows each; scores 0..19, ESTABLISHED high
    rows = [_scored(f"E{i}", "late", 10 + i, "EAST", "ESTABLISHED") for i in range(10)]
    rows += [_scored(f"D{i}", "late", i, "EAST", "DEVELOPING") for i in range(10)]
    # a second run with its own top scores: it must not shift the first run's cut
    rows += [
        _scored(f"L{i}", "late", 100 + i, "EAST", "ESTABLISHED", scored="2026-10-08")
        for i in range(10)
    ]
    df = pd.DataFrame(rows)
    by_sector = df.loc[grade.reflag(df, ("sector",)), "service_request_id"].tolist()
    by_cell = df.loc[grade.reflag(df, grade.POST_HOC_QUOTA), "service_request_id"].tolist()
    assert by_sector == ["E8", "E9", "L9"]  # the run's top 10% of 20 rows: 2 rows
    assert by_cell == ["E9", "D9", "L9"]  # one per cell of 10


def test_post_hoc_quota_reported_beside_the_scoring_jobs_grade():
    # EAST: established requests score high and are all on time; developing ones are late
    rows = [_scored(f"E{i}", "on_time", 50 + i, "EAST", "ESTABLISHED") for i in range(100)]
    rows += [_scored(f"D{i}", "late", i, "EAST", "DEVELOPING") for i in range(100)]
    report = grade.grade(_with_sector_flags(rows))

    assert report["graded"]["flagged"] == 20 and report["graded"]["recall"] == 0.0
    alt = report["post_hoc_quota"]
    assert alt["decision"] == "D-047" and alt["quota_by"] == "sector x srg"
    assert alt["sector_reflag_mismatches"] == 0
    assert alt["graded"]["flagged"] == 20  # 10 per growth class
    assert alt["graded"]["recall"] == 0.1  # the developing cell's top 10 are all late
    lines = grade.summary_lines(report)
    assert any(line.startswith("post hoc (D-047)") for line in lines)


def test_reflag_mismatch_is_counted():
    rows = [_scored(f"E{i}", "late", i, "EAST", "ESTABLISHED") for i in range(40)]
    df = _with_sector_flags(rows)
    df.loc[0, "flagged"] = True  # a flag the stored scores can't explain
    assert grade.grade(df)["post_hoc_quota"]["sector_reflag_mismatches"] == 1


def test_no_score_column_means_no_post_hoc_section():
    rows = [_row(f"E{i}", "late", i < 4, "EAST") for i in range(40)]
    assert "post_hoc_quota" not in grade.grade(pd.DataFrame(rows))
