"""Stage 7 Layer B (D-045): job kinds from names, rows per snapshot, the 2× rule. No AWS calls."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.monitor import pipeline_health as ph


def test_job_kind_from_launcher_and_pipeline_names():
    assert ph.job_kind("mlops-aiops-score-20261001-235814", "mlops-aiops") == "score"
    assert ph.job_kind("mlops-aiops-monitor-check-20261003-101010", "mlops-aiops") == (
        "monitor-check"
    )
    assert ph.job_kind("pipelines-clk3bbv23tm2-Train-AbC123", "mlops-aiops") == "pipeline-train"
    assert ph.job_kind("someone-elses-job", "mlops-aiops") is None


def test_rows_come_from_the_newest_snapshot_before_the_job():
    created = pd.Series(
        [datetime(2026, 9, 22, tzinfo=UTC), datetime(2026, 9, 28, tzinfo=UTC),
         datetime(2026, 10, 2, tzinfo=UTC)]
    )  # fmt: skip
    rows = ph.rows_at(created, ph.SNAPSHOT_ROWS)
    assert rows.isna().iloc[0]
    assert rows.iloc[1:].tolist() == [2_911_486, 2_921_625]


def _job(name, kind, seconds, status="Completed", day=29):
    return {
        "name": name,
        "kind": kind,
        "status": status,
        "created": datetime(2026, 9, day, tzinfo=UTC),
        "billed_seconds": seconds,
        "instance_type": "ml.t3.xlarge",
    }


def test_assess_flags_the_slow_job_and_the_failed_one():
    jobs = pd.DataFrame(
        [
            _job("a", "pipeline-train", 249),
            _job("b", "pipeline-train", 249),
            _job("c", "pipeline-train", 219),
            _job("d", "pipeline-train", 600),
            _job("e", "score", None, status="Failed"),
        ]
    )
    out = ph.assess(jobs).set_index("name")
    assert out["flag"].to_dict() == {"a": False, "b": False, "c": False, "d": True, "e": True}
    assert out.loc["e", "reason"] == "failed"
    assert out.loc["a", "usd"] > 0
    text = ph.alert_text(out.reset_index())
    assert "d (pipeline-train)" in text and "e (score): failed" in text


def test_a_bigger_snapshot_is_not_a_slow_job():
    # the same seconds per million rows on twice the data: not flagged
    jobs = pd.DataFrame(
        [_job(str(i), "score", 100) for i in range(3)] + [_job("big", "score", 200, day=30)]
    )
    out = ph.assess(jobs, {"2026-09-01": 1_000_000, "2026-09-30": 2_000_000})
    assert not out["flag"].any()
    assert ph.alert_text(out) is None


def test_kinds_that_skip_the_snapshot_are_judged_on_seconds():
    jobs = pd.DataFrame(
        [_job(str(i), "monitor-check", 280) for i in range(3)]
        + [_job("slow", "monitor-check", 700)]
    )
    out = ph.assess(jobs).set_index("name")
    assert out["sec_per_mrows"].isna().all()  # it read a CSV, not the snapshot
    assert out["flag"].to_dict() == {"0": False, "1": False, "2": False, "slow": True}
