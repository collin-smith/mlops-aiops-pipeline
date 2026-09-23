"""Data-validation gate (Stage 3). Each test breaks the fixture in one way a real pull can
break, and checks that the gate notices. No AWS calls."""

import json

import pandas as pd
import pytest

from src.pipeline.validate import (
    REPORT_NAME,
    DataValidationError,
    Thresholds,
    build_report,
    main,
    validate,
)

# The fixture is 400 rows, so scale the volume and group-size floors down to match.
SMALL = Thresholds(min_rows=100, group_min_rows=20, bulk_close_min_rows=10)


def _asof(df: pd.DataFrame) -> pd.Timestamp:
    return pd.to_datetime(df["requested_date"], utc=True).max().normalize()


def _results(df, **kw):
    return {r.name: r for r in validate(df, asof=_asof(df), thresholds=SMALL, **kw)}


def _with_montgomery(df: pd.DataFrame) -> pd.DataFrame:
    """Add a community whose potholes were closed in one backfill: most of its requests
    are shut on a single day, months after they came in (the exploratory pass saw a
    196-day mean for Montgomery)."""
    rows = []
    close_day = pd.Timestamp("2025-03-01", tz="UTC")
    for i in range(60):
        req = close_day - pd.Timedelta(days=150 + i * 2)
        rows.append(
            {
                "service_request_id": f"SR-M{i:04d}",
                "requested_date": req.isoformat(),
                "closed_date": close_day.isoformat(),
                "updated_date": close_day.isoformat(),
                "status_description": "Closed",
                "service_name": "Pothole Repair",
                "agency_responsible": "Roads",
                "comm_name": "MONTGOMERY",
                "comm_code": "MON",
                "source": "Phone",
                "longitude": -114.15,
                "latitude": 51.07,
            }
        )
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def test_clean_fixture_passes(requests_frame):
    results = validate(requests_frame, asof=_asof(requests_frame), thresholds=SMALL)
    report = build_report(results, asof=_asof(requests_frame), strict=False)
    assert report["passed"], report
    assert report["warnings"] == []


def test_missing_column_fails_and_stops(requests_frame):
    results = validate(
        requests_frame.drop(columns=["service_name"]),
        asof=_asof(requests_frame),
        thresholds=SMALL,
    )
    assert [r.name for r in results] == ["schema"]
    assert not results[0].passed
    assert "service_name" in results[0].observed["missing"]


def test_truncated_pull_fails(requests_frame):
    # e.g. Socrata's silent 1,000-row cap without paging
    assert not _results(requests_frame.head(50))["row_count"].passed


def test_row_count_swing_vs_previous_run_fails(requests_frame):
    assert not _results(requests_frame, baseline_rows=1_000)["row_count"].passed
    assert _results(requests_frame, baseline_rows=390)["row_count"].passed


def test_duplicate_ids_fail(requests_frame):
    doubled = pd.concat([requests_frame, requests_frame.head(20)], ignore_index=True)
    assert not _results(doubled)["unique_ids"].passed


def test_null_service_name_fails(requests_frame):
    df = requests_frame.copy()
    df.loc[:9, "service_name"] = None
    check = _results(df)["null_rates"]
    assert not check.passed
    assert "service_name" in check.detail


def test_stale_snapshot_fails(requests_frame):
    later = _asof(requests_frame) + pd.Timedelta(days=60)
    results = {r.name: r for r in validate(requests_frame, asof=later, thresholds=SMALL)}
    assert not results["dates"].passed
    assert results["dates"].observed["stale_days"] >= 60


def test_future_dated_and_negative_durations_fail(requests_frame):
    df = requests_frame.copy()
    asof = _asof(df)
    df.loc[0, "requested_date"] = (asof + pd.Timedelta(days=30)).isoformat()
    # closed a day before it was requested
    for i in range(1, 6):
        req = pd.Timestamp(df.loc[i, "requested_date"])
        df.loc[i, "closed_date"] = (req - pd.Timedelta(days=1)).isoformat()
    check = {r.name: r for r in validate(df, asof=asof, thresholds=SMALL)}["dates"]
    assert not check.passed
    assert check.observed["future_rows"] == 1
    assert check.observed["negative_duration_rate"] > 0


def test_montgomery_style_bulk_close_warns(requests_frame):
    df = _with_montgomery(requests_frame)
    results = _results(df)
    bulk, outliers = results["bulk_close"], results["group_outliers"]
    assert not bulk.passed and not outliers.passed
    assert [g["comm_name"] for g in bulk.observed["groups"]] == ["MONTGOMERY"]
    assert [g["comm_name"] for g in outliers.observed["groups"]] == ["MONTGOMERY"]
    # structural checks are unaffected: it's a warning, not a failure
    report = build_report(list(results.values()), asof=_asof(df), strict=False)
    assert report["passed"]
    assert set(report["warnings"]) == {"bulk_close", "group_outliers"}


def test_slow_but_genuine_community_is_not_flagged(requests_frame):
    """A community that's consistently slower (Bowness-like, ~7× the norm) with closures
    spread over many days is a finding, not an artifact."""
    df = requests_frame.copy()
    slow = (df["service_name"] == "Pothole Repair") & (df["comm_name"] == "BOWNESS")
    closed = df["closed_date"].notna()
    req = pd.to_datetime(df.loc[slow & closed, "requested_date"], utc=True)
    age = pd.to_datetime(df.loc[slow & closed, "closed_date"], utc=True) - req
    df.loc[slow & closed, "closed_date"] = (req + age * 7).map(pd.Timestamp.isoformat)
    results = _results(df)
    assert results["bulk_close"].passed
    assert results["group_outliers"].passed


def test_main_writes_report_then_raises(tmp_path, requests_frame):
    in_dir, out_dir = tmp_path / "input", tmp_path / "output"
    in_dir.mkdir()
    requests_frame.head(50).to_parquet(in_dir / "part-0.parquet")

    with pytest.raises(DataValidationError, match="row_count"):
        main(
            ["--input", str(in_dir), "--output", str(out_dir), "--asof", "2025-06-01"],
            thresholds=SMALL,
        )
    report = json.loads((out_dir / REPORT_NAME).read_text())
    assert report["passed"] is False
    assert "row_count" in report["failed"]


def test_strict_turns_warnings_into_failures(tmp_path, requests_frame):
    df = _with_montgomery(requests_frame)
    in_dir, out_dir = tmp_path / "input", tmp_path / "output"
    in_dir.mkdir()
    df.to_parquet(in_dir / "part-0.parquet")
    args = ["--input", str(in_dir), "--output", str(out_dir), "--asof", str(_asof(df).date())]

    assert main(args, thresholds=SMALL)["warnings"] == ["bulk_close", "group_outliers"]
    with pytest.raises(DataValidationError, match="bulk_close"):
        main([*args, "--strict"], thresholds=SMALL)
