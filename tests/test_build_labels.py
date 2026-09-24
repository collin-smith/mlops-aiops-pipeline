import numpy as np
import pandas as pd
import pytest

from src.features.build_labels import (
    LeakageError,
    apply_labels,
    assert_no_leakage,
    build_training_labels,
    compute_thresholds,
    labelled_subset,
    time_split,
)


def test_open_requests_are_dropped_from_labelled_subset(requests_frame):
    frame = labelled_subset(requests_frame)
    assert frame["closed_date"].notna().all()
    assert (frame["days_to_close"] >= 0).all()
    # 1 in 20 rows were open
    assert len(frame) < len(requests_frame)


def test_thresholds_are_per_category(requests_frame):
    frame = labelled_subset(requests_frame)
    thr = compute_thresholds(frame, percentile=0.75)
    assert "__global__" in thr.index
    # trees take far longer than potholes -> higher threshold
    assert thr["Tree Concern"] > thr["Pothole Repair"] * 3


def test_breach_rate_matches_percentile(requests_frame):
    frame = labelled_subset(requests_frame)
    thr = compute_thresholds(frame, percentile=0.75)
    labelled = apply_labels(frame, thr)
    # by construction ~25% breach, per category
    rate = labelled.groupby("service_name")["breach"].mean()
    assert (rate.between(0.15, 0.35)).all()


def test_time_split_has_no_overlap(requests_frame):
    frame = labelled_subset(requests_frame)
    train, test = time_split(frame, holdout_years=1)
    assert len(train) + len(test) == len(frame)
    if len(test):
        assert (
            pd.to_datetime(train["requested_date"], utc=True).max()
            <= pd.to_datetime(test["requested_date"], utc=True).min()
        )


def test_thresholds_computed_on_train_only(requests_frame):
    """The threshold Series must be derivable from train without seeing test rows."""
    train_lab, test_lab, thr = build_training_labels(requests_frame)
    # test labels use the same threshold object -> identical per-category values
    merged = test_lab.merge(
        thr.rename("expected"), left_on="service_name", right_index=True, how="left"
    )
    known = merged["expected"].notna()
    assert np.allclose(merged.loc[known, "threshold_days"], merged.loc[known, "expected"])


def test_unseen_category_falls_back_to_global(requests_frame):
    frame = labelled_subset(requests_frame)
    thr = compute_thresholds(frame)
    novel = frame.head(3).copy()
    novel["service_name"] = "Brand New Category"
    labelled = apply_labels(novel, thr)
    assert np.allclose(labelled["threshold_days"], thr["__global__"])


def test_leakage_guard_rejects_outcome_columns():
    bad = pd.DataFrame({"service_name": ["x"], "closed_date": ["2021-01-01"]})
    with pytest.raises(LeakageError):
        assert_no_leakage(bad)


def test_leakage_guard_passes_clean_features():
    ok = pd.DataFrame({"service_name": ["x"], "req_month": [1], "cat_open_30d": [2]})
    assert_no_leakage(ok)  # no raise


def test_mixed_iso_timestamp_formats_all_parse():
    """pandas guesses a date format from the first value and, with errors="coerce",
    silently turns differently-formatted values into NaT, which drops them from the
    label. Whole-second and fractional-second ISO strings must both parse."""
    from src.features.build_labels import add_days_to_close

    df = pd.DataFrame(
        {
            "requested_date": ["2024-03-01T00:00:00+00:00", "2024-03-01T08:15:00.123+00:00"],
            "closed_date": ["2024-03-03T00:04:54.315697+00:00", "2024-03-04T00:00:00+00:00"],
        }
    )
    assert add_days_to_close(df)["days_to_close"].notna().all()


def _with_purge(df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """A backlog purge: ``n`` long-open Tree Concern tickets all closed on one day."""
    day = pd.Timestamp("2024-06-01", tz="UTC")
    rows = [
        {
            **df.iloc[0].to_dict(),
            "service_request_id": f"SR-P{i:04d}",
            "requested_date": (day - pd.Timedelta(days=900 + i)).isoformat(),
            "closed_date": day.isoformat(),
            "status_description": "Closed",
            "service_name": "Tree Concern",
        }
        for i in range(n)
    ]
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def _purged_ids(frame: pd.DataFrame) -> pd.Series:
    return frame["service_request_id"].str.startswith("SR-P")


def test_flag_keeps_purged_rows_labelled_but_out_of_the_threshold(requests_frame):
    df = _with_purge(requests_frame)
    train, _, thr = build_training_labels(df, purge_handling="flag")
    purged = train.loc[_purged_ids(train)]
    assert len(purged) == 40
    assert purged["purge_closed"].all()
    assert (purged["breach"] == 1).all()  # they really were open that long
    # the threshold matches a build that never saw the purge
    _, _, clean_thr = build_training_labels(requests_frame, purge_handling="flag")
    assert thr["Tree Concern"] == pytest.approx(clean_thr["Tree Concern"])


def test_keep_lets_the_purge_inflate_the_threshold(requests_frame):
    df = _with_purge(requests_frame)
    _, _, kept = build_training_labels(df, purge_handling="keep")
    _, _, flagged = build_training_labels(df, purge_handling="flag")
    assert kept["Tree Concern"] > flagged["Tree Concern"]


def test_exclude_drops_purged_rows(requests_frame):
    train, test, _ = build_training_labels(_with_purge(requests_frame), purge_handling="exclude")
    assert not _purged_ids(train).any() and not _purged_ids(test).any()


def test_purge_flag_counts_as_a_leaky_column(requests_frame):
    train, _, _ = build_training_labels(requests_frame)
    with pytest.raises(LeakageError, match="purge_closed"):
        assert_no_leakage(train[["service_name", "purge_closed"]])


def test_unknown_purge_handling_is_rejected(requests_frame):
    with pytest.raises(ValueError, match="purge_handling"):
        build_training_labels(requests_frame, purge_handling="ignore")


def _with_open(df: pd.DataFrame, ages_days: list[int]) -> pd.DataFrame:
    """Open Pothole Repair requests of the given ages at the snapshot."""
    newest = pd.to_datetime(df["requested_date"], utc=True).max()
    rows = [
        {
            **df.iloc[0].to_dict(),
            "service_request_id": f"SR-O{i:04d}",
            "requested_date": (newest - pd.Timedelta(days=age)).isoformat(),
            "closed_date": None,
            "status_description": "Open",
            "service_name": "Pothole Repair",
        }
        for i, age in enumerate(ages_days)
    ]
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def test_open_request_past_its_threshold_is_a_certain_breach(requests_frame):
    # potholes close in ~3 days here, so 60 days open is certainly late; 0 days is unknown
    df = _with_open(requests_frame, [60, 60, 0, 0])
    _, test, thr = build_training_labels(df)
    opened = test.loc[test["service_request_id"].str.startswith("SR-O")]
    assert len(opened) == 2  # the two still inside their threshold have no label yet
    assert (opened["breach"] == 1).all() and opened["censored"].all()
    assert opened["days_to_close"].isna().all()
    assert thr["Pothole Repair"] < 60


def test_overdue_open_can_be_turned_off(requests_frame):
    df = _with_open(requests_frame, [60, 60])
    _, test, _ = build_training_labels(df, include_overdue_open=False)
    assert not test["service_request_id"].str.startswith("SR-O").any()
    assert not test["censored"].any()


def test_censored_counts_as_a_leaky_column(requests_frame):
    train, _, _ = build_training_labels(requests_frame)
    with pytest.raises(LeakageError, match="censored"):
        assert_no_leakage(train[["service_name", "censored"]])
