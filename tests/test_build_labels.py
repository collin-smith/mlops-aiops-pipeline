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
