"""Breach-label construction.

    days_to_close  = closed_date - requested_date            (days, float)
    threshold[c]   = BREACH_PERCENTILE of days_to_close for service_name == c,
                     computed on the TRAINING split only
    breach         = 1 if days_to_close > threshold[service_name] else 0

Per-category thresholds stop the model from just learning "pothole fast, tree slow".
Open requests (null closed_date) get no label — they are exactly the rows Batch
Transform scores in Stage 6.

The article's honesty note: this is a *proxy* threshold (a category's own historical
75th percentile), not a City-published SLA.
"""

from __future__ import annotations

import pandas as pd

from src.common.config import (
    BREACH_PERCENTILE,
    HOLDOUT_YEARS,
    LEAKY_COLUMNS,
    TRAIN_WINDOW_YEARS,
)


class LeakageError(AssertionError):
    """Raised when an outcome-derived column reaches the feature matrix."""


def add_days_to_close(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    req = pd.to_datetime(out["requested_date"], errors="coerce", utc=True, format="ISO8601")
    clo = pd.to_datetime(out["closed_date"], errors="coerce", utc=True, format="ISO8601")
    out["days_to_close"] = (clo - req).dt.total_seconds() / 86_400.0
    return out


def labelled_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that can carry a label: closed, non-negative resolution time, in window."""
    out = add_days_to_close(df)
    req = pd.to_datetime(out["requested_date"], errors="coerce", utc=True, format="ISO8601")
    cutoff = req.max() - pd.DateOffset(years=TRAIN_WINDOW_YEARS)
    keep = out["days_to_close"].notna() & (out["days_to_close"] >= 0) & (req >= cutoff)
    return out.loc[keep].reset_index(drop=True)


def time_split(
    df: pd.DataFrame, holdout_years: int = HOLDOUT_YEARS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by ``requested_date``: train = older, test = most recent ``holdout_years``."""
    req = pd.to_datetime(df["requested_date"], errors="coerce", utc=True, format="ISO8601")
    boundary = req.max() - pd.DateOffset(years=holdout_years)
    train = df.loc[req < boundary].reset_index(drop=True)
    test = df.loc[req >= boundary].reset_index(drop=True)
    return train, test


def compute_thresholds(train_df: pd.DataFrame, percentile: float = BREACH_PERCENTILE) -> pd.Series:
    """Per-``service_name`` threshold, plus a ``__global__`` fallback for unseen categories."""
    df = train_df if "days_to_close" in train_df else add_days_to_close(train_df)
    per_cat = df.groupby("service_name")["days_to_close"].quantile(percentile)
    per_cat.loc["__global__"] = df["days_to_close"].quantile(percentile)
    return per_cat.rename("threshold_days")


def apply_labels(df: pd.DataFrame, thresholds: pd.Series) -> pd.DataFrame:
    out = df if "days_to_close" in df else add_days_to_close(df)
    out = out.copy()
    global_thr = float(thresholds.get("__global__", thresholds.median()))
    thr = out["service_name"].map(thresholds).fillna(global_thr)
    out["threshold_days"] = thr
    out["breach"] = (out["days_to_close"] > thr).astype("int8")
    return out


def build_training_labels(
    df: pd.DataFrame, percentile: float = BREACH_PERCENTILE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """End-to-end: window -> time split -> thresholds (train only) -> labels on both.

    Returns ``(train_labelled, test_labelled, thresholds)``.
    """
    frame = labelled_subset(df)
    train, test = time_split(frame)
    thresholds = compute_thresholds(train, percentile)
    return apply_labels(train, thresholds), apply_labels(test, thresholds), thresholds


def assert_no_leakage(features: pd.DataFrame) -> None:
    """Fail loudly if any outcome-derived column is present in the feature matrix."""
    offenders = sorted(set(features.columns) & set(LEAKY_COLUMNS))
    if offenders:
        raise LeakageError(f"leaky columns in feature matrix: {offenders}")
