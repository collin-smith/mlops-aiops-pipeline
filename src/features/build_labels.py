"""Breach-label construction.

    days_to_close  = closed_date - requested_date            (days, float)
    threshold[c]   = BREACH_PERCENTILE of days_to_close for service_name == c,
                     computed on the TRAINING split only
    breach         = 1 if days_to_close > threshold[service_name] else 0

Per-category thresholds stop the model from just learning "pothole fast, tree slow".

Open requests (null closed_date) are right-censored: we only know they have been open
*at least* this long. One already older than its category's threshold is a certain
breach, so it's labelled 1 (``censored`` = True). One still inside its threshold has no
label yet and is left out; those are the rows Batch Transform scores in Stage 6.
Dropping every open request instead biases the recent end of the data towards fast
closes: on the 2026-09-23 pull, 13,378 certain breaches in the test year were being
thrown away, and the test breach rate read 15.8% instead of 18.1%.

The article's honesty note: this is a *proxy* threshold (a category's own historical
75th percentile), not a City-published SLA.

Backlog purges: the City routinely closes batches of long-open tickets on one day (3,505
traffic-sign tickets on 2025-08-26, at a median age of 1,085 days). Their breach label is
probably right, since they really were open that long, but their days_to_close is not
service time. ``purge_handling`` decides what to do with them (see ``PURGE_HANDLING``).
"""

from __future__ import annotations

import pandas as pd

from src.common.config import (
    BREACH_PERCENTILE,
    HOLDOUT_YEARS,
    LEAKY_COLUMNS,
    PURGE_HANDLING,
    TRAIN_WINDOW_YEARS,
)
from src.pipeline.validate import stale_bulk_close_mask

PURGE_HANDLINGS = ("flag", "exclude", "keep")


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
    df: pd.DataFrame,
    holdout_years: int = HOLDOUT_YEARS,
    *,
    boundary: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by ``requested_date``: train = older, test = most recent ``holdout_years``."""
    req = pd.to_datetime(df["requested_date"], errors="coerce", utc=True, format="ISO8601")
    if boundary is None:
        boundary = req.max() - pd.DateOffset(years=holdout_years)
    train = df.loc[req < boundary].reset_index(drop=True)
    test = df.loc[req >= boundary].reset_index(drop=True)
    return train, test


def compute_thresholds(
    train_df: pd.DataFrame,
    percentile: float = BREACH_PERCENTILE,
    *,
    ignore_purged: bool = False,
) -> pd.Series:
    """Per-``service_name`` threshold, plus a ``__global__`` fallback for unseen categories.

    With ``ignore_purged``, rows marked ``purge_closed`` don't count towards the threshold,
    so a backlog purge can't push a category's "normal" up to years.
    """
    df = train_df if "days_to_close" in train_df else add_days_to_close(train_df)
    if ignore_purged and "purge_closed" in df:
        df = df.loc[~df["purge_closed"]]
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


def overdue_open(df: pd.DataFrame, thresholds: pd.Series, asof: pd.Timestamp) -> pd.DataFrame:
    """Open requests already older than their category's threshold: certain breaches."""
    req = pd.to_datetime(df["requested_date"], errors="coerce", utc=True, format="ISO8601")
    cutoff = req.max() - pd.DateOffset(years=TRAIN_WINDOW_YEARS)
    is_open = df["closed_date"].isna() & (req >= cutoff)
    age = (asof - req).dt.total_seconds() / 86_400.0
    global_thr = float(thresholds.get("__global__", thresholds.median()))
    thr = df["service_name"].map(thresholds).fillna(global_thr)
    out = df.loc[is_open & (age > thr)].copy()
    out["days_to_close"] = float("nan")  # unknown: still open
    out["threshold_days"] = thr[out.index]
    out["breach"] = 1
    out["breach"] = out["breach"].astype("int8")
    out["censored"] = True
    return out.reset_index(drop=True)


def build_training_labels(
    df: pd.DataFrame,
    percentile: float = BREACH_PERCENTILE,
    purge_handling: str = PURGE_HANDLING,
    *,
    include_overdue_open: bool = True,
    asof: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """End-to-end: window -> time split -> thresholds (train only) -> labels on both.

    Returns ``(train_labelled, test_labelled, thresholds)``. Both frames carry
    ``purge_closed`` and ``censored`` columns, which are outcome-derived and never
    features. ``asof`` is when the snapshot was taken; it defaults to the day after the
    newest request.
    """
    if purge_handling not in PURGE_HANDLINGS:
        raise ValueError(f"purge_handling must be one of {PURGE_HANDLINGS}, not {purge_handling!r}")
    # Mark purges on the full pull, the same frame the validation gate sees.
    marked = df.assign(purge_closed=stale_bulk_close_mask(df).to_numpy())
    req = pd.to_datetime(marked["requested_date"], errors="coerce", utc=True, format="ISO8601")
    boundary = req.max() - pd.DateOffset(years=HOLDOUT_YEARS)  # one boundary for both parts
    if asof is None:
        asof = req.max().normalize() + pd.Timedelta(days=1)

    frame = labelled_subset(marked)
    if purge_handling == "exclude":
        frame = frame.loc[~frame["purge_closed"]].reset_index(drop=True)
    train, test = time_split(frame, boundary=boundary)
    thresholds = compute_thresholds(train, percentile, ignore_purged=purge_handling == "flag")
    train = apply_labels(train, thresholds).assign(censored=False)
    test = apply_labels(test, thresholds).assign(censored=False)

    if include_overdue_open:
        late_train, late_test = time_split(
            overdue_open(marked, thresholds, asof), boundary=boundary
        )
        train = pd.concat([train, late_train], ignore_index=True)
        test = pd.concat([test, late_test], ignore_index=True)
    return train, test, thresholds


def assert_no_leakage(features: pd.DataFrame) -> None:
    """Fail loudly if any outcome-derived column is present in the feature matrix."""
    offenders = sorted(set(features.columns) & set(LEAKY_COLUMNS))
    if offenders:
        raise LeakageError(f"leaky columns in feature matrix: {offenders}")
