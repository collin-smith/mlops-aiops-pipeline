"""Intake-time feature construction — nothing the desk clerk wouldn't know at intake.

Features
--------
categorical : service_name, agency_responsible, comm_name, source
calendar    : req_month, req_dow, req_is_weekend, req_is_holiday_week
backlog     : cat_open_30d      (rolling 30-day request count for that service_name)
              comm_req_30d      (rolling 30-day request count for that community)

No column derived from updated_date / closed_date / status_description. The
``assert_no_leakage`` check in ``build_labels`` runs over the output of ``build_features``.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src.common.config import LEAKY_COLUMNS

CATEGORICAL = ["service_name", "agency_responsible", "comm_name", "source"]
CALENDAR = ["req_month", "req_dow", "req_is_weekend", "req_is_holiday_week"]
BACKLOG = ["cat_open_30d", "comm_req_30d"]
FEATURE_COLUMNS = CATEGORICAL + CALENDAR + BACKLOG

_AB_HOLIDAYS_CACHE: dict[tuple[int, int], set] = {}


def _holiday_weeks(years: range) -> set:
    key = (years.start, years.stop)
    if key not in _AB_HOLIDAYS_CACHE:
        ab = holidays.Canada(prov="AB", years=list(years))
        weeks = {pd.Timestamp(d).to_period("W") for d in ab}
        _AB_HOLIDAYS_CACHE[key] = weeks
    return _AB_HOLIDAYS_CACHE[key]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    req = pd.to_datetime(out["requested_date"], errors="coerce", utc=True, format="ISO8601")
    out["req_month"] = req.dt.month.astype("int8")
    out["req_dow"] = req.dt.dayofweek.astype("int8")
    out["req_is_weekend"] = (req.dt.dayofweek >= 5).astype("int8")

    yrs = range(int(req.dt.year.min()), int(req.dt.year.max()) + 2)
    holiday_weeks = _holiday_weeks(yrs)
    weeks = req.dt.tz_localize(None).dt.to_period("W")
    out["req_is_holiday_week"] = weeks.isin(holiday_weeks).astype("int8")
    return out


_WINDOW = np.timedelta64(30, "D")


def _rolling_30d_count(df: pd.DataFrame, key: str, out_col: str) -> pd.Series:
    """For each row, how many requests shared ``key`` in the 30 days *before* it.

    Strictly-before: a request never counts itself, nor same-instant siblings that
    happen to sort after it — counting those would be lookahead. ``requested_date``
    on this dataset is date-granular, so same-day requests do not inflate each other.
    """
    req = pd.to_datetime(df["requested_date"], errors="coerce", utc=True, format="ISO8601")
    # keep native datetime64 (unit may be us or ns depending on pandas) and let numpy
    # do unit-aware datetime arithmetic — do NOT hand-roll an int64 nanosecond window.
    t = req.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()

    tmp = pd.DataFrame(
        {"rid": np.arange(len(df), dtype="int64"), "k": df[key].astype("string").fillna("__na__")}
    )
    tmp["t"] = t
    tmp = tmp.sort_values(["k", "t"], kind="stable")

    out = np.zeros(len(df), dtype="int32")
    for _, g in tmp.groupby("k", sort=False):
        arr = g["t"].to_numpy()
        lo = np.searchsorted(arr, arr - _WINDOW, side="left")
        hi = np.searchsorted(arr, arr, side="left")  # elements strictly earlier
        out[g["rid"].to_numpy()] = (hi - lo).astype("int32")

    return pd.Series(out, index=df.index, name=out_col)


def add_backlog_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cat_open_30d"] = _rolling_30d_count(out, "service_name", "cat_open_30d")
    out["comm_req_30d"] = _rolling_30d_count(out, "comm_name", "comm_req_30d")
    return out


def build_features(df: pd.DataFrame, *, keep_key: bool = True) -> pd.DataFrame:
    """Return the model-ready feature matrix (+ ``service_request_id`` if ``keep_key``)."""
    out = add_backlog_features(add_calendar_features(df))
    cols = FEATURE_COLUMNS[:]
    if keep_key and "service_request_id" in out:
        cols = ["service_request_id", *cols]
    result = out[cols].copy()
    for c in CATEGORICAL:
        result[c] = result[c].astype("category")

    leaked = sorted(set(result.columns) & set(LEAKY_COLUMNS))
    if leaked:  # defensive: FEATURE_COLUMNS is a fixed allowlist, but never trust that alone
        raise AssertionError(f"leaky columns in feature matrix: {leaked}")
    return result
