"""Intake-time feature construction — nothing the desk clerk wouldn't know at intake.

Features
--------
categorical : service_name, agency_responsible, comm_name, source
calendar    : req_month, req_dow, req_is_weekend, req_is_holiday_week
backlog     : cat_open_30d      (rolling 30-day request count for that service_name)
              comm_req_30d      (rolling 30-day request count for that community)
history     : the Stage 5 challenger's six (``src/features/history.py``, D-042): backlog
              against normal, and recent late rates by category, community, sector and their
              combinations, counting only outcomes already decided at intake time

No column derived from updated_date / closed_date / status_description. The
``assert_no_leakage`` check in ``build_labels`` runs over the output of ``build_features``.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src.common.config import LEAKY_COLUMNS
from src.features.history import HISTORY_COLUMNS, History, add_history_features

CATEGORICAL = ["service_name", "agency_responsible", "comm_name", "source"]
CALENDAR = ["req_month", "req_dow", "req_is_weekend", "req_is_holiday_week"]
BACKLOG = ["cat_open_30d", "comm_req_30d"]
FEATURE_COLUMNS = CATEGORICAL + CALENDAR + BACKLOG
# baseline = the Stage 2 model (D-037); challenger = Stage 5 (D-042)
FEATURE_SETS = {"baseline": FEATURE_COLUMNS, "challenger": FEATURE_COLUMNS + HISTORY_COLUMNS}

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


def build_features(
    df: pd.DataFrame,
    *,
    keep_key: bool = True,
    history: History | None = None,
    sectors: pd.Series | None = None,
) -> pd.DataFrame:
    """Return the model-ready feature matrix (+ ``service_request_id`` if ``keep_key``).

    With ``history`` (and ``sectors``, comm_code -> sector) it's the challenger's feature
    set; without, the baseline's.
    """
    out = add_backlog_features(add_calendar_features(df))
    cols = FEATURE_SETS["baseline"][:]
    if history is not None:
        if sectors is None:
            raise ValueError("history features need the sector lookup")
        out = add_history_features(out, history, sectors)
        cols = FEATURE_SETS["challenger"][:]
    if keep_key and "service_request_id" in out:
        cols = ["service_request_id", *cols]
    result = out[cols].copy()
    for c in CATEGORICAL:
        result[c] = result[c].astype("category")

    leaked = sorted(set(result.columns) & set(LEAKY_COLUMNS))
    if leaked:  # defensive: FEATURE_COLUMNS is a fixed allowlist, but never trust that alone
        raise AssertionError(f"leaky columns in feature matrix: {leaked}")
    return result


def align_categories(features: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Give ``features`` the categorical levels the model was trained on (``reference``).

    A level the model never saw becomes missing, which XGBoost routes down its learned
    default branch. Mapping it first avoids pandas' deprecated behaviour of silently
    constructing a Categorical from values outside its categories.
    """
    out = features.copy()
    for c in CATEGORICAL:
        levels = reference[c].cat.categories
        out[c] = pd.Categorical(out[c].where(out[c].isin(levels)), categories=levels)
    return out


def align_to_schema(features: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """``align_categories`` for a saved model: ``schema`` is its ``feature_schema.json``.

    Puts the columns in training order and gives each categorical column the training
    levels, in the same order, so the category codes XGBoost sees match the ones it
    learned. Anything scoring or evaluating a model artifact goes through this.
    """
    out = features[schema["columns"]].copy()
    for c, levels in schema["categories"].items():
        values = out[c].astype("string")
        out[c] = pd.Categorical(values.where(values.isin(levels)), categories=levels)
    return out
