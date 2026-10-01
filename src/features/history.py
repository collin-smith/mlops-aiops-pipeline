"""History features for the Stage 5 challenger: how a request's category and area have been
doing lately, as known at the moment the request comes in.

At intake time *t*, a past request's outcome is known only once its deadline has passed
(its filed time plus its category's threshold, the same threshold the label uses). By
then it either closed in time or it didn't. So these features count only past requests
whose deadline fell in the window before *t*:

* late = closed after its deadline, or not closed at all. Both are knowable at *t*,
  because the deadline is already behind *t*. A request closed after *t* but past its
  deadline was, at *t*, open and overdue.
* A request's own outcome never counts: its deadline is after its own filing time.

Features (each NaN when too few decided requests to say):

    cat_backlog_ratio      requests in the category in the 30 days before t, over its
                           typical 30 days in the year before t ("backlog against normal")
    cat_late_90d           share of the category's decided requests that ran late
    comm_late_90d          the same for the community
    sector_late_90d        the same for the sector
    cat_sector_late_90d    the category, within the sector
    comm_cat_late_180d     the category, within the community (180 days: it's sparse)

D-042. The lookups sort once and binary-search; building composite keys row by row
was far too slow on 2.9M rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UNKNOWN_SECTOR = "UNKNOWN"
_SEC = np.int64(10**10)  # key code * _SEC + epoch seconds: one sortable number per (key, time)

# name -> (key columns, window days, minimum decided requests)
LATE_RATES: dict[str, tuple[tuple[str, ...], int, int]] = {
    "cat_late_90d": (("service_name",), 90, 20),
    "comm_late_90d": (("comm_name",), 90, 20),
    "sector_late_90d": (("sector",), 90, 20),
    "cat_sector_late_90d": (("service_name", "sector"), 90, 10),
    "comm_cat_late_180d": (("service_name", "comm_name"), 180, 5),
}
HISTORY_COLUMNS = ["cat_backlog_ratio", *LATE_RATES]


def _times(s: pd.Series) -> np.ndarray:
    """Naive-UTC datetime64[s]."""
    t = pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")
    return t.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[s]")


def _secs(t: np.ndarray) -> np.ndarray:
    return t.astype("datetime64[s]").astype("int64")


def sector_lookup(communities: pd.DataFrame) -> pd.Series:
    """comm_code -> the City planning sector."""
    return communities.drop_duplicates("comm_code").set_index("comm_code")["sector"]


def with_sector(df: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    return df.assign(sector=df["comm_code"].map(sectors).fillna(UNKNOWN_SECTOR).astype(str))


@dataclass(frozen=True)
class History:
    """Everything the history features look back at: the full snapshot's requests with
    their deadlines and outcomes, and when each was filed."""

    outcomes: pd.DataFrame  # service_name, comm_name, sector, deadline, late
    filed: pd.DataFrame  # service_name, t


def build_history(requests: pd.DataFrame, thresholds: pd.Series, sectors: pd.Series) -> History:
    """``requests`` is the whole snapshot (not a train or test split); ``thresholds`` are
    the per-category days the label uses, from the model's training split."""
    frame = with_sector(requests, sectors)
    thr = frame["service_name"].map(thresholds)
    known = thr.notna().to_numpy()
    req = _times(frame["requested_date"])
    closed = _times(frame["closed_date"])
    days = thr.to_numpy(dtype="float64")
    deadline = req[known] + (days[known] * 86_400).astype("int64").astype("timedelta64[s]")
    late = np.isnat(closed[known]) | (closed[known] > deadline)
    outcomes = frame.loc[known, ["service_name", "comm_name", "sector"]].reset_index(drop=True)
    outcomes["deadline"] = deadline
    outcomes["late"] = late.astype("int8")
    filed = pd.DataFrame({"service_name": frame["service_name"].to_numpy(), "t": req})
    return History(outcomes=outcomes, filed=filed)


def _codes(ref: pd.DataFrame, query: pd.DataFrame, keys: tuple[str, ...]):
    """One integer per key combination, the same in both frames."""
    rc = np.zeros(len(ref), dtype="int64")
    qc = np.zeros(len(query), dtype="int64")
    for k in keys:
        both = pd.concat([ref[k], query[k]], ignore_index=True).astype("string").fillna("__na__")
        codes, uniq = pd.factorize(both)
        rc = rc * (len(uniq) + 1) + codes[: len(ref)]
        qc = qc * (len(uniq) + 1) + codes[len(ref) :]
    return rc, qc


def late_rate(query: pd.DataFrame, outcomes: pd.DataFrame, keys, window_days: int, min_n: int):
    """Share of same-key requests whose deadline fell in [t - window, t) that ran late."""
    rc, qc = _codes(outcomes, query, tuple(keys))
    rv = rc * _SEC + _secs(outcomes["deadline"].to_numpy())
    order = np.argsort(rv, kind="stable")
    rv = rv[order]
    cum = np.concatenate([[0], np.cumsum(outcomes["late"].to_numpy()[order])])
    t = _secs(_times(query["requested_date"]))
    hi = np.searchsorted(rv, qc * _SEC + t, side="left")
    lo = np.searchsorted(rv, qc * _SEC + t - window_days * 86_400, side="left")
    n = hi - lo
    rate = (cum[hi] - cum[lo]) / np.where(n > 0, n, 1)
    return np.where(n >= min_n, rate, np.nan)


def backlog_ratio(query: pd.DataFrame, filed: pd.DataFrame, min_year: int = 30):
    """Category requests in the 30 days before t, over its typical 30 days in the past year."""
    rc, qc = _codes(filed, query, ("service_name",))
    rv = np.sort(rc * _SEC + _secs(filed["t"].to_numpy()))
    q = qc * _SEC + _secs(_times(query["requested_date"]))
    hi = np.searchsorted(rv, q, side="left")
    n30 = hi - np.searchsorted(rv, q - 30 * 86_400, side="left")
    n365 = hi - np.searchsorted(rv, q - 365 * 86_400, side="left")
    typical = n365 * 30 / 365
    return np.where(n365 >= min_year, n30 / np.maximum(typical, 1e-9), np.nan)


def add_history_features(frame: pd.DataFrame, history: History, sectors: pd.Series):
    """``frame`` plus the HISTORY_COLUMNS, computed as of each row's filing time."""
    out = with_sector(frame, sectors)
    out["cat_backlog_ratio"] = backlog_ratio(out, history.filed)
    for name, (keys, window, min_n) in LATE_RATES.items():
        out[name] = late_rate(out, history.outcomes, keys, window, min_n)
    return out
