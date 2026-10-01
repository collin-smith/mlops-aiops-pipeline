"""Stage 5 history features: values on a hand-worked example, and no lookahead. No AWS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FEATURE_SETS, build_features
from src.features.history import (
    HISTORY_COLUMNS,
    backlog_ratio,
    build_history,
    late_rate,
    sector_lookup,
)

SECTORS = sector_lookup(pd.DataFrame([{"comm_code": "X1", "sector": "WEST"}]))
THRESHOLDS = pd.Series({"A": 10.0})  # category A: late after 10 days


def _requests(closed=("2025-01-06", "2025-01-25", None, None)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service_name": "A",
            "comm_name": "X",
            "comm_code": "X1",
            "requested_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-20"],
            "closed_date": list(closed),
        }
    )


def _query(dates) -> pd.DataFrame:
    return pd.DataFrame({"service_name": "A", "comm_name": "X", "requested_date": dates})


def test_late_rate_counts_only_outcomes_decided_before_intake():
    """Deadlines Jan 11 (closed on time), 12 (closed late), 13 (never closed), 30."""
    h = build_history(_requests(), THRESHOLDS, SECTORS)
    assert h.outcomes["late"].tolist() == [0, 1, 1, 1]
    rate = late_rate(_query(["2025-01-11", "2025-01-13", "2025-02-15"]), h.outcomes,
                     ("service_name",), 90, 1)  # fmt: skip
    # Jan 11: nothing decided yet. Jan 13: Jan 11 on time, Jan 12 late. Feb 15: all four.
    assert np.isnan(rate[0])
    assert rate[1:].tolist() == [0.5, 0.75]


def test_closed_late_after_intake_was_already_overdue_at_intake():
    """Closed Jan 25, deadline Jan 12: at Jan 13 it was open and overdue, so it's late."""
    h = build_history(_requests(), THRESHOLDS, SECTORS)
    assert late_rate(_query(["2025-01-13"]), h.outcomes, ("service_name",), 90, 1)[0] == 0.5


def test_future_outcomes_cannot_change_a_feature():
    """Rewrite what happened to every request whose deadline is after t: nothing at t moves."""
    q = _query(["2025-01-13"])
    before = build_history(_requests(), THRESHOLDS, SECTORS)
    # requests 3 and 4 (deadlines Jan 13 and 30) now close on time instead
    rewritten = ("2025-01-06", "2025-01-25", "2025-01-04", "2025-01-21")
    after = build_history(_requests(closed=rewritten), THRESHOLDS, SECTORS)
    a = late_rate(q, before.outcomes, ("service_name",), 90, 1)
    b = late_rate(q, after.outcomes, ("service_name",), 90, 1)
    assert a.tolist() == b.tolist()


def test_a_request_never_sees_its_own_outcome():
    """Its deadline is after its own filing time, so it can't be in its own window."""
    reqs = _requests()
    h = build_history(reqs, THRESHOLDS, SECTORS)
    own = late_rate(reqs, h.outcomes, ("service_name",), 90, 1)
    # the first three have no decided predecessors yet
    assert np.isnan(own[:3]).all()
    # the fourth (Jan 20, itself late) sees the first three: 2 of 3 late. With its own
    # outcome it would be 3 of 4.
    assert own[3] == pytest.approx(2 / 3)


def test_too_few_decided_requests_is_missing_not_zero():
    h = build_history(_requests(), THRESHOLDS, SECTORS)
    assert np.isnan(late_rate(_query(["2025-02-15"]), h.outcomes, ("service_name",), 90, 5)[0])


def test_composite_keys_keep_groups_apart():
    reqs = pd.concat([_requests(), _requests().assign(comm_name="Y")], ignore_index=True)
    reqs.loc[4:, "closed_date"] = "2025-01-02"  # every Y request on time
    h = build_history(reqs, THRESHOLDS, SECTORS)
    q = pd.concat([_query(["2025-02-15"]), _query(["2025-02-15"]).assign(comm_name="Y")])
    rate = late_rate(q, h.outcomes, ("service_name", "comm_name"), 90, 1)
    assert rate.tolist() == [0.75, 0.0]


def test_backlog_ratio_compares_the_last_month_with_a_typical_one():
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    filed = pd.DataFrame({"service_name": "A", "requested_date": days.strftime("%Y-%m-%d")})
    # one request a day all year, then a burst: 3 a day through January 2025
    burst = pd.date_range("2025-01-01", "2025-01-30", freq="D").repeat(3)
    filed = pd.concat(
        [filed, pd.DataFrame({"service_name": "A", "requested_date": burst.strftime("%Y-%m-%d")})]
    )
    h = build_history(filed.assign(comm_name="X", comm_code="X1", closed_date=None),
                      THRESHOLDS, SECTORS)  # fmt: skip
    steady, busy = backlog_ratio(_query(["2024-12-31", "2025-01-31"]), h.filed)
    assert steady == pytest.approx(1.0, abs=0.05)
    assert busy > 2.5


def test_build_features_adds_the_challenger_columns_only_with_history(requests_frame):
    base = build_features(requests_frame, keep_key=False)
    assert list(base.columns) == FEATURE_SETS["baseline"]
    reqs = requests_frame.assign(comm_code="X1")
    h = build_history(reqs, pd.Series({s: 5.0 for s in reqs["service_name"].unique()}), SECTORS)
    full = build_features(reqs, keep_key=False, history=h, sectors=SECTORS)
    assert list(full.columns) == FEATURE_SETS["challenger"]
    assert set(HISTORY_COLUMNS) <= set(full.columns)
    with pytest.raises(ValueError):
        build_features(reqs, keep_key=False, history=h)
