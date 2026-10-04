"""Stage 7 Layer C (D-045): each scorecard series by hand, provisional months, flags. No AWS."""

from __future__ import annotations

import pandas as pd
import pytest

from src.monitor import civic_scorecard as cs

M1, M2 = pd.Period("2025-01", "M"), pd.Period("2025-02", "M")


def _lab(rows):
    return pd.DataFrame(rows, columns=["month", "service_name", "sector", "dow", "breach"])


def test_city_rate_holds_the_mix_fixed():
    # month 2 files far more of the slow type; at fixed mix the rate doesn't move
    rows = [(M1, "fast", "EAST", 0, 0)] * 9 + [(M1, "fast", "EAST", 0, 1)]
    rows += [(M1, "slow", "EAST", 0, 0)] * 5 + [(M1, "slow", "EAST", 0, 1)] * 5
    rows += [(M2, "fast", "EAST", 0, 0)] * 9 + [(M2, "fast", "EAST", 0, 1)]
    rows += [(M2, "slow", "EAST", 0, 0)] * 50 + [(M2, "slow", "EAST", 0, 1)] * 50
    out = cs.city_late_mix_fixed(_lab(rows)).set_index("month")["value"]
    assert out[M1] == pytest.approx(out[M2])
    raw_m2 = 51 / 110
    assert out[M2] != pytest.approx(raw_m2)


def test_sector_obs_expected_compares_like_with_like():
    # EAST is late twice as often as WEST on the same type, in the same month
    rows = [(M1, "t", "EAST", 0, 1)] * 4 + [(M1, "t", "EAST", 0, 0)] * 6
    rows += [(M1, "t", "WEST", 0, 1)] * 2 + [(M1, "t", "WEST", 0, 0)] * 8
    rows += [(M1, "t", "UNKNOWN", 0, 1)] * 3
    out = cs.sector_obs_expected(_lab(rows)).set_index("key")["value"]
    expected = 9 / 23  # the month's city-wide rate for type t, UNKNOWN included
    assert out["EAST"] == pytest.approx(0.4 / expected)
    assert out["WEST"] == pytest.approx(0.2 / expected)
    assert "UNKNOWN" not in out


def test_waste_east_and_weekend_series():
    rows = [(M1, cs.WASTE, "EAST", 4, 1)] * 3 + [(M1, cs.WASTE, "EAST", 1, 0)] * 3
    rows += [(M1, cs.WASTE, "WEST", 1, 1)] * 6
    lab = _lab(rows)
    waste = cs.waste_east_late(lab).iloc[0]
    assert waste["value"] == 0.5 and waste["n"] == 6
    wk = cs.fri_sat_vs_mon_thu(lab).iloc[0]
    # EAST Fri: 3/3 late vs expected 0.5 -> 2.0; EAST Tue 0/3 vs 0.5 -> 0, WEST Tue 6/6 vs 1 -> 1
    # weekday pooled: late 6/9, expected (3*0.5 + 6*1)/9 -> ratio (2/3)/(7.5/9)
    assert wk["value"] == pytest.approx(2.0 / ((6 / 9) / (7.5 / 9)))


def test_slowest_share_over_a_trailing_year():
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for i in range(100):
        req = base + pd.Timedelta(days=i * 3)
        days = 100 if i < 5 else 1  # five slow requests, ninety-five fast ones
        rows.append(
            {
                "service_request_id": f"R{i}",
                "requested_date": req.isoformat(),
                "closed_date": (req + pd.Timedelta(days=days)).isoformat(),
                "service_name": "t",
            }
        )
    raw = pd.DataFrame(rows)
    out, pending = cs.slowest5_share(raw, pd.Timestamp("2026-01-01", tz="UTC"))
    first = out.iloc[0]
    assert first["month"] == pd.Period("2024-12", "M")
    assert first["value"] == pytest.approx(500 / (500 + 95 * 1), rel=0.05)
    assert (pending == 0).all()


def test_provisional_months_are_never_flagged():
    months = pd.period_range("2021-01", periods=48, freq="M")
    values = [0.2] * 36 + [0.6] * 12  # a big shift, but the last year is provisional
    card = pd.DataFrame(
        {
            "month": months,
            "series": "s",
            "key": "",
            "value": values,
            "n": 100,
            "provisional": [False] * 36 + [True] * 12,
        }
    )
    out = cs.add_flags(card)
    assert not out["flag"].any()
    assert out.loc[out["provisional"], "residual"].isna().all()


def test_scorecard_end_to_end_on_the_synthetic_frame(requests_frame):
    communities = pd.DataFrame([{"comm_code": "X", "sector": "EAST", "srg": "ESTABLISHED"}])
    asof = pd.Timestamp("2025-06-01", tz="UTC")
    card = cs.scorecard(requests_frame, communities, asof)
    assert list(card.columns) == cs.COLUMNS
    assert set(card["series"]) >= {"city_late_mix_fixed", "sector_obs_expected"}
    assert card["value"].dropna().between(0, 50).all()
    flagged = cs.add_flags(card)
    assert "flag" in flagged and not flagged.loc[flagged["provisional"], "flag"].any()
