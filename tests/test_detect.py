"""Stage 7 detectors (D-045): the 2× rule, seasonal residuals, the EWMA chart, sustained runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitor import detect


def test_ratio_flags_one_slow_job_against_its_own_kind():
    values = pd.Series([100, 110, 95, 250, 30, 31, 90])
    kinds = pd.Series(["train", "train", "train", "train", "score", "score", "score"])
    flags = detect.ratio_flags(values, kinds)
    assert flags.tolist() == [False, False, False, True, False, False, True]


def test_ratio_flags_needs_two_others_to_judge():
    values = pd.Series([10.0, 100.0])
    assert not detect.ratio_flags(values, pd.Series(["a", "a"])).any()


def _monthly(values, start="2021-01"):
    return pd.Series(values, index=pd.period_range(start, periods=len(values), freq="M"))


def test_seasonal_residuals_compare_with_the_same_month_in_earlier_years():
    # a pure seasonal pattern repeats exactly: every residual after year one is zero
    season = [10, 20, 30, 40, 50, 60, 70, 60, 50, 40, 30, 20]
    resid = detect.seasonal_residuals(_monthly(season * 3))
    assert resid.iloc[:12].isna().all()
    assert np.allclose(resid.iloc[12:], 0.0)


def test_seasonal_residuals_use_the_mean_of_all_earlier_same_months():
    s = _monthly([1.0] * 12 + [3.0] * 12 + [5.0] * 12)
    resid = detect.seasonal_residuals(s)
    assert resid.iloc[12] == 2.0  # 3 - mean(1)
    assert resid.iloc[24] == 3.0  # 5 - mean(1, 3)


def test_ewma_quiet_on_noise_and_flags_a_level_shift():
    rng = np.random.default_rng(0)
    calm = pd.Series(rng.normal(0, 1, 40))
    assert not detect.sustained(detect.ewma_flags(calm)).any()
    shifted = pd.concat([calm, pd.Series(rng.normal(4, 1, 10))], ignore_index=True)
    flags = detect.sustained(detect.ewma_flags(shifted))
    assert flags.iloc[40:].sum() >= 7 and not flags.iloc[:40].any()


def test_ewma_refuses_too_short_a_history():
    assert not detect.ewma_flags(pd.Series([0, 0, 0, 50.0])).any()


def test_sustained_keeps_only_runs():
    flags = pd.Series([True, False, True, True, False, True, True, True, True])
    assert detect.sustained(flags, run=3).tolist() == [False] * 5 + [True] * 4
