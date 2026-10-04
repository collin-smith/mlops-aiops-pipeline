"""The two detectors Stage 7 shares between pipeline health and the civic scorecard (D-045).

* ``ratio_flags``: a stated rule for very short histories. A value is flagged when it's
  more than ``factor`` times the median of the *other* values of its kind. The pipeline
  has a handful of jobs per kind, too few for any fitted model to mean anything.
* ``ewma_flags`` over ``seasonal_residuals``: an EWMA control chart, for series long enough
  to have one. Each month is compared with the same calendar month in earlier years, not
  with last month, so January doesn't fire every year. ``sustained`` then keeps only runs
  of consecutive flags, because a single noisy month is not a shift.

A flag says a number moved, not why. Everything here is a reason to look.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATIO_FACTOR = 2.0
EWMA_LAMBDA = 0.3
EWMA_L = 3.0
# The EWMA's in-control mean and spread come from the first this-many points. Below this,
# the chart refuses to run and says so, rather than draw limits from almost nothing.
EWMA_MIN_POINTS = 10
SUSTAINED_RUN = 3


def ratio_flags(values: pd.Series, kinds: pd.Series, factor: float = RATIO_FACTOR) -> pd.Series:
    """True where a value exceeds ``factor`` × the median of the other values of its kind.

    A kind needs at least two other values to judge one; with fewer, nothing is flagged.
    """
    flags = pd.Series(False, index=values.index)
    for _, idx in values.groupby(kinds).groups.items():
        group = values.loc[idx]
        if len(group) < 3:
            continue
        for i in group.index:
            others = group.drop(i)
            flags.loc[i] = bool(group.loc[i] > factor * others.median())
    return flags


def seasonal_residuals(values: pd.Series) -> pd.Series:
    """Each month minus the mean of the same calendar month in earlier years.

    ``values`` is indexed by month (a ``Period`` or timestamp index). Months with no earlier
    same-month value are NaN: the first year of any series has nothing to compare with.
    """
    s = values.sort_index()
    months = pd.PeriodIndex(s.index, freq="M")
    out = pd.Series(np.nan, index=s.index)
    for i, m in enumerate(months):
        prior = s.iloc[:i][(months[:i].month == m.month) & (months[:i].year < m.year)]
        prior = prior.dropna()
        if len(prior) and pd.notna(s.iloc[i]):
            out.iloc[i] = s.iloc[i] - prior.mean()
    return out


def ewma_flags(
    values: pd.Series,
    lam: float = EWMA_LAMBDA,
    L: float = EWMA_L,
    min_points: int = EWMA_MIN_POINTS,
) -> pd.Series:
    """True where the EWMA of ``values`` leaves its control limits.

    The target and sigma come from the first ``min_points`` non-missing values (the
    in-control period); the limits use the exact time-varying EWMA variance. Missing
    values are skipped and stay unflagged.
    """
    x = values.dropna()
    flags = pd.Series(False, index=values.index)
    if len(x) < min_points:
        return flags
    base = x.iloc[:min_points]
    mu, sigma = float(base.mean()), float(base.std(ddof=1))
    if not sigma or np.isnan(sigma):
        return flags
    z = mu
    for t, (i, v) in enumerate(x.items(), start=1):
        z = lam * float(v) + (1 - lam) * z
        half_width = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
        flags.loc[i] = abs(z - mu) > half_width
    return flags


def sustained(flags: pd.Series, run: int = SUSTAINED_RUN) -> pd.Series:
    """Keep only flags that are part of ``run`` or more consecutive flags."""
    f = flags.astype(bool)
    block = (f != f.shift()).cumsum()
    length = f.groupby(block).transform("size")
    return f & (length >= run)
