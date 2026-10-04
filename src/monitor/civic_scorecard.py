"""The companion piece's findings as monthly series that can be recomputed (Stage 7 Layer C, D-045).

Stage 8 and the companion piece publish findings from one snapshot. This turns each into a
number per month, backfilled from 2021, so a finding can be re-checked on the next snapshot
instead of frozen. Five series, kept short on purpose (more series, more false alarms):

1. ``city_late_mix_fixed``: the city late rate, request mix held fixed at the all-years mix
2. ``sector_obs_expected``: per sector, late rate over the rate expected from its mix (key =
   sector; expected = the same month's city-wide rate for each request type)
3. ``waste_east_late``: residential waste in the EAST sector, late rate
4. ``fri_sat_vs_mon_thu``: requests filed Friday or Saturday against Monday to Thursday,
   each as observed over expected for the same type in the same sector
5. ``slowest5_share``: the share of all waiting days owed to the slowest 5% of closed
   requests (purges out), over the 12 months of filings ending that month

"Late" is the model's label: slower than the type's own p75 (``build_labels``), not a City
standard. A request counts only once its outcome is settled: closed, or open past its
deadline, and its deadline has passed by the snapshot (D-037's rule; otherwise on-time
closures arrive first and recent months look better than they were). Months still waiting
on more than ``PROVISIONAL_SHARE`` of their requests (``SLOW_PENDING_SHARE`` for series 5) are
marked ``provisional`` and kept out of detection.

Each series then runs through ``detect``: compared with the same month in earlier years,
an EWMA chart, and only runs of three flags kept. A flag is a reason to look, not a finding.

    python -m src.monitor.civic_scorecard --data data/processed/311 \
        --communities data/raw/communities/asof=2026-10-01 --out scorecard.csv \
        [--alert --topic-arn <alerts topic>]

It runs locally ($0); copy the CSV to ``s3://<bucket>/monitoring/scorecard/`` afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_labels import build_training_labels
from src.monitor import detect
from src.pipeline.train import COMMUNITIES_FILE, load_requests
from src.pipeline.validate import stale_bulk_close_mask
from src.promote import fairness

WASTE = "WRS - Waste - Residential"
# A month is provisional while more of its requests than this are still before their deadline.
# The deadline-passed rule already keeps each rate unbiased; what's pending is whole request
# types with long deadlines (signals, signs: months), which only shift a month's mix. So 6%,
# which on the 2026-10-01 snapshot settles through 2026-07. The waiting-days series is
# stricter (SLOW_PENDING_SHARE): there the pending requests are the open ones, i.e. the slow.
PROVISIONAL_SHARE = 0.06
SLOW_PENDING_SHARE = 0.02
WINDOW_MONTHS = 12
SLOWEST = 0.05
COLUMNS = ["month", "series", "key", "value", "n", "provisional"]


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")


def _month(t: pd.Series) -> pd.Series:
    return t.dt.tz_localize(None).dt.to_period("M")


def settled(raw: pd.DataFrame, communities: pd.DataFrame, asof: pd.Timestamp):
    """The labelled requests whose deadline passed by ``asof``, plus each month's unsettled
    share (requests of any kind whose deadline hadn't passed yet)."""
    train, test, thresholds = build_training_labels(raw, asof=asof)
    lab = pd.concat([train, test], ignore_index=True)
    req = _dt(lab["requested_date"])
    deadline = req + pd.to_timedelta(lab["threshold_days"] * 86_400, unit="s")
    keep = deadline < asof
    lab = fairness.attach_groups(lab.loc[keep], communities, ("sector",))
    lab["month"] = _month(req[keep])
    lab["dow"] = req[keep].dt.dayofweek

    raw_req = _dt(raw["requested_date"])
    global_thr = float(thresholds.get("__global__", thresholds.median()))
    thr = raw["service_name"].map(thresholds).fillna(global_thr)
    raw_deadline = raw_req + pd.to_timedelta(thr * 86_400, unit="s")
    pending = (raw_deadline >= asof).groupby(_month(raw_req)).mean()
    return lab.reset_index(drop=True), pending


def _rows(series: str, frame: pd.DataFrame, key: str = "") -> pd.DataFrame:
    """``frame`` (index month, columns value and n) as long scorecard rows."""
    out = frame.reset_index().rename(columns={"index": "month"})
    return out.assign(series=series, key=key)


def city_late_mix_fixed(lab: pd.DataFrame) -> pd.DataFrame:
    weights = lab["service_name"].value_counts(normalize=True)
    rows = {}
    for month, g in lab.groupby("month"):
        by = g.groupby("service_name")["breach"].mean()
        w = weights[by.index]
        rows[month] = {"value": float((by * w).sum() / w.sum()), "n": len(g)}
    return _rows("city_late_mix_fixed", pd.DataFrame.from_dict(rows, orient="index"))


def sector_obs_expected(lab: pd.DataFrame) -> pd.DataFrame:
    exp = lab.groupby(["month", "service_name"])["breach"].transform("mean")
    frame = lab.assign(expected=exp).loc[lambda d: d["sector"] != fairness.UNKNOWN]
    g = frame.groupby(["sector", "month"]).agg(
        late=("breach", "mean"), expected=("expected", "mean"), n=("breach", "size")
    )
    g["value"] = g["late"] / g["expected"].where(g["expected"] > 0)
    parts = [
        _rows("sector_obs_expected", sub.droplevel(0)[["value", "n"]], key=sector)
        for sector, sub in g.groupby(level=0)
    ]
    return pd.concat(parts, ignore_index=True)


def waste_east_late(lab: pd.DataFrame) -> pd.DataFrame:
    g = lab.loc[(lab["service_name"] == WASTE) & (lab["sector"] == "EAST")]
    agg = g.groupby("month")["breach"].agg(value="mean", n="size")
    return _rows("waste_east_late", agg, key="EAST")


def fri_sat_vs_mon_thu(lab: pd.DataFrame) -> pd.DataFrame:
    exp = lab.groupby(["service_name", "sector"])["breach"].transform("mean")
    frame = lab.assign(expected=exp)
    rows = {}
    for month, g in frame.groupby("month"):
        weekend = g.loc[g["dow"].isin([4, 5])]
        weekday = g.loc[g["dow"].isin([0, 1, 2, 3])]
        if weekend.empty or weekday.empty:
            continue
        oe_end = weekend["breach"].mean() / weekend["expected"].mean()
        oe_day = weekday["breach"].mean() / weekday["expected"].mean()
        rows[month] = {"value": float(oe_end / oe_day) if oe_day else np.nan, "n": len(weekend)}
    return _rows("fri_sat_vs_mon_thu", pd.DataFrame.from_dict(rows, orient="index"))


def slowest5_share(raw: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Over each trailing 12 months of filings: the slowest 5%'s share of waiting days.

    Closed requests only, purges out, as in the companion piece; the window's open share is
    returned as ``pending``, because the requests still open are the slow ones.
    """
    req, closed = _dt(raw["requested_date"]), _dt(raw["closed_date"])
    days = (closed - req).dt.total_seconds() / 86_400
    month = _month(req)
    is_open = raw["closed_date"].isna()
    keep = days.notna() & (days >= 0) & ~stale_bulk_close_mask(raw)
    frame = pd.DataFrame({"month": month[keep], "days": days[keep]})
    rows, pending = {}, {}
    months = pd.period_range(month.min() + WINDOW_MONTHS - 1, _month(pd.Series([asof]))[0] - 1)
    for m in months:
        lo = m - WINDOW_MONTHS + 1
        d = np.sort(frame.loc[frame["month"].between(lo, m), "days"].to_numpy())[::-1]
        if not len(d) or not d.sum():
            continue
        k = int(len(d) * SLOWEST)
        rows[m] = {"value": float(d[:k].sum() / d.sum()), "n": len(d)}
        pending[m] = float(is_open[month.between(lo, m)].mean())
    out = _rows("slowest5_share", pd.DataFrame.from_dict(rows, orient="index"))
    return out, pd.Series(pending)


def scorecard(raw: pd.DataFrame, communities: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Every series, one row per month (and key), with ``provisional`` set."""
    lab, pending = settled(raw, communities, asof)
    parts = [
        city_late_mix_fixed(lab),
        sector_obs_expected(lab),
        waste_east_late(lab),
        fri_sat_vs_mon_thu(lab),
    ]
    card = pd.concat(parts, ignore_index=True)
    card["provisional"] = card["month"].map(pending).fillna(1.0) > PROVISIONAL_SHARE
    slow, slow_pending = slowest5_share(raw, asof)
    slow["provisional"] = slow["month"].map(slow_pending).fillna(1.0) > SLOW_PENDING_SHARE
    card = pd.concat([card, slow], ignore_index=True)
    card["n"] = card["n"].astype("int64")
    return card[COLUMNS].sort_values(["series", "key", "month"]).reset_index(drop=True)


def add_flags(card: pd.DataFrame) -> pd.DataFrame:
    """``flag`` per row: a sustained EWMA excursion against the same months in earlier
    years. Provisional months are never judged and never flagged."""
    out = card.copy()
    out["residual"] = np.nan
    out["flag"] = False
    for _, idx in out.groupby(["series", "key"]).groups.items():
        rows = out.loc[idx]
        final = rows.loc[~rows["provisional"]].set_index("month")["value"]
        if final.empty:
            continue
        resid = detect.seasonal_residuals(final)
        flags = detect.sustained(detect.ewma_flags(resid))
        pos = rows.index[~rows["provisional"].to_numpy()]
        out.loc[pos, "residual"] = resid.to_numpy()
        out.loc[pos, "flag"] = flags.to_numpy()
    return out


def latest(card: pd.DataFrame) -> pd.DataFrame:
    """Each series' newest settled month, its value and whether it's flagged."""
    final = card.loc[~card["provisional"]]
    return final.sort_values("month").groupby(["series", "key"]).tail(1).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="processed Parquet directory")
    ap.add_argument(
        "--communities", type=Path, required=True, help=f"directory holding {COMMUNITIES_FILE}"
    )
    ap.add_argument("--out", type=Path, required=True, help="the scorecard CSV to write")
    ap.add_argument("--alert", action="store_true", help="publish recent flags to SNS")
    ap.add_argument("--topic-arn", help="the alerts topic (terraform output alerts_topic_arn)")
    args = ap.parse_args(argv)
    if args.alert and not args.topic_arn:
        ap.error("--alert needs --topic-arn")

    raw = load_requests(args.data)
    communities = fairness.load_communities(args.communities / COMMUNITIES_FILE)
    asof = _dt(raw["requested_date"]).max().normalize() + pd.Timedelta(days=1)
    card = add_flags(scorecard(raw, communities, asof))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    card.to_csv(args.out, index=False)
    now = latest(card)
    print(f"snapshot {asof:%Y-%m-%d}: {len(card):,} rows, {int(card['flag'].sum())} flagged")
    print(now[["series", "key", "month", "value", "n", "flag"]].to_string(index=False))
    recent = card.loc[card["flag"] & (card["month"] >= now["month"].max() - 5)]
    summary = {
        "asof": f"{asof:%Y-%m-%d}",
        "flagged_recent": recent[["series", "key"]].drop_duplicates().to_dict("records"),
    }
    print(json.dumps(summary, default=str))
    if args.alert and summary["flagged_recent"]:
        import boto3

        from src.common.config import get_config

        names = ", ".join(f"{r['series']} {r['key']}".strip() for r in summary["flagged_recent"])
        text = (
            f"Civic scorecard, snapshot {summary['asof']}: sustained shifts in {names}.\n"
            "Compared with the same months in earlier years. A flag is a reason to look, "
            "not a finding."
        )
        boto3.client("sns", region_name=get_config().region).publish(
            TopicArn=args.topic_arn, Subject="mlops-aiops: civic scorecard flag", Message=text
        )
        print("alert sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
