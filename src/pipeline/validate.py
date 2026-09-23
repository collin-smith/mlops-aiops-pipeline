"""Data-validation gate — the first thing a pipeline run does with a raw pull.

Without this gate, a bad pull trains quietly. The exploratory pass found one
example already: Montgomery's pothole requests average 196 days to close, against
single digits for most communities. That is almost certainly a bulk backfill close,
not 196 days of waiting, and it would feed straight into ``comm_name`` and the
per-category thresholds.

Every check has a severity:

* ``fail`` — the data is structurally wrong (missing columns, a truncated pull, a stale
  snapshot, broken dates). The step exits non-zero, so the SageMaker Processing step
  fails and the pipeline stops before any training spend.
* ``warn`` — the data is usable but something looks like an artifact (a bulk close, a
  city-wide purge of long-open tickets, a community far off its category's norm). It is
  recorded in ``validation.json`` next to the run's outputs. ``--strict`` turns warnings
  into failures.

Checks are plain pandas over the raw frame, with no AWS calls, so they are unit-tested
on a local fixture (``tests/test_validate.py``) the same way the feature code is.

The default thresholds are starting values. Calibrate them on the first real pull,
where the 5-year window is ~2.9M rows.

Run: ``python -m src.pipeline.validate --input <parquet dir> --output <dir> [--asof YYYY-MM-DD]``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.common.config import INTAKE_COLUMNS

log = logging.getLogger(__name__)

REPORT_NAME = "validation.json"

# Columns the label and features can't be built without. ``closed_date`` is not an
# intake column, but the label needs it.
REQUIRED_COLUMNS: tuple[str, ...] = (*INTAKE_COLUMNS, "closed_date")


class DataValidationError(RuntimeError):
    """Raised when at least one ``fail`` check (or any check, under ``strict``) fails."""


@dataclass(frozen=True)
class Thresholds:
    # Volume. The 5-year window is ~2.9M rows, so 500k only catches a truncated pull.
    min_rows: int = 500_000
    # Compared with the previous run's report, when one is passed in.
    max_row_change: float = 0.25
    # service_request_id is the join key for scoring output, so it must be unique.
    max_duplicate_id_rate: float = 0.001
    # Per-column null ceilings. requested_date and service_name drive the label.
    max_null_rate: dict[str, float] = field(
        default_factory=lambda: {
            "requested_date": 0.0,
            "service_name": 0.001,
            "agency_responsible": 0.01,
            "source": 0.01,
            "comm_name": 0.10,  # some requests are city-wide and have no community
        }
    )
    max_unparseable_date_rate: float = 0.001
    # closed before requested. labelled_subset drops these; many of them means a bad pull.
    max_negative_duration_rate: float = 0.005
    # The newest request must be no more than this many days before the as-of date.
    max_staleness_days: int = 14
    # Artifact detectors (warn), over (service_name, comm_name) groups with enough rows.
    group_min_rows: int = 200
    # A group whose closures pile onto one calendar day: the backfill-close signature.
    bulk_close_day_share: float = 0.25
    bulk_close_min_rows: int = 50
    # A group whose mean days_to_close is this many times its category's median group
    # mean, AND at least this many days more. Montgomery is ~40× the median; the slowest
    # genuine communities (Bowness, 37 days) are nearer 7×.
    group_mean_ratio: float = 10.0
    group_mean_min_excess_days: float = 30.0
    # City-wide stale bulk close: at least this many tickets of one category closed on one
    # day, each older than both limits below. The 2025-08-26 traffic-signs purge closed
    # 3,513 tickets with a median age of 1,085 days (category median: 20); Montgomery's
    # 2024-10-20 backfill closed 38 pothole tickets from 2021-23.
    stale_close_min_rows: int = 30
    stale_close_age_ratio: float = 10.0  # × the category's median days_to_close
    # 120, not 90: "311 Contact Us" closes batches at 91-98 days old, which looks like an
    # automatic 90-day close, a policy rather than a cleanup. On the 2026-09-23 pull, 120
    # drops those (577 -> 0) and keeps every Montgomery and signs-purge ticket: 14,043
    # tickets flagged (0.48%) in 182 category-days across 60 categories.
    stale_close_min_age_days: float = 120.0


@dataclass
class CheckResult:
    name: str
    severity: str  # "fail" | "warn"
    passed: bool
    detail: str
    observed: dict = field(default_factory=dict)


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")


# --- structural checks (fail) ---------------------------------------------------------


def check_schema(df: pd.DataFrame) -> CheckResult:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return CheckResult(
        "schema",
        "fail",
        not missing,
        f"missing columns: {missing}" if missing else "all required columns present",
        {"missing": missing},
    )


def check_row_count(df: pd.DataFrame, t: Thresholds, baseline_rows: int | None) -> CheckResult:
    n = len(df)
    problems = []
    if n < t.min_rows:
        problems.append(f"{n:,} rows < minimum {t.min_rows:,}")
    change = None
    if baseline_rows:
        change = (n - baseline_rows) / baseline_rows
        if abs(change) > t.max_row_change:
            problems.append(
                f"{change:+.1%} vs previous run ({baseline_rows:,} rows), "
                f"limit ±{t.max_row_change:.0%}"
            )
    return CheckResult(
        "row_count",
        "fail",
        not problems,
        "; ".join(problems) or f"{n:,} rows",
        {"rows": n, "baseline_rows": baseline_rows, "change": change},
    )


def check_unique_ids(df: pd.DataFrame, t: Thresholds) -> CheckResult:
    dup = int(df["service_request_id"].duplicated().sum())
    rate = dup / max(len(df), 1)
    return CheckResult(
        "unique_ids",
        "fail",
        rate <= t.max_duplicate_id_rate,
        f"{dup:,} duplicate service_request_id ({rate:.3%}), limit {t.max_duplicate_id_rate:.3%}",
        {"duplicates": dup, "rate": rate},
    )


def check_null_rates(df: pd.DataFrame, t: Thresholds) -> CheckResult:
    rates = {c: float(df[c].isna().mean()) for c in t.max_null_rate if c in df.columns}
    over = {c: r for c, r in rates.items() if r > t.max_null_rate[c]}
    detail = (
        "; ".join(f"{c} {r:.2%} > {t.max_null_rate[c]:.2%}" for c, r in over.items())
        or "null rates within limits"
    )
    return CheckResult("null_rates", "fail", not over, detail, {"rates": rates})


def check_dates(df: pd.DataFrame, t: Thresholds, asof: pd.Timestamp) -> CheckResult:
    req = _dates(df["requested_date"])
    clo = _dates(df["closed_date"])
    n = max(len(df), 1)
    # A value that was present but didn't parse, not one that was null to begin with.
    bad_req = int((req.isna() & df["requested_date"].notna()).sum())
    bad_clo = int((clo.isna() & df["closed_date"].notna()).sum())
    unparseable = (bad_req + bad_clo) / n
    negative = float(((clo - req).dt.total_seconds() < 0).sum()) / n
    future = int((req > asof + pd.Timedelta(days=1)).sum())
    newest = req.max()
    stale_days = (asof - newest).days if pd.notna(newest) else None

    problems = []
    if unparseable > t.max_unparseable_date_rate:
        problems.append(f"{unparseable:.2%} unparseable dates")
    if negative > t.max_negative_duration_rate:
        problems.append(f"{negative:.2%} closed before requested")
    if future:
        problems.append(f"{future:,} requests dated after as-of {asof:%Y-%m-%d}")
    if stale_days is None or stale_days > t.max_staleness_days:
        problems.append(
            f"newest request is {stale_days} days before as-of, limit {t.max_staleness_days}"
        )
    return CheckResult(
        "dates",
        "fail",
        not problems,
        "; ".join(problems) or "dates parse, order and freshness OK",
        {
            "unparseable_rate": unparseable,
            "negative_duration_rate": negative,
            "future_rows": future,
            "newest_requested": None if pd.isna(newest) else newest.isoformat(),
            "stale_days": stale_days,
        },
    )


# --- artifact detectors (warn) --------------------------------------------------------


def _closed_groups(df: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    req = _dates(df["requested_date"])
    clo = _dates(df["closed_date"])
    out = pd.DataFrame(
        {
            "service_name": df["service_name"],
            "comm_name": df["comm_name"],
            "closed_day": clo.dt.floor("D"),
            "days_to_close": (clo - req).dt.total_seconds() / 86_400.0,
        }
    ).dropna()
    out = out[out["days_to_close"] >= 0]
    size = out.groupby(["service_name", "comm_name"])["days_to_close"].transform("size")
    return out[size >= t.group_min_rows]


def check_bulk_close(df: pd.DataFrame, t: Thresholds) -> CheckResult:
    """A community/category whose closures pile onto one day: a backfill, not a fix."""
    g = _closed_groups(df, t)
    flagged = []
    for (svc, comm), grp in g.groupby(["service_name", "comm_name"], sort=False):
        per_day = grp["closed_day"].value_counts()
        top_day, top_n = per_day.index[0], int(per_day.iloc[0])
        share = top_n / len(grp)
        if share >= t.bulk_close_day_share and top_n >= t.bulk_close_min_rows:
            flagged.append(
                {
                    "service_name": svc,
                    "comm_name": comm,
                    "rows": len(grp),
                    "closed_day": top_day.strftime("%Y-%m-%d"),
                    "closed_that_day": top_n,
                    "share": round(share, 3),
                    "mean_days_to_close": round(float(grp["days_to_close"].mean()), 1),
                }
            )
    detail = (
        "; ".join(
            f"{f['comm_name']} / {f['service_name']}: {f['share']:.0%} closed on {f['closed_day']}"
            for f in flagged
        )
        or "no single-day bulk closes"
    )
    return CheckResult("bulk_close", "warn", not flagged, detail, {"groups": flagged})


def stale_bulk_close_mask(df: pd.DataFrame, t: Thresholds | None = None) -> pd.Series:
    """True for each ticket closed in a stale bulk close: one of at least
    ``stale_close_min_rows`` tickets of its category closed on the same day, each far
    older than that category normally takes.

    It's a records cleanup, not service. The breach label of these tickets is probably
    right (they really were open that long), but their days_to_close is not, so duration
    statistics and category thresholds should leave them out. The Stage 2 label reuses this
    mask, so the gate and the label share one definition.
    """
    t = t or Thresholds()
    req = _dates(df["requested_date"])
    clo = _dates(df["closed_date"])
    days = (clo - req).dt.total_seconds() / 86_400.0
    cat_median = days.where(days >= 0).groupby(df["service_name"]).transform("median")
    limit = (cat_median * t.stale_close_age_ratio).clip(lower=t.stale_close_min_age_days)
    stale = days >= limit
    day = clo.dt.floor("D")
    per_day = stale.groupby([df["service_name"], day]).transform("sum")
    return (stale & (per_day >= t.stale_close_min_rows)).fillna(False).astype(bool)


def check_stale_bulk_close(df: pd.DataFrame, t: Thresholds) -> CheckResult:
    """City-wide: one day on which many long-open tickets of a category were closed at once."""
    mask = stale_bulk_close_mask(df, t)
    flagged = []
    if mask.any():
        hit = df.loc[mask]
        req = _dates(hit["requested_date"])
        clo = _dates(hit["closed_date"])
        frame = pd.DataFrame(
            {
                "service_name": hit["service_name"],
                "closed_day": clo.dt.strftime("%Y-%m-%d"),
                "age": (clo - req).dt.total_seconds() / 86_400.0,
                "comm_name": hit["comm_name"],
            }
        )
        for (svc, day), grp in frame.groupby(["service_name", "closed_day"], sort=True):
            flagged.append(
                {
                    "service_name": svc,
                    "closed_day": day,
                    "tickets": len(grp),
                    "communities": int(grp["comm_name"].nunique()),
                    "median_age_days": round(float(grp["age"].median()), 1),
                }
            )
    # Log the biggest events; validation.json keeps every one.
    biggest = sorted(flagged, key=lambda f: f["tickets"], reverse=True)[:5]
    detail = "; ".join(
        f"{f['service_name']} on {f['closed_day']}: {f['tickets']:,} tickets, "
        f"median age {f['median_age_days']:.0f}d"
        for f in biggest
    )
    if len(flagged) > len(biggest):
        detail += f"; and {len(flagged) - len(biggest)} smaller events"
    if flagged:
        detail = f"{int(mask.sum()):,} tickets in {len(flagged)} events. Largest: {detail}"
    return CheckResult(
        "stale_bulk_close",
        "warn",
        not flagged,
        detail or "no stale bulk closes",
        {"events": flagged, "tickets": int(mask.sum())},
    )


def check_group_outliers(df: pd.DataFrame, t: Thresholds) -> CheckResult:
    """A community far slower than every other community for the same category."""
    g = _closed_groups(df, t)
    means = g.groupby(["service_name", "comm_name"])["days_to_close"].mean().rename("mean")
    means = means.reset_index()
    means["cat_median"] = means.groupby("service_name")["mean"].transform("median")
    hit = means[
        (means["mean"] >= t.group_mean_ratio * means["cat_median"])
        & (means["mean"] - means["cat_median"] >= t.group_mean_min_excess_days)
    ]
    flagged = [
        {
            "service_name": r.service_name,
            "comm_name": r.comm_name,
            "mean_days_to_close": round(float(r.mean), 1),
            "category_median_days": round(float(r.cat_median), 1),
        }
        for r in hit.itertuples()
    ]
    detail = (
        "; ".join(
            f"{f['comm_name']} / {f['service_name']}: {f['mean_days_to_close']}d "
            f"vs {f['category_median_days']}d typical"
            for f in flagged
        )
        or "no community far outside its category's norm"
    )
    return CheckResult("group_outliers", "warn", not flagged, detail, {"groups": flagged})


# --- gate -----------------------------------------------------------------------------


def validate(
    df: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    thresholds: Thresholds | None = None,
    baseline_rows: int | None = None,
) -> list[CheckResult]:
    t = thresholds or Thresholds()
    schema = check_schema(df)
    if not schema.passed:  # nothing else can run without the columns
        return [schema]
    return [
        schema,
        check_row_count(df, t, baseline_rows),
        check_unique_ids(df, t),
        check_null_rates(df, t),
        check_dates(df, t, asof),
        check_bulk_close(df, t),
        check_stale_bulk_close(df, t),
        check_group_outliers(df, t),
    ]


def build_report(results: list[CheckResult], *, asof: pd.Timestamp, strict: bool) -> dict:
    failed = [r.name for r in results if not r.passed and (r.severity == "fail" or strict)]
    warned = [r.name for r in results if not r.passed and r.severity == "warn" and not strict]
    rows = next((r.observed["rows"] for r in results if r.name == "row_count"), None)
    return {
        "asof": asof.strftime("%Y-%m-%d"),
        "strict": strict,
        "passed": not failed,
        "failed": failed,
        "warnings": warned,
        "rows": rows,
        "checks": [asdict(r) for r in results],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, help="directory of raw Parquet")
    p.add_argument("--output", required=True, help=f"directory to write {REPORT_NAME} into")
    p.add_argument("--asof", help="snapshot date YYYY-MM-DD (default: today, UTC)")
    p.add_argument("--baseline", help=f"previous run's {REPORT_NAME}, for the row-count check")
    p.add_argument("--strict", action="store_true", help="treat warnings as failures")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, *, thresholds: Thresholds | None = None) -> dict:
    """Validate, write the report, and raise ``DataValidationError`` if the gate fails.

    The report is written before raising, so a failed run still leaves the evidence.
    """
    args = _parse_args(argv)
    asof = pd.Timestamp(args.asof, tz="UTC") if args.asof else pd.Timestamp.now(tz="UTC")
    asof = asof.normalize()

    df = pd.read_parquet(args.input)
    baseline_rows = None
    if args.baseline and Path(args.baseline).exists():
        baseline_rows = json.loads(Path(args.baseline).read_text()).get("rows")

    results = validate(df, asof=asof, thresholds=thresholds, baseline_rows=baseline_rows)
    report = build_report(results, asof=asof, strict=args.strict)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_NAME).write_text(json.dumps(report, indent=2, default=str))

    for r in results:
        status = "PASS" if r.passed else r.severity.upper()
        log.info("%-5s %-15s %s", status, r.name, r.detail)
    if not report["passed"]:
        raise DataValidationError(f"data validation failed: {report['failed']}")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except DataValidationError as e:
        log.error("%s", e)
        sys.exit(1)
