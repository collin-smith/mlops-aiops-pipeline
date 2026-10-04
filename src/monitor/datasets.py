"""The CSVs Model Monitor's analyzer compares, built locally (Stage 7 Layer A, D-045).

* ``baseline``: the challenger's training rows (the last two years of the training split),
  features as training saw them. The analyzer turns this into ``statistics.json`` and
  ``constraints.json``.
* ``intake``: every request filed in the ``--days`` before the snapshot, featurised as of
  its filing time with ``score.py``'s context window. That's the intake, which is what
  data drift is about.

Not the scored set. Shadow scoring only sees open requests still inside their deadline,
which leans hard towards slow categories filed recently. The analyzer would flag that,
correctly, and it would mean nothing about the city.

    python -m src.monitor.datasets baseline --data data/processed/311 \
        --communities data/raw/communities/asof=2026-10-01 --model <dir> --out baseline.csv
    python -m src.monitor.datasets intake --data data/processed/311 \
        --communities data/raw/communities/asof=2026-10-01 --model <dir> --out intake.csv

``--model`` is a directory holding the shadow model's ``model.tar.gz``; its
``thresholds.csv`` judges the history features, as in scoring. Without it the thresholds
are recomputed from the snapshot's training split, which matches the model's only while
the training split hasn't moved, and the script says so.
"""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

from src.deploy.score import context_frame, snapshot_time
from src.features.build_features import CATEGORICAL, FEATURE_SETS
from src.features.build_labels import assert_no_leakage, build_training_labels
from src.pipeline.train import (
    DEFAULT_TRAIN_YEARS,
    featurize,
    load_requests,
    load_sectors,
    recent,
)

COLUMNS = FEATURE_SETS["challenger"]
# The analyzer only tracks a distribution for low-cardinality strings: on the 2026-10-01
# baseline it kept one for agency_responsible (47 values) and source (4), and none for
# service_name (571) or comm_name (320), so the request mix itself went unwatched. This
# column (not a model feature) gives it a mix it can see: the department prefix of the
# service name, the 30 largest from that baseline (98.6% of requests), the rest "Other".
# Fixed here so the baseline and every later check map the same way.
DEPARTMENTS = (
    "Roads", "WRS", "Bylaw", "Parks", "WATS", "AT", "Corporate", "CT", "DBBS Inspection",
    "Finance", "CFD", "RSP", "DBBS", "AS", "311 Contact Us", "Recreation", "GFL", "CT AC",
    "ACPL", "Opinions on Business Units", "Community Safety", "Business Safety",
    "DBBS Concern", "PSD", "Law", "Active Living Program Application", "HR",
    "After Hours Transit", "VFH", "Customer Service & Communications",
)  # fmt: skip
OTHER = "Other"
CSV_COLUMNS = [*COLUMNS, "department"]
INTAKE_DAYS = 30
BASELINE_SAMPLE = 200_000  # the analyzer runs Spark on one ml.t3; see the smoke test
SEED = 42


def model_thresholds(model_dir: Path) -> pd.Series:
    """``thresholds.csv`` from ``model.tar.gz``, without loading the booster (no xgboost)."""
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(model_dir / "model.tar.gz") as tar:
        tar.extract("thresholds.csv", tmp, filter="data")
        return pd.read_csv(Path(tmp) / "thresholds.csv", index_col=0).iloc[:, 0]


def _requested(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["requested_date"], errors="coerce", utc=True, format="ISO8601")


def department(service_name: pd.Series) -> pd.Series:
    """The service name's prefix before " - ", one of ``DEPARTMENTS`` or "Other"."""
    prefix = service_name.astype("string").str.split(" - ").str[0].str.strip()
    return prefix.where(prefix.isin(DEPARTMENTS), OTHER).fillna(OTHER)


def _csv_ready(X: pd.DataFrame) -> pd.DataFrame:
    """Strings for the analyzer (a missing level stays empty), plus the department."""
    for c in CATEGORICAL:
        X[c] = X[c].astype("string")
    X["department"] = department(X["service_name"])
    return X[CSV_COLUMNS].reset_index(drop=True)


def intake_rows(raw: pd.DataFrame, asof: pd.Timestamp, days: int = INTAKE_DAYS) -> pd.DataFrame:
    """Every request filed in the ``days`` before ``asof``, open or closed."""
    req = _requested(raw)
    return raw.loc[(req > asof - pd.Timedelta(days=days)) & (req <= asof)]


def feature_table(rows, raw, thresholds, sectors) -> pd.DataFrame:
    """The challenger's features for ``rows``, as plain CSV-ready columns.

    The rolling counts look across the whole snapshot from a month before the oldest row
    (``score.context_frame``), so they match what training and scoring computed.
    """
    context = context_frame(raw, rows)
    X = featurize(context, raw, thresholds, sectors, "challenger").loc[rows.index, COLUMNS]
    assert_no_leakage(X)
    return _csv_ready(X)


def baseline_table(raw, thresholds, sectors, sample: int = BASELINE_SAMPLE) -> pd.DataFrame:
    """The challenger's training rows, sampled to ``sample`` (0 = all)."""
    train, _, _ = build_training_labels(raw)
    train = recent(train, DEFAULT_TRAIN_YEARS["challenger"])
    # Featurise the whole split, as training did, and sample afterwards: the rolling counts
    # count the rows they're given, so featurising a sample would undercount every one.
    X = featurize(train, raw, thresholds, sectors, "challenger")[COLUMNS]
    if sample and len(X) > sample:
        X = X.sample(n=sample, random_state=SEED).sort_index()
    assert_no_leakage(X)
    return _csv_ready(X)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["baseline", "intake"])
    ap.add_argument("--data", type=Path, required=True, help="processed Parquet directory")
    ap.add_argument("--communities", type=Path, required=True, help="communities.json's dir")
    ap.add_argument("--out", type=Path, required=True, help="the CSV to write")
    ap.add_argument("--model", type=Path, help="directory holding the model's model.tar.gz")
    ap.add_argument("--days", type=int, default=INTAKE_DAYS, help="intake window (days)")
    ap.add_argument("--sample", type=int, default=BASELINE_SAMPLE, help="baseline rows (0=all)")
    ap.add_argument("--rows", type=int, default=0, help="keep only the first N rows (smoke)")
    args = ap.parse_args(argv)

    raw = load_requests(args.data)
    sectors = load_sectors(args.communities)
    if args.model:
        thresholds = model_thresholds(args.model)
    else:
        print("no --model: thresholds recomputed from this snapshot's training split")
        thresholds = build_training_labels(raw)[2]

    if args.kind == "baseline":
        table = baseline_table(raw, thresholds, sectors, args.sample)
    else:
        asof = snapshot_time(raw)
        table = feature_table(intake_rows(raw, asof, args.days), raw, thresholds, sectors)
        print(f"intake: {len(table):,} requests filed in the {args.days} days to {asof:%Y-%m-%d}")
    if args.rows:
        table = table.head(args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table):,} rows × {table.shape[1]} columns to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
