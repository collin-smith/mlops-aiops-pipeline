"""Stage 6 shadow scoring: score today's open requests with a model that isn't approved for use.

The Stage 5 challenger failed the fairness gate, and the floor stayed at 0.8 (D-042). It
still scores real requests, in shadow (D-044): every row it writes says
``usage = shadow-not-for-use``, nothing operational reads ``scored/``, and Stage 7 grades
the scores against how the requests actually turn out.

Which requests get a score: open ones still **inside** their category's deadline at the
snapshot. An open request already past its deadline is already late (``build_labels``
labels it 1), so a score there forecasts nothing and would flatter Stage 7's precision.

    python -m src.deploy.score --model <dir with model.tar.gz> --data <parquet dir> \
        --communities <dir with communities.json> --out <dir> \
        --model-ref <package ARN> --model-tag v1

The features are the model's own (``feature_schema.json``), computed as of each request's
filing time, the way training saw them. The history features only count outcomes decided
before that time, which is what makes scoring live data legitimate. The flag is the top
10% within each sector (D-042's operating point).

Writes ``scores/scores-<tag>.parquet`` (one row per scored request) and
``report/summary.json`` under ``--out``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.features.build_features import align_to_schema
from src.features.build_labels import assert_no_leakage
from src.features.history import sector_lookup
from src.pipeline.evaluate import COMMUNITIES_FILE, load_artifact, predict
from src.pipeline.train import featurize, load_requests
from src.promote import fairness

USAGE = "shadow-not-for-use"
SUMMARY_FILE = "summary.json"
# The rolling counts look back 30 days before each request; the context frame starts a
# little earlier than the oldest scored request so its counts see a full window.
CONTEXT_DAYS = 31
OUTPUT_COLUMNS = [
    "usage",
    "service_request_id",
    "service_name",
    "comm_code",
    "comm_name",
    "sector",
    "srg",
    "requested_date",
    "deadline",
    "threshold_days",
    "category_seen",
    "score",
    "rank_in_sector",
    "flagged",
    "model_ref",
    "snapshot_asof",
    "scored_at",
]


def _requested(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["requested_date"], errors="coerce", utc=True, format="ISO8601")


def snapshot_time(raw: pd.DataFrame) -> pd.Timestamp:
    """When the snapshot was taken: the day after its newest request (as ``build_labels``)."""
    return _requested(raw).max().normalize() + pd.Timedelta(days=1)


def scoring_rows(raw: pd.DataFrame, thresholds: pd.Series, asof: pd.Timestamp) -> pd.DataFrame:
    """Open requests still inside their deadline at ``asof``, with that deadline.

    The exact complement of ``build_labels.overdue_open`` among open requests: that one
    takes age > threshold (already late), this one age <= threshold (not decided yet).
    A category the model never saw gets the global threshold, as in the label.
    """
    req = _requested(raw)
    global_thr = float(thresholds.get("__global__", thresholds.median()))
    thr = raw["service_name"].map(thresholds).fillna(global_thr)
    deadline = req + pd.to_timedelta(thr * 86_400, unit="s")
    keep = raw["closed_date"].isna() & req.notna() & (req <= asof) & (deadline >= asof)
    seen = set(thresholds.index) - {"__global__"}
    out = raw.loc[keep].copy()
    out["threshold_days"] = thr[keep]
    out["deadline"] = deadline[keep]
    out["category_seen"] = out["service_name"].isin(seen)
    return out


def context_frame(raw: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The requests the scored rows' rolling counts look across.

    Training computed those counts over its whole split, closed and open alike, so scoring
    computes them over the whole snapshot from a little before the oldest scored request,
    not over the open requests alone, which would undercount every one of them.
    """
    req = _requested(raw)
    start = _requested(rows).min() - pd.Timedelta(days=CONTEXT_DAYS)
    return raw.loc[req >= start]


def flag(scored: pd.DataFrame, top_fraction: float = 0.10) -> pd.DataFrame:
    """Rank within sector and flag the top ``top_fraction`` of each (D-042)."""
    t = fairness.FairnessThresholds(top_fraction=top_fraction)
    out = scored.copy()
    out["flagged"] = fairness.operating_point(out, t).to_numpy()
    out["rank_in_sector"] = (
        out.groupby("sector")["pred"].rank(method="first", ascending=False).astype("int64")
    )
    return out


def score(
    model_dir: Path,
    raw: pd.DataFrame,
    communities: pd.DataFrame,
    model_ref: str,
    *,
    asof: pd.Timestamp | None = None,
    scored_at: datetime | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Shadow scores for the snapshot's open, undecided requests, and a summary."""
    booster, schema, thresholds = load_artifact(model_dir)
    feature_set = schema.get("feature_set", "baseline")
    asof = asof if asof is not None else snapshot_time(raw)
    rows = scoring_rows(raw, thresholds, asof)
    if rows.empty:
        raise SystemExit(f"no open requests inside their deadline at {asof:%Y-%m-%d}")

    context = context_frame(raw, rows)
    sectors = sector_lookup(communities)
    X = featurize(context, raw, thresholds, sectors, feature_set)
    X = align_to_schema(X.loc[rows.index], schema)  # the rows' own order, so scores line up
    assert_no_leakage(X)

    scored = fairness.attach_groups(rows, communities).assign(pred=predict(booster, X))
    scored = flag(scored)
    stamp = (scored_at or datetime.now(UTC)).replace(microsecond=0)
    out = scored.rename(columns={"pred": "score"}).assign(
        usage=USAGE,
        model_ref=model_ref,
        snapshot_asof=asof.strftime("%Y-%m-%d"),
        scored_at=stamp.isoformat(),
        requested_date=_requested(scored),
    )[OUTPUT_COLUMNS]
    return out.reset_index(drop=True), summarize(out, asof, feature_set, model_ref)


def summarize(out: pd.DataFrame, asof: pd.Timestamp, feature_set: str, model_ref: str) -> dict:
    by_sector = out.groupby("sector").agg(scored=("score", "size"), flagged=("flagged", "sum"))
    return {
        "usage": USAGE,
        "model_ref": model_ref,
        "feature_set": feature_set,
        "snapshot_asof": asof.strftime("%Y-%m-%d"),
        "rows_scored": len(out),
        "rows_flagged": int(out["flagged"].sum()),
        "unseen_category_share": round(float((~out["category_seen"]).mean()), 4),
        "score_mean": round(float(out["score"].mean()), 4),
        "by_sector": {
            s: {"scored": int(r.scored), "flagged": int(r.flagged)} for s, r in by_sector.iterrows()
        },
    }


def write_outputs(out: pd.DataFrame, summary: dict, out_dir: Path, model_tag: str) -> Path:
    """``scores/scores-<tag>.parquet`` (what Athena reads) and ``report/summary.json`` (what
    the launcher prints). Separate folders, so the table's location holds only Parquet."""
    scores_dir, report_dir = out_dir / "scores", out_dir / "report"
    scores_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = scores_dir / f"scores-{model_tag}.parquet"
    # millisecond timestamps: the Parquet unit every reader handles
    out.to_parquet(path, index=False, coerce_timestamps="ms", allow_truncated_timestamps=True)
    (report_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True, help="directory holding model.tar.gz")
    ap.add_argument("--data", type=Path, required=True, help="processed Parquet directory")
    ap.add_argument(
        "--communities", type=Path, required=True, help=f"directory holding {COMMUNITIES_FILE}"
    )
    ap.add_argument("--out", type=Path, required=True, help="where scores/ and report/ go")
    ap.add_argument("--model-ref", required=True, help="the shadow model package ARN (lineage)")
    ap.add_argument("--model-tag", required=True, help="short name for the file, e.g. v1")
    args = ap.parse_args(argv)

    communities = fairness.load_communities(args.communities / COMMUNITIES_FILE)
    raw = load_requests(args.data)
    out, summary = score(args.model, raw, communities, args.model_ref)
    write_outputs(out, summary, args.out, args.model_tag)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
