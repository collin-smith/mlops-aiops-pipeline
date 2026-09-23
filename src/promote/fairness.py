"""Fairness check for the Stage 4 promotion gate: does the model find slow requests
equally well in every part of the city?

The operating point is the one the model is used at: flag the **top decile** of risk
scores, with one cutoff for the whole city. For each group of communities we measure:

* ``recall``    : of the requests that did breach, the share the model flagged. This is
                  the fairness criterion (equal opportunity). A sector where the model
                  misses slow requests is a sector where triage doesn't help.
* ``lift``      : breach rate among flagged requests ÷ the group's own breach rate. It
                  must stay well above 1 everywhere, or the flag means little in that group.
* ``base_rate`` and ``flag_rate`` are reported for context. They are not gated: base
                  rates really do differ (17–29% across communities), and when they do, no
                  model can equalise every fairness metric at once.

Groups come from the City's Community District Boundaries (``surr-xmvs``). They are not
address quadrants, since the open data has no quadrant field:

* ``sector`` : the City's 8 planning sectors (NORTHWEST, CENTRE, SOUTHEAST, …)
* ``srg``    : ESTABLISHED / DEVELOPING / …, which roughly follows the older-NW /
               newer-SE gap the exploratory pass found

Gate: in every dimension, the lowest group recall ÷ the highest must be at least
``min_recall_ratio`` (0.8, after the four-fifths rule of thumb), and every group's lift
at least ``min_group_lift``. Groups too small to measure are reported, not gated.
Requests with no community (``UNKNOWN``) are reported but never gated. Fewer than two
measurable groups is a failure: the gate can't vouch for what it can't measure.

This checks the model's *performance* across areas. It says nothing about why areas
wait different lengths of time. That is Stage 8's question, and the City hasn't
validated it.

Run: ``python -m src.promote.fairness --predictions <dir> --communities <json> --output <dir>``
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPORT_NAME = "fairness.json"
DIMENSIONS: tuple[str, ...] = ("sector", "srg")
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FairnessThresholds:
    top_fraction: float = 0.10  # the triage operating point: top decile, one city-wide cutoff
    min_group_rows: int = 1_000
    min_group_positives: int = 100
    min_recall_ratio: float = 0.80
    min_group_lift: float = 1.5


def attach_groups(
    df: pd.DataFrame, communities: pd.DataFrame, dims: tuple[str, ...] = DIMENSIONS
) -> pd.DataFrame:
    """Join each prediction's ``comm_code`` to its community's ``sector`` / ``srg``."""
    lookup = communities.drop_duplicates("comm_code").set_index("comm_code")
    out = df.copy()
    for dim in dims:
        out[dim] = out["comm_code"].map(lookup[dim]).fillna(UNKNOWN).astype(str)
    return out


def group_metrics(df: pd.DataFrame, dim: str, flagged: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"group": df[dim], "breach": df["breach"].astype(int), "flagged": flagged.astype(int)}
    )
    frame["hit"] = frame["breach"] & frame["flagged"]
    g = frame.groupby("group").agg(
        rows=("breach", "size"),
        positives=("breach", "sum"),
        flagged=("flagged", "sum"),
        hits=("hit", "sum"),
    )
    g["base_rate"] = g["positives"] / g["rows"]
    g["flag_rate"] = g["flagged"] / g["rows"]
    g["recall"] = g["hits"] / g["positives"].where(g["positives"] > 0)
    precision = g["hits"] / g["flagged"].where(g["flagged"] > 0)
    g["lift"] = precision / g["base_rate"].where(g["base_rate"] > 0)
    return g.reset_index()


def _clean(v):
    """JSON has no NaN; the report must parse in the pipeline's JsonGet."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return round(v, 4) if isinstance(v, float) else v


def fairness_report(
    df: pd.DataFrame,
    communities: pd.DataFrame,
    thresholds: FairnessThresholds | None = None,
    dims: tuple[str, ...] = DIMENSIONS,
) -> dict:
    """``df`` needs ``breach`` (0/1), ``pred`` (score) and ``comm_code``."""
    t = thresholds or FairnessThresholds()
    scored = attach_groups(df.dropna(subset=["pred", "breach"]), communities, dims)
    cutoff = float(scored["pred"].quantile(1 - t.top_fraction))
    flagged = scored["pred"] >= cutoff

    dimensions = {}
    for dim in dims:
        g = group_metrics(scored, dim, flagged)
        g["gated"] = (
            (g["rows"] >= t.min_group_rows)
            & (g["positives"] >= t.min_group_positives)
            & (g["group"] != UNKNOWN)
        )
        gated = g[g["gated"]]
        problems = []
        if len(gated) < 2:
            problems.append(f"only {len(gated)} group(s) large enough to compare")
            recall_ratio = min_lift = None
            worst = None
        else:
            recall_ratio = float(gated["recall"].min() / gated["recall"].max())
            min_lift = float(gated["lift"].min())
            worst = str(gated.loc[gated["recall"].idxmin(), "group"])
            if recall_ratio < t.min_recall_ratio:
                problems.append(
                    f"recall ratio {recall_ratio:.2f} < {t.min_recall_ratio} (lowest: {worst})"
                )
            low_lift = gated.loc[gated["lift"] < t.min_group_lift, "group"].tolist()
            if low_lift:
                problems.append(f"lift < {t.min_group_lift} in {low_lift}")
        dimensions[dim] = {
            "passed": not problems,
            "problems": problems,
            "recall_ratio": {"value": _clean(recall_ratio)},
            "min_group_lift": {"value": _clean(min_lift)},
            "lowest_recall_group": worst,
            "groups": [
                {k: _clean(v) for k, v in row.items()} for row in g.to_dict(orient="records")
            ],
        }

    def _worst(key: str):
        vals = [d[key]["value"] for d in dimensions.values()]
        return None if any(v is None for v in vals) else min(vals)

    return {
        "passed": all(d["passed"] for d in dimensions.values()),
        "thresholds": asdict(t),
        "cutoff": round(cutoff, 6),
        "rows": len(scored),
        # Worst case across dimensions: the two paths the Stage 4 ConditionStep reads.
        "recall_ratio": {"value": _worst("recall_ratio")},
        "min_group_lift": {"value": _worst("min_group_lift")},
        "dimensions": dimensions,
    }


def load_communities(path: str | Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(Path(path).read_text()))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", required=True, help="dir with predictions.csv")
    p.add_argument("--communities", required=True, help="communities.json from ingest")
    p.add_argument("--output", required=True, help=f"dir to write {REPORT_NAME} into")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, *, thresholds: FairnessThresholds | None = None) -> dict:
    """Write the report. It doesn't raise: the Stage 4 ConditionStep reads the report and
    decides, the same way it decides on PR-AUC."""
    args = _parse_args(argv)
    preds = pd.read_csv(Path(args.predictions) / "predictions.csv", dtype={"comm_code": str})
    report = fairness_report(preds, load_communities(args.communities), thresholds)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / REPORT_NAME).write_text(json.dumps(report, indent=2, allow_nan=False))
    for dim, d in report["dimensions"].items():
        log.info("%-6s %-4s %s", dim, "PASS" if d["passed"] else "FAIL", "; ".join(d["problems"]))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
