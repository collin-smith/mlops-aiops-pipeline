#!/usr/bin/env python3
"""Choropleth of Calgary's communities, shaded by one 311 metric. Article figure, $0.

Boundaries come from the City's Community District Boundaries open dataset (`surr-xmvs`),
joined to the metric on `comm_code` — the same key the 311 data carries. Output is an
article asset, so it is written under ../output/stages/articles/assets/, never into code/.

Three metric sources:

    # Real: breach rate by community, exported from Athena after Stage 1 (query below)
    python scripts/community_map.py --metrics breach_by_community.csv

    # Mix-adjusted (the article figure): each community's late rate over what its own request
    # mix predicts, from the companion analysis (community_oe.py). Diverging, centred on 1.0.
    python scripts/community_map.py --ratio community_oe.csv

    # Preview, no AWS: average days to close a pothole request, straight from the City API.
    # One category only, so the map isn't confounded by each community's request mix.
    python scripts/community_map.py --preview-potholes

Athena query for --metrics (save the result as CSV; columns comm_code, value, n). Breach is
category-relative by construction: each request is compared with its own category's p75.

    WITH thresholds AS (
      SELECT service_name,
             approx_percentile(date_diff('day', requested_date, closed_date), 0.75) AS p75
      FROM   mlops_aiops.processed_311
      WHERE  closed_date IS NOT NULL
      GROUP  BY service_name
    )
    SELECT r.comm_code,
           avg(CASE WHEN date_diff('day', r.requested_date, r.closed_date) > t.p75
                    THEN 1.0 ELSE 0.0 END) AS value,
           count(*) AS n
    FROM   mlops_aiops.processed_311 r
    JOIN   thresholds t USING (service_name)
    WHERE  r.closed_date IS NOT NULL
    GROUP  BY r.comm_code;

Credit line on the figure: "Contains information licensed under the Open Government
Licence – City of Calgary." Needs matplotlib (`uv sync --extra ml`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import requests

BOUNDARIES_URL = "https://data.calgary.ca/resource/surr-xmvs.geojson"
SERVICE_URL = "https://data.calgary.ca/resource/iahh-g8bj.json"
POTHOLE_CATEGORY = "Roads - Pothole Maintenance"
WINDOW = "requested_date >= '2021-01-01'"
HEADERS = {"User-Agent": "mlops-aiops-pipeline/0.1 community-map"}
CREDIT = "Contains information licensed under the Open Government Licence – City of Calgary."

# Sequential blue, light -> dark (validated reference ramp, steps 150/300/400/500/650).
BINS_HEX = ("#b7d3f6", "#6da7ec", "#3987e5", "#256abf", "#104281")
NO_DATA = "#e4e3df"
# Diverging blue <-> red with a neutral grey midpoint (reference palette's diverging pair):
# blue 400 / 250, neutral, a red tint matched to blue 250's lightness, red. Faster than the
# mix predicts is blue, slower is red. Fixed breaks, so the bands mean the same on every map.
RATIO_HEX = ("#3987e5", "#86b6ef", "#f0efec", "#ef9797", "#e34948")
RATIO_BREAKS = (0.85, 0.95, 1.05, 1.15)
RATIO_LABELS = (
    "15%+ faster than its mix predicts",
    "5–15% faster",
    "as expected (within 5%)",
    "5–15% slower",
    "15%+ slower",
)
INK, INK_MUTED, SURFACE = "#1f1f1e", "#6b6a66", "#ffffff"

MLOPS_ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSETS = MLOPS_ROOT / "output" / "stages" / "articles" / "assets"
CACHE = pathlib.Path(__file__).resolve().parents[1] / "data" / "community_boundaries.geojson"


# --- pure helpers (tested without matplotlib or the network) ---


def quantile_breaks(values: list[float], k: int = len(BINS_HEX)) -> list[float]:
    """k-1 interior cut points so each class holds ~the same number of communities."""
    v = sorted(values)
    if len(v) < k:
        raise ValueError(f"need at least {k} communities with data, got {len(v)}")
    return [v[math.ceil(len(v) * i / k) - 1] for i in range(1, k)]


def classify(value: float, breaks: list[float]) -> int:
    """Index of the class a value falls in: 0 = lowest (lightest)."""
    return sum(value > b for b in breaks)


def ratio_class(value: float) -> int:
    """Diverging band for an observed/expected ratio: 0 = much faster ... 4 = much slower."""
    return sum(value > b for b in RATIO_BREAKS)


def filter_min_n(metric: dict[str, tuple[float, int]], min_n: int) -> dict[str, float]:
    """Drop communities with too few requests to shade honestly; they render as no-data."""
    return {code: v for code, (v, n) in metric.items() if n >= min_n}


def exterior_rings(geometry: dict) -> list[list[tuple[float, float]]]:
    """Outer rings of a (Multi)Polygon. Community polygons have no holes worth drawing."""
    if geometry["type"] == "Polygon":
        polys = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polys = geometry["coordinates"]
    else:
        return []
    return [[(x, y) for x, y in poly[0]] for poly in polys]


# --- I/O ---


def load_boundaries(refresh: bool = False) -> dict:
    if refresh or not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(BOUNDARIES_URL, params={"$limit": 5000}, headers=HEADERS, timeout=120)
        r.raise_for_status()
        CACHE.write_text(r.text)
    return json.loads(CACHE.read_text())


def load_metrics_csv(path: pathlib.Path) -> dict[str, tuple[float, int]]:
    with path.open(newline="") as f:
        return {
            row["comm_code"]: (float(row["value"]), int(float(row["n"])))
            for row in csv.DictReader(f)
            if row.get("comm_code") and row.get("value") not in (None, "")
        }


def fetch_pothole_days() -> dict[str, tuple[float, int]]:
    params = {
        "$select": "comm_code,count(*) as n,avg(date_diff_d(closed_date,requested_date)) as d",
        "$where": f"{WINDOW} AND closed_date IS NOT NULL AND service_name='{POTHOLE_CATEGORY}'",
        "$group": "comm_code",
        "$limit": 5000,
    }
    r = requests.get(SERVICE_URL, params=params, headers=HEADERS, timeout=180)
    r.raise_for_status()
    return {
        row["comm_code"]: (float(row["d"]), int(row["n"]))
        for row in r.json()
        if row.get("comm_code") and "d" in row
    }


# --- render ---


def render(
    boundaries: dict,
    values: dict[str, float],
    *,
    title: str,
    subtitle: str,
    legend_fmt: str,
    out_stem: pathlib.Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    breaks = quantile_breaks(list(values.values()))
    polys, colors = [], []
    for feat in boundaries["features"]:
        code = feat["properties"].get("comm_code")
        fill = BINS_HEX[classify(values[code], breaks)] if code in values else NO_DATA
        for ring in exterior_rings(feat["geometry"]):
            polys.append(ring)
            colors.append(fill)

    fig, ax = plt.subplots(figsize=(8, 9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    # White edges are the 2px-style surface gap between adjacent fills.
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=SURFACE, linewidths=0.6))
    ax.autoscale_view()
    ax.set_aspect(1 / math.cos(math.radians(51.05)))  # lon/lat -> roughly true shape
    ax.axis("off")

    lo, hi = min(values.values()), max(values.values())
    edges = [lo, *breaks, hi]
    handles = [
        Patch(
            facecolor=BINS_HEX[i],
            edgecolor="none",
            label=f"{legend_fmt.format(edges[i])} – {legend_fmt.format(edges[i + 1])}",
        )
        for i in range(len(BINS_HEX))
    ]
    handles.append(Patch(facecolor=NO_DATA, edgecolor="none", label="no data / too few requests"))
    leg = ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handleheight=1.2,
        labelcolor=INK,
    )
    leg.set_title("Quintiles of communities", prop={"size": 8, "weight": "bold"})
    leg.get_title().set_color(INK)

    fig.text(0.06, 0.965, title, fontsize=13, fontweight="bold", color=INK, ha="left")
    fig.text(0.06, 0.94, subtitle, fontsize=9, color=INK_MUTED, ha="left")
    fig.text(0.06, 0.025, CREDIT, fontsize=7, color=INK_MUTED, ha="left")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), facecolor=SURFACE)
    fig.savefig(out_stem.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_stem}.png + .svg ({len(values)} communities shaded)")


def render_ratio(boundaries: dict, values: dict[str, float], *, min_n: int, out_stem) -> None:
    """The mix-adjusted map: diverging bands around 1.0; too-few-requests is hatched, so it
    can't be mistaken for the neutral "as expected" grey."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    shaded, empty, colors = [], [], []
    for feat in boundaries["features"]:
        code = feat["properties"].get("comm_code")
        for ring in exterior_rings(feat["geometry"]):
            if code in values:
                shaded.append(ring)
                colors.append(RATIO_HEX[ratio_class(values[code])])
            else:
                empty.append(ring)

    fig, ax = plt.subplots(figsize=(8, 9.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.add_collection(PolyCollection(empty, facecolors=SURFACE, edgecolors=NO_DATA,
                                     linewidths=0.5, hatch="////"))  # fmt: skip
    ax.add_collection(PolyCollection(shaded, facecolors=colors, edgecolors=SURFACE, linewidths=0.6))
    ax.autoscale_view()
    ax.set_aspect(1 / math.cos(math.radians(51.05)))
    ax.axis("off")

    counts = [sum(ratio_class(v) == i for v in values.values()) for i in range(len(RATIO_HEX))]
    handles = [
        Patch(facecolor=RATIO_HEX[i], edgecolor="none", label=f"{RATIO_LABELS[i]} ({counts[i]})")
        for i in reversed(range(len(RATIO_HEX)))
    ]
    handles.append(Patch(facecolor=SURFACE, edgecolor=NO_DATA, hatch="////",
                         label=f"fewer than {min_n:,} requests: too few to say"))  # fmt: skip
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8,
                    handlelength=1.4, handleheight=1.2, labelcolor=INK)  # fmt: skip
    leg.set_title(
        "Communities, late rate vs. their request mix", prop={"size": 8, "weight": "bold"}
    )
    leg.get_title().set_color(INK)

    fig.text(0.06, 0.965, "Calgary 311: which communities wait longer for the same requests",
             fontsize=12.5, fontweight="bold", color=INK, ha="left")  # fmt: skip
    fig.text(0.06, 0.94, "Late rate against what each community's own mix of requests predicts. "
             "Last 2 years, to Sept 2026.", fontsize=8.8, color=INK_MUTED, ha="left")  # fmt: skip
    note = (
        "Late = slower than 3 of 4 past requests of the same type. "
        "Shows where, not why; not reviewed by the City."
    )
    fig.text(0.06, 0.04, note, fontsize=7.2, color=INK_MUTED, ha="left")
    fig.text(0.06, 0.022, CREDIT, fontsize=7, color=INK_MUTED, ha="left")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.06)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_stem}.png ({len(values)} communities shaded; bands {counts})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--metrics", type=pathlib.Path, help="CSV: comm_code,value,n (Athena)")
    src.add_argument("--ratio", type=pathlib.Path, help="CSV: comm_code,value,n (obs/expected)")
    src.add_argument("--preview-potholes", action="store_true", help="City API, no AWS")
    ap.add_argument("--min-n", type=int, default=None, help="min requests to shade a community")
    ap.add_argument("--refresh-boundaries", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, help="output stem (no extension)")
    args = ap.parse_args(argv)

    boundaries = load_boundaries(args.refresh_boundaries)
    if args.ratio:
        min_n = args.min_n or 1000
        values = filter_min_n(load_metrics_csv(args.ratio), min_n)
        stem = args.out or ASSETS / "community-map-mix-adjusted"
        render_ratio(boundaries, values, min_n=min_n, out_stem=stem)
        return 0
    if args.preview_potholes:
        metric, min_n = fetch_pothole_days(), args.min_n or 50
        title = "How long a pothole request takes to close, by community"
        subtitle = f"Average days, closed requests since 2021 · communities with ≥{min_n} requests"
        legend_fmt, stem = "{:.0f} d", "community-map-pothole-days"
    else:
        metric, min_n = load_metrics_csv(args.metrics), args.min_n or 200
        title = "Share of requests that ran past their category's norm, by community"
        subtitle = (
            "Breach = closed later than the category's 75th-percentile time · "
            f"communities with ≥{min_n} requests"
        )
        legend_fmt, stem = "{:.0%}", "community-map-breach-rate"

    values = filter_min_n(metric, min_n)
    render(
        boundaries,
        values,
        title=title,
        subtitle=subtitle,
        legend_fmt=legend_fmt,
        out_stem=args.out or ASSETS / stem,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
