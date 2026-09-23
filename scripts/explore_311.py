#!/usr/bin/env python3
"""Pre-build exploratory pass over Calgary 311 — runs against the live Socrata API.

No AWS, no cost. Reproduces the aggregate numbers in `docs/exploratory-findings.md`.
Numbers drift daily as the City updates the dataset.

    python scripts/explore_311.py            # aggregate queries only
    python scripts/explore_311.py --model    # also pull a sample + a quick separability check

An app token (env SOCRATA_APP_TOKEN) is optional but avoids throttling on the sample pull.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests

BASE = "https://data.calgary.ca/resource/iahh-g8bj.json"
META = "https://data.calgary.ca/api/views/iahh-g8bj.json"
WINDOW = "requested_date >= '2021-01-01'"
HEADERS = {"User-Agent": "mlops-aiops-pipeline/0.1 explore"}
if os.environ.get("SOCRATA_APP_TOKEN"):
    HEADERS["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]


def q(params: dict, *, retries: int = 4) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(BASE, params=params, headers=HEADERS, timeout=90)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  ! {type(e).__name__}; retry {attempt + 1}/{retries}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise SystemExit("gave up — try again or set SOCRATA_APP_TOKEN")


def rule(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def aggregates() -> None:
    rule("dataset")
    n_all = q({"$select": "count(*) as n"})[0]["n"]
    n_5y = q({"$select": "count(*) as n", "$where": WINDOW})[0]["n"]
    print(f"rows all-time: {int(n_all):,}   last 5y: {int(n_5y):,}")

    rule("status breakdown (last 5y)")
    for r in q(
        {
            "$select": "status_description,count(*) as n",
            "$where": WINDOW,
            "$group": "status_description",
            "$order": "n desc",
        }
    ):
        print(f"  {r['status_description']:20s} {int(r['n']):>10,}")

    rule("open requests (null closed_date, last 5y) — Stage 6 scoring targets")
    open_n = q({"$select": "count(*) as n", "$where": WINDOW + " AND closed_date IS NULL"})[0]["n"]
    print(f"  {int(open_n):,}")

    rule("top 15 categories by volume (last 5y)")
    for r in q(
        {
            "$select": "service_name,count(*) as n",
            "$where": WINDOW,
            "$group": "service_name",
            "$order": "n desc",
            "$limit": 15,
        }
    ):
        print(f"  {int(r['n']):>8,}  {r['service_name']}")

    rule("resolution time by category — slowest 15 (avg days, n>5000, last 5y)")
    for r in q(
        {
            "$select": "service_name,count(*) as n,"
            "avg(date_diff_d(closed_date,requested_date)) as d",
            "$where": WINDOW + " AND closed_date IS NOT NULL",
            "$group": "service_name",
            "$having": "n > 5000",
            "$order": "d desc",
            "$limit": 15,
        }
    ):
        print(f"  {float(r['d']):>7.1f}d  (n={int(r['n']):>7,})  {r['service_name']}")

    rule("seasonality — monthly totals, all categories (last 5y)")
    rows = q(
        {
            "$select": "date_extract_m(requested_date) as m,count(*) as n",
            "$where": WINDOW,
            "$group": "m",
            "$order": "m",
        }
    )
    print("  " + "  ".join(f"{int(r['m']):02d}:{int(r['n']):>7,}" for r in rows))

    for cat in (
        "Bylaw - Snow and Ice on Sidewalk",
        "Bylaw - Long Grass - Weeds Infraction",
        "Roads - Pothole Maintenance",
    ):
        rows = q(
            {
                "$select": "date_extract_m(requested_date) as m,count(*) as n",
                "$where": WINDOW + f" AND service_name='{cat}'",
                "$group": "m",
                "$order": "m",
            }
        )
        print(f"  {cat}:\n    " + " ".join(f"{int(r['m'])}:{int(r['n'])}" for r in rows))

    rule("pothole resolution time by community — slowest & fastest (n>200, last 5y)")
    common = {
        "$select": "comm_name,count(*) as n,avg(date_diff_d(closed_date,requested_date)) as d",
        "$where": WINDOW
        + " AND closed_date IS NOT NULL AND service_name='Roads - Pothole Maintenance'",
        "$group": "comm_name",
        "$having": "n > 200",
    }
    for label, order in (("slowest", "d desc"), ("fastest", "d asc")):
        print(f"  {label}:")
        for r in q({**common, "$order": order, "$limit": 6}):
            print(f"    {float(r['d']):>6.1f}d  (n={int(r['n']):>4})  {r['comm_name']}")


def model_check() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    sel = ":id,requested_date,closed_date,service_name,agency_responsible,comm_name,source"

    def pull(where: str, pages: int = 3, limit: int = 10000) -> pd.DataFrame:
        out: list = []
        last = None
        for _ in range(pages):
            w = where + (f" AND :id > '{last}'" if last else "")
            page = q({"$select": sel, "$where": w, "$order": ":id", "$limit": limit})
            if not page:
                break
            last = page[-1][":id"]
            for row in page:
                row.pop(":id", None)
            out.append(pd.DataFrame(page))
            if len(page) < limit:
                break
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    rule("separability check (temporal split)")
    print("pulling ~230k closed requests (2024 sample -> train, 2025 -> test) ...")
    parts = []
    for wnd in (
        "requested_date >= '2024-01-01' AND requested_date < '2024-04-01'",
        "requested_date >= '2024-04-01' AND requested_date < '2024-07-01'",
        "requested_date >= '2024-07-01' AND requested_date < '2024-10-01'",
        "requested_date >= '2024-10-01' AND requested_date < '2025-01-01'",
        "requested_date >= '2025-01-01' AND requested_date < '2025-08-01'",
    ):
        parts.append(pull("closed_date IS NOT NULL AND " + wnd, pages=3))
    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    for c in ("requested_date", "closed_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["dtc"] = (df.closed_date - df.requested_date).dt.total_seconds() / 86400
    df = df[(df.dtc >= 0) & df.dtc.notna()].copy()
    df["mon"] = df.requested_date.dt.month.astype(int)
    df["dow"] = df.requested_date.dt.dayofweek.astype(int)
    df["yr"] = df.requested_date.dt.year

    thr = df[df.yr < 2025].groupby("service_name")["dtc"].quantile(0.75).rename("thr")
    df = df.join(thr, on="service_name")
    df = df[df.thr.notna()].copy()
    df["breach"] = (df.dtc > df.thr).astype(int)
    df = df.sort_values("requested_date")
    df["cat_30d"] = df.groupby("service_name", group_keys=False).apply(
        lambda g: g.set_index("requested_date")["breach"].rolling("30D").count().values - 1
    )
    df["cat_30d"] = pd.to_numeric(df["cat_30d"], errors="coerce").fillna(0)
    for c in ["service_name", "agency_responsible", "comm_name", "source"]:
        df[c] = df[c].astype("string").fillna("NA").astype(str)

    tr, te = df[df.yr < 2025], df[df.yr == 2025]
    cat = ["service_name", "agency_responsible", "comm_name", "source"]
    num = ["mon", "dow", "cat_30d"]
    pre = ColumnTransformer(
        [
            (
                "c",
                OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False),
                cat,
            ),
            ("n", "passthrough", num),
        ]
    )
    base = te.breach.mean()
    print(
        f"train {len(tr):,} (breach {tr.breach.mean():.1%})  test {len(te):,} (breach {base:.1%})"
    )
    for name, clf in [
        ("logreg", LogisticRegression(max_iter=400)),
        ("histgbm", HistGradientBoostingClassifier(max_depth=6, max_iter=300, learning_rate=0.08)),
    ]:
        m = Pipeline([("p", pre), ("m", clf)]).fit(tr[cat + num], tr.breach)
        p = m.predict_proba(te[cat + num])[:, 1]
        roc = roc_auc_score(te.breach, p)
        pr = average_precision_score(te.breach, p)
        line = f"  {name:8s} ROC {roc:.3f}  PR {pr:.3f}"
        if name == "histgbm":
            top = te.breach[p >= np.quantile(p, 0.9)].mean()
            line += f"   top-decile breach {top:.1%} ({top / base:.1f}x lift)"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", action="store_true", help="also run the separability check (needs scikit-learn)"
    )
    args = ap.parse_args()

    meta = requests.get(META, headers=HEADERS, timeout=60).json()
    print(f"dataset rows updated: {meta.get('rowsUpdatedAt')}  (unix ts)")
    aggregates()
    if args.model:
        model_check()
