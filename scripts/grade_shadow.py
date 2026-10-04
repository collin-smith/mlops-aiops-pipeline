#!/usr/bin/env python3
"""Grade the shadow scores on real outcomes, in Athena (Stage 7, D-045).

    uv run python scripts/grade_shadow.py            # after the second snapshot is crawled
    uv run python scripts/grade_shadow.py --show     # print the queries; run nothing

Run it once ``processed/311`` holds a snapshot newer than the scored one (D-036 keeps one
snapshot there, so the ``shadow_outcomes`` view grades against it with no changes). It
refreshes the view (DDL is free), pulls every score with its outcome (a few MB scanned),
grades them with ``src.deploy.grade``, prints the report and writes it to
``s3://<bucket>/monitoring/shadow-grade/asof=<outcome snapshot>/grade.json``.

The report is a report, not a gate. The model stays in shadow whatever it says.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.config import get_config  # noqa: E402
from src.deploy import grade, views  # noqa: E402

BOOLEAN = ("flagged", "category_seen")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--show", action="store_true", help="print the queries; run nothing")
    ap.add_argument("--out", type=Path, help="also write grade.json here")
    args = ap.parse_args(argv)
    if args.show:
        for sql in (*views.VIEWS.values(), views.GRADE_ROWS, views.SHADOW_GRADE):
            print(sql, end="\n\n")
        return 0

    from src.common.athena import run_query

    cfg = get_config()
    views.create_views(run_query)
    rows = run_query(views.GRADE_ROWS)
    print(f"{len(rows):,} scores read ({rows.attrs.get('scanned_mb')} MB scanned)")
    for c in BOOLEAN:  # Athena returns every value as text
        rows[c] = rows[c].str.lower().eq("true")
    report = grade.grade(rows)
    print("\n".join(grade.summary_lines(report)))
    if report["rows_graded"] == 0:
        print("nothing to grade yet: no scored deadline has passed in the latest snapshot")
        return 0

    body = json.dumps(report, indent=2, default=str)
    key = f"monitoring/shadow-grade/asof={report['outcome_snapshot']}/grade.json"
    boto3.client("s3", region_name=cfg.region).put_object(
        Bucket=cfg.bucket, Key=key, Body=body.encode()
    )
    print(f"wrote s3://{cfg.bucket}/{key}")
    if args.out:
        args.out.write_text(body)
    print("the article's table, in the Athena console:")
    print(views.SHADOW_GRADE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
