#!/usr/bin/env python3
"""Check the project's job history against the 2× rule (Stage 7 Layer B, D-045).

    uv run python scripts/pipeline_health.py
    uv run python scripts/pipeline_health.py --snapshot 2026-10-20=2950000 \
        --alert --topic-arn "$(terraform -chdir=infra output -raw alerts_topic_arn)"

Run it by hand after a retrain or a scoring run; nothing schedules it. It reads SageMaker's
job history (free), prints one line per job, and with ``--alert`` emails the existing
``mlops-aiops-alerts`` topic if anything is flagged. It starts no jobs and adds no metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.config import get_config  # noqa: E402
from src.monitor import pipeline_health  # noqa: E402


def parse_snapshot(text: str) -> tuple[str, int]:
    asof, _, rows = text.partition("=")
    if not rows:
        raise argparse.ArgumentTypeError("use <asof>=<rows>, e.g. 2026-10-20=2950000")
    return asof, int(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--snapshot", type=parse_snapshot, action="append", default=[],
        help="another snapshot's row count, <asof>=<rows> (repeatable)",
    )  # fmt: skip
    ap.add_argument("--alert", action="store_true", help="publish flags to SNS")
    ap.add_argument("--topic-arn", help="the alerts topic (terraform output alerts_topic_arn)")
    ap.add_argument("--csv", type=Path, help="also write the assessed table here")
    args = ap.parse_args(argv)
    if args.alert and not args.topic_arn:
        ap.error("--alert needs --topic-arn")

    cfg = get_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    jobs = pipeline_health.job_records(sm, cfg.project)
    if jobs.empty:
        print("no project Processing jobs found")
        return 0
    snapshots = {**pipeline_health.SNAPSHOT_ROWS, **dict(args.snapshot)}
    assessed = pipeline_health.assess(jobs, snapshots)
    cols = ["created", "kind", "status", "billed_seconds", "usd", "sec_per_mrows", "reason"]
    print(assessed[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"{len(assessed)} jobs, ≈ ${assessed['usd'].sum():.3f} estimated in all")
    if args.csv:
        assessed.to_csv(args.csv, index=False)

    text = pipeline_health.alert_text(assessed)
    if text is None:
        print("nothing flagged")
        return 0
    print(text)
    if args.alert:
        boto3.client("sns", region_name=cfg.region).publish(
            TopicArn=args.topic_arn, Subject="mlops-aiops: pipeline-health flag", Message=text
        )
        print("alert sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
