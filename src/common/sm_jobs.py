"""Helpers for SageMaker Processing jobs launched from a laptop (D-039).

Training and batch scoring run as Processing jobs because the account has no Training
or Transform quota. These are the pieces the hand-run launchers share: the image
address, waiting for the job, and echoing its log.
"""

from __future__ import annotations

import contextlib
import time
from datetime import UTC, datetime
from pathlib import Path

# The only instance types with a non-zero Processing quota in ca-central-1 (2026-09-28).
PROCESSING_INSTANCE_TYPES = ("ml.t3.medium", "ml.t3.large", "ml.t3.xlarge")

# AWS's built-in XGBoost images. The registry account differs per region; these come from
# the SageMaker SDK's image_uri_config/xgboost.json, so the launchers need no SDK.
_XGBOOST_REGISTRY = {"ca-central-1": "341280168497"}
XGBOOST_VERSIONS = ("1.7-1", "3.2-0")

TERMINAL = ("Completed", "Failed", "Stopped")
LOG_GROUP = "/aws/sagemaker/ProcessingJobs"


def xgboost_image(region: str, version: str) -> str:
    if region not in _XGBOOST_REGISTRY:
        raise SystemExit(f"no XGBoost image registry listed for {region}; add it in sm_jobs.py")
    if version not in XGBOOST_VERSIONS:
        raise ValueError(f"XGBoost image {version} not listed; use one of {XGBOOST_VERSIONS}")
    account = _XGBOOST_REGISTRY[region]
    return f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:{version}"


def job_name(project: str, kind: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{project}-{kind}-{stamp}"


def upload_files(s3, bucket: str, prefix: str, files: dict[str, Path]) -> str:
    """Upload ``{key under prefix: local path}`` and return the ``s3://`` prefix."""
    for key, path in files.items():
        s3.upload_file(str(path), bucket, f"{prefix}/{key}")
    return f"s3://{bucket}/{prefix}/"


def summarize(desc: dict) -> dict:
    """Status plus the run time that is actually billed (start to end of processing)."""
    out = {"status": desc["ProcessingJobStatus"]}
    if desc.get("FailureReason"):
        out["failure_reason"] = desc["FailureReason"]
    start, end = desc.get("ProcessingStartTime"), desc.get("ProcessingEndTime")
    if start and end:
        out["billed_seconds"] = round((end - start).total_seconds())
    return out


def wait_for_job(sm, name: str, poll_seconds: int = 15) -> dict | None:
    """Poll until the job ends. Ctrl+C stops the job (so it stops billing) and returns None."""
    try:
        while True:
            desc = sm.describe_processing_job(ProcessingJobName=name)
            if desc["ProcessingJobStatus"] in TERMINAL:
                return desc
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        sm.stop_processing_job(ProcessingJobName=name)
        print(f"stopped {name}")
        return None


def print_job_log(logs, name: str, tail: int = 60) -> None:
    """Echo the job's own output so nobody has to open CloudWatch."""
    events = []
    for _ in range(6):  # the log usually lands in CloudWatch a little after the job ends
        # FilterLogEvents can return an empty page plus a nextToken, so read every page
        with contextlib.suppress(logs.exceptions.ResourceNotFoundException):
            pages = logs.get_paginator("filter_log_events").paginate(
                logGroupName=LOG_GROUP, logStreamNamePrefix=name
            )
            events = [e for page in pages for e in page["events"]]
        if events:
            break
        time.sleep(10)
    for e in events[-tail:]:
        print("  |", e["message"].rstrip())
    if not events:
        print(f"no log lines yet; see CloudWatch {LOG_GROUP}, stream prefix {name}")
