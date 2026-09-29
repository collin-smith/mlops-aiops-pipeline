#!/usr/bin/env python3
"""Smoke test: can this account run a SageMaker Processing job, and how fast is a t3?

AWS denied the Training Job quota (D-039), so training and batch scoring run as
Processing jobs on the ml.t3 types, which do have a default quota. This runs one short
job before any pipeline code relies on that. It checks that the account plan lets the job
start, that the project role can pull the built-in XGBoost image, that the job is
tagged, and how long a small XGBoost fit takes on the instance.

    ROLE=$(terraform -chdir=infra output -raw sagemaker_role_arn)
    uv run python scripts/smoke_processing.py --role-arn "$ROLE"
    uv run python scripts/smoke_processing.py --role-arn "$ROLE" \
        --instance-type ml.t3.xlarge

It uploads scripts/smoke_probe.py to s3://$MLOPS_BUCKET/code/smoke/ and mounts it as the
job's only input. There's no output; the script echoes the job's log when it finishes. The
job stops by itself after --max-runtime seconds, and Ctrl+C stops it too. Cost: about a
minute of an ml.t3 instance, well under $0.01.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import aws_tags, get_config  # noqa: E402
from src.common.guards import budget_hardstop_active  # noqa: E402
from src.common.sm_jobs import PROCESSING_INSTANCE_TYPES as INSTANCE_TYPES  # noqa: E402
from src.common.sm_jobs import job_name as _job_name  # noqa: E402
from src.common.sm_jobs import print_job_log, summarize, wait_for_job, xgboost_image  # noqa: E402

# The probe runs inside the container. It's uploaded to S3 and mounted as a job input, the
# same way the training script will be (an entrypoint argument is capped at 256 characters).
PROBE_FILE = Path(__file__).with_name("smoke_probe.py")
CODE_DIR = "/opt/ml/processing/input/code"


def job_name(project: str, now: datetime | None = None) -> str:
    return _job_name(project, "smoke", now)


def processing_job_request(
    name: str,
    role_arn: str,
    image_uri: str,
    instance_type: str,
    max_runtime: int,
    code_s3_prefix: str,
) -> dict:
    """The CreateProcessingJob request: one small instance, a hard time limit, tagged."""
    if instance_type not in INSTANCE_TYPES:
        raise ValueError(
            f"{instance_type} has no Processing quota here; use one of {INSTANCE_TYPES}"
        )
    if not 60 <= max_runtime <= 1800:
        raise ValueError(f"max_runtime must be 60–1800 s for a smoke test, got {max_runtime}")
    return {
        "ProcessingJobName": name,
        "RoleArn": role_arn,
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": ["python3", f"{CODE_DIR}/{PROBE_FILE.name}"],
        },
        "ProcessingInputs": [
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": code_s3_prefix,
                    "LocalPath": CODE_DIR,
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 1,
            }
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime},
        "Tags": aws_tags(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--role-arn", required=True, help="the project SageMaker role (terraform output)"
    )
    ap.add_argument("--instance-type", default="ml.t3.medium", choices=INSTANCE_TYPES)
    ap.add_argument("--max-runtime", type=int, default=600, help="seconds (default 600)")
    args = ap.parse_args(argv)

    cfg = get_config()
    if budget_hardstop_active(boto3.client("iam"), args.role_arn, cfg.project):
        raise SystemExit("budget hard-stop is active (deny policy attached); not starting a job")

    image = xgboost_image(cfg.region, "1.7-1")
    name = job_name(cfg.project)
    code_prefix = f"s3://{cfg.bucket}/code/smoke/"
    boto3.client("s3", region_name=cfg.region).upload_file(
        str(PROBE_FILE), cfg.bucket, f"code/smoke/{PROBE_FILE.name}"
    )
    request = processing_job_request(
        name, args.role_arn, image, args.instance_type, args.max_runtime, code_prefix
    )

    sm = boto3.client("sagemaker", region_name=cfg.region)
    sm.create_processing_job(**request)
    print(f"started {name} on {args.instance_type} ({image})")
    desc = wait_for_job(sm, name)
    if desc is None:
        return 1
    summary = summarize(desc)
    print(summary)
    print_job_log(boto3.client("logs", region_name=cfg.region), name)
    return 0 if summary["status"] == "Completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
