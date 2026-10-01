#!/usr/bin/env python3
"""Train the Stage 2 baseline on SageMaker, as a Processing job (D-039).

Uploads src/ to s3://$MLOPS_BUCKET/code/<job>/, mounts it and the processed Parquet
snapshot as job inputs, runs ``python -m src.pipeline.train`` inside AWS's built-in
XGBoost image, and writes the model to s3://$MLOPS_BUCKET/model-artifacts/<job>/.

    ROLE=$(terraform -chdir=infra output -raw sagemaker_role_arn)
    uv run python scripts/train_job.py --role-arn "$ROLE"
    uv run python scripts/train_job.py --role-arn "$ROLE" --max-train-rows 1000000

The job stops by itself after --max-runtime seconds, and Ctrl+C stops it too. When it
finishes, the script prints the job's log and metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.config import aws_tags, get_config  # noqa: E402
from src.common.guards import budget_hardstop_active  # noqa: E402
from src.common.sm_jobs import (  # noqa: E402
    PROCESSING_INSTANCE_TYPES,
    REQUIREMENTS,
    code_files,
    job_name,
    print_job_log,
    summarize,
    upload_files,
    wait_for_job,
    xgboost_image,
)

IN_CODE = "/opt/ml/processing/input/code"
IN_DATA = "/opt/ml/processing/input/data"
OUT = "/opt/ml/processing/output"


def entrypoint(max_train_rows: int) -> list[str]:
    cmd = (
        f"cd {IN_CODE} && pip install -q -r {REQUIREMENTS} && "
        # the Stage 2 model (D-037); the pipeline trains the Stage 5 challenger
        f"python3 -m src.pipeline.train --feature-set baseline --data {IN_DATA} --out {OUT}"
    )
    if max_train_rows:
        cmd += f" --max-train-rows {max_train_rows}"
    return ["bash", "-c", cmd]


def _s3_input(name: str, uri: str, local: str) -> dict:
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": uri,
            "LocalPath": local,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
        },
    }


def processing_job_request(
    name: str,
    role_arn: str,
    image_uri: str,
    instance_type: str,
    max_runtime: int,
    code_uri: str,
    data_uri: str,
    output_uri: str,
    max_train_rows: int = 0,
) -> dict:
    """The CreateProcessingJob request: one small instance, a hard time limit, tagged."""
    if instance_type not in PROCESSING_INSTANCE_TYPES:
        raise ValueError(
            f"{instance_type} has no Processing quota here; use one of {PROCESSING_INSTANCE_TYPES}"
        )
    if not 300 <= max_runtime <= 7200:
        raise ValueError(f"max_runtime must be 300–7200 s, got {max_runtime}")
    return {
        "ProcessingJobName": name,
        "RoleArn": role_arn,
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": entrypoint(max_train_rows),
        },
        "ProcessingInputs": [
            _s3_input("code", code_uri, IN_CODE),
            _s3_input("data", data_uri, IN_DATA),
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "model",
                    "S3Output": {
                        "S3Uri": output_uri,
                        "LocalPath": OUT,
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 5,
            }
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime},
        "Environment": {"PYTHONUNBUFFERED": "1"},
        "Tags": aws_tags(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--role-arn", required=True, help="the project SageMaker role (terraform output)"
    )
    ap.add_argument("--instance-type", default="ml.t3.xlarge", choices=PROCESSING_INSTANCE_TYPES)
    ap.add_argument("--max-runtime", type=int, default=3600, help="seconds (default 3600)")
    ap.add_argument("--max-train-rows", type=int, default=0, help="sample the training rows")
    ap.add_argument("--image-version", default="3.2-0", help="built-in XGBoost image tag")
    args = ap.parse_args(argv)

    cfg = get_config()
    if budget_hardstop_active(boto3.client("iam"), args.role_arn, cfg.project):
        raise SystemExit("budget hard-stop is active (deny policy attached); not starting a job")

    name = job_name(cfg.project, "train")
    s3 = boto3.client("s3", region_name=cfg.region)
    code_uri = upload_files(s3, cfg.bucket, f"code/{name}", code_files())
    output_uri = f"{cfg.model_artifacts_prefix}/{name}/"
    request = processing_job_request(
        name,
        args.role_arn,
        xgboost_image(cfg.region, args.image_version),
        args.instance_type,
        args.max_runtime,
        code_uri,
        f"{cfg.processed_prefix}/",
        output_uri,
        args.max_train_rows,
    )

    sm = boto3.client("sagemaker", region_name=cfg.region)
    sm.create_processing_job(**request)
    print(f"started {name} on {args.instance_type}; the model goes to {output_uri}")
    desc = wait_for_job(sm, name)
    if desc is None:
        return 1
    summary = summarize(desc)
    print(summary)
    print_job_log(boto3.client("logs", region_name=cfg.region), name)
    if summary["status"] != "Completed":
        return 1
    body = s3.get_object(Bucket=cfg.bucket, Key=f"model-artifacts/{name}/metrics.json")["Body"]
    metrics = json.loads(body.read())
    print(json.dumps({k: v for k, v in metrics.items() if k != "params"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
