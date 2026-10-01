#!/usr/bin/env python3
"""Shadow-score the open requests on SageMaker, as a Processing job (Stage 6, D-044).

    ROLE=$(terraform -chdir=infra output -raw sagemaker_role_arn)
    uv run python scripts/score_job.py --role-arn "$ROLE" --asof 2026-10-01
    uv run python scripts/score_job.py --role-arn "$ROLE" --asof 2026-10-01 --show

It scores with the newest version in the shadow group (``--version`` picks another) and
refuses one that isn't marked ``shadow-not-for-use``. ``--asof`` is the snapshot date in
``processed/311``; it picks the community lookup frozen with that snapshot and names the
``scored/asof=<date>/`` folder the ``shadow_scores`` table reads.

The job runs ``python -m src.deploy.score`` in AWS's built-in XGBoost image, on one small
instance with a hard time limit and the project tag. It isn't a pipeline step: scoring
runs on its own cadence. Afterwards the script prints the job's log and summary, and
creates or replaces the Athena view ``shadow_outcomes`` (DDL is free). ``--show`` prints
the job request and starts nothing.
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
from src.common.metrics import estimate_cost_usd  # noqa: E402
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
from src.deploy import shadow, views  # noqa: E402

IN_CODE = "/opt/ml/processing/input/code"
IN_DATA = "/opt/ml/processing/input/data"
IN_MODEL = "/opt/ml/processing/input/model"
IN_COMMUNITIES = "/opt/ml/processing/input/communities"
OUT = "/opt/ml/processing/output"


# bash -c <script> <$0> <args...>, as the pipeline's steps do: each entrypoint element must
# stay under 256 characters, so the arguments travel as ContainerArguments.
_BOOT = f'cd {IN_CODE} && pip install -q -r {REQUIREMENTS} && exec python3 -m src.deploy.score "$@"'
ENTRYPOINT = ["bash", "-c", _BOOT, "score"]


def arguments(model_ref: str, model_tag: str) -> list[str]:
    return [
        "--model", IN_MODEL, "--data", IN_DATA, "--communities", IN_COMMUNITIES,
        "--out", OUT, "--model-ref", model_ref, "--model-tag", model_tag,
    ]  # fmt: skip


def _input(name: str, uri: str, local: str) -> dict:
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": uri,
            "LocalPath": local,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
        },
    }


def _output(name: str, local: str, uri: str) -> dict:
    return {
        "OutputName": name,
        "S3Output": {"S3Uri": uri, "LocalPath": local, "S3UploadMode": "EndOfJob"},
    }


def processing_job_request(
    *,
    name: str,
    role_arn: str,
    image_uri: str,
    instance_type: str,
    max_runtime: int,
    bucket: str,
    asof: str,
    code_uri: str,
    model_uri: str,
    model_ref: str,
    model_tag: str,
) -> dict:
    """CreateProcessingJob: one small instance, a hard time limit, tagged, two outputs."""
    if instance_type not in PROCESSING_INSTANCE_TYPES:
        raise ValueError(
            f"{instance_type} has no Processing quota here; use one of {PROCESSING_INSTANCE_TYPES}"
        )
    if not 300 <= max_runtime <= 3600:
        raise ValueError(f"max_runtime must be 300–3600 s, got {max_runtime}")
    return {
        "ProcessingJobName": name,
        "RoleArn": role_arn,
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": ENTRYPOINT,
            "ContainerArguments": arguments(model_ref, model_tag),
        },
        "ProcessingInputs": [
            _input("code", code_uri, IN_CODE),
            _input("data", f"s3://{bucket}/processed/311/", IN_DATA),
            _input("communities", f"s3://{bucket}/raw/communities/asof={asof}/", IN_COMMUNITIES),
            _input("model", model_uri, IN_MODEL),
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                # the table's partition: Parquet only, nothing else in the folder
                _output("scores", f"{OUT}/scores", f"s3://{bucket}/scored/asof={asof}/"),
                _output("report", f"{OUT}/report", f"s3://{bucket}/scored-reports/{name}/"),
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
    ap.add_argument("--asof", required=True, help="the snapshot date in processed/311")
    ap.add_argument("--version", help="shadow group version to score with (default: newest)")
    ap.add_argument("--instance-type", default="ml.t3.xlarge", choices=PROCESSING_INSTANCE_TYPES)
    ap.add_argument("--max-runtime", type=int, default=1800, help="seconds (default 1800)")
    ap.add_argument("--image-version", default="3.2-0", help="built-in XGBoost image tag")
    ap.add_argument("--show", action="store_true", help="print the request; start nothing")
    args = ap.parse_args(argv)

    cfg = get_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    group = shadow.shadow_group(cfg.project)
    desc = shadow.pick_version(sm, group, args.version)
    model_ref = desc["ModelPackageArn"]
    model_tag = f"v{desc['ModelPackageVersion']}"
    model_uri = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    print(f"scoring with {model_ref} ({desc['ModelApprovalStatus']}, usage={shadow.USAGE})")

    name = job_name(cfg.project, "score")
    request = processing_job_request(
        name=name,
        role_arn=args.role_arn,
        image_uri=xgboost_image(cfg.region, args.image_version),
        instance_type=args.instance_type,
        max_runtime=args.max_runtime,
        bucket=cfg.bucket,
        asof=args.asof,
        code_uri=f"s3://{cfg.bucket}/code/{name}/",
        model_uri=model_uri,
        model_ref=model_ref,
        model_tag=model_tag,
    )
    if args.show:
        print(json.dumps(request, indent=2))
        return 0
    if budget_hardstop_active(boto3.client("iam"), args.role_arn, cfg.project):
        raise SystemExit("budget hard-stop is active (deny policy attached); not starting a job")

    s3 = boto3.client("s3", region_name=cfg.region)
    upload_files(s3, cfg.bucket, f"code/{name}", code_files())
    sm.create_processing_job(**request)
    print(
        f"started {name} on {args.instance_type}; scores go to s3://{cfg.bucket}/scored/asof={args.asof}/"
    )
    job = wait_for_job(sm, name)
    if job is None:
        return 1
    summary = summarize(job)
    if "billed_seconds" in summary:
        summary["usd"] = round(estimate_cost_usd(args.instance_type, summary["billed_seconds"]), 4)
    print(summary)
    print_job_log(boto3.client("logs", region_name=cfg.region), name)
    if summary["status"] != "Completed":
        return 1
    key = f"scored-reports/{name}/summary.json"
    report = json.loads(s3.get_object(Bucket=cfg.bucket, Key=key)["Body"].read())
    print(json.dumps(report, indent=2))

    from src.common.athena import run_query

    print(f"views: {', '.join(views.create_views(run_query))}")
    print("the article's table, in the Athena console:")
    print(views.TOP_FLAGGED.format(asof=args.asof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
