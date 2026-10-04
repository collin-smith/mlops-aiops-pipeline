#!/usr/bin/env python3
"""Model Monitor's analyzer as one-off Processing jobs, no schedule (Stage 7 Layer A, D-045).

    ROLE=$(terraform -chdir=infra output -raw sagemaker_role_arn)
    # 1. smoke: a 1,000-row baseline, to prove the image runs on ml.t3 (about a cent)
    uv run python scripts/monitor_job.py baseline --role-arn "$ROLE" --csv smoke.csv --label smoke
    # 2. the baseline, from the challenger's training rows
    uv run python scripts/monitor_job.py baseline --role-arn "$ROLE" --csv baseline.csv
    # 3. a check: the 30-day intake, or the disclosed injected copy
    uv run python scripts/monitor_job.py check --role-arn "$ROLE" --csv intake.csv \
        --label intake-2026-10-01
    uv run python scripts/monitor_job.py check --role-arn "$ROLE" --csv injected.csv \
        --label injected-2026-10-01 --injected

The CSVs come from ``src.monitor.datasets`` and ``src.monitor.inject_drift``. The script
uploads the CSV, runs AWS's ``sagemaker-model-monitor-analyzer`` image (the one the SDK's
``suggest_baseline`` and monitoring schedules run) as a plain Processing job, and prints
the result. ``CreateMonitoringSchedule`` is never called: there is no endpoint or data
capture to schedule against, nothing here runs on a timer, and ci.yml refuses the call.

The analyzer publishes CloudWatch metrics by default. Every request here sets
``publish_cloudwatch_metrics=Disabled``, so a check adds no custom metrics.

S3 layout: ``monitoring/datasets/<label>/`` (real inputs), ``monitoring/injected/<label>/``
(synthetic), ``monitoring/baseline/<label>/`` (statistics.json, constraints.json),
``monitoring/checks/<label>/`` (constraint_violations.json, statistics.json).
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
    job_name,
    print_job_log,
    summarize,
    wait_for_job,
)

# The analyzer image's registry account, from the SageMaker SDK's
# image_uri_config/model-monitor.json; the launchers need no SDK.
_ANALYZER_REGISTRY = {"ca-central-1": "536280801234"}

IN_DATA = "/opt/ml/processing/input/baseline_dataset_input"
IN_BASELINE = "/opt/ml/processing/baseline"
OUT = "/opt/ml/processing/output"
DATASET_FORMAT = {"csv": {"header": True, "output_columns_position": "START"}}
VIOLATIONS = "constraint_violations.json"


def analyzer_image(region: str) -> str:
    if region not in _ANALYZER_REGISTRY:
        raise SystemExit(f"no analyzer registry listed for {region}; add it in monitor_job.py")
    return (
        f"{_ANALYZER_REGISTRY[region]}.dkr.ecr.{region}.amazonaws.com/"
        "sagemaker-model-monitor-analyzer"
    )


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


def environment(baseline_uri: str | None) -> dict:
    """The analyzer's settings. ``baseline_uri`` None means "suggest a baseline"."""
    env = {
        "dataset_format": json.dumps(DATASET_FORMAT),
        "dataset_source": IN_DATA,
        "output_path": OUT,
        "publish_cloudwatch_metrics": "Disabled",  # no custom metrics, ever (D-045)
    }
    if baseline_uri:
        env["baseline_constraints"] = f"{IN_BASELINE}/constraints.json"
        env["baseline_statistics"] = f"{IN_BASELINE}/statistics.json"
    return env


def analyzer_request(
    *,
    name: str,
    role_arn: str,
    image_uri: str,
    instance_type: str,
    max_runtime: int,
    dataset_uri: str,
    output_uri: str,
    baseline_uri: str | None = None,
) -> dict:
    """CreateProcessingJob for one analyzer run: small, time-limited, tagged, no metrics."""
    if instance_type not in PROCESSING_INSTANCE_TYPES:
        raise ValueError(
            f"{instance_type} has no Processing quota here; use one of {PROCESSING_INSTANCE_TYPES}"
        )
    if not 300 <= max_runtime <= 3600:
        raise ValueError(f"max_runtime must be 300–3600 s, got {max_runtime}")
    inputs = [_input("dataset", dataset_uri, IN_DATA)]
    if baseline_uri:
        inputs.append(_input("baseline", baseline_uri, IN_BASELINE))
    return {
        "ProcessingJobName": name,
        "RoleArn": role_arn,
        "AppSpecification": {"ImageUri": image_uri},
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "monitoring",
                    "S3Output": {"S3Uri": output_uri, "LocalPath": OUT, "S3UploadMode": "EndOfJob"},
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
        "Environment": environment(baseline_uri),
        "Tags": aws_tags(),
    }


def violation_lines(report: dict) -> list[str]:
    """One line per violation in ``constraint_violations.json``."""
    rows = report.get("violations", [])
    if not rows:
        return ["no violations"]
    return [f"{v.get('feature_name')}: {v.get('constraint_check_type')}" for v in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["baseline", "check"])
    ap.add_argument("--role-arn", required=True, help="the project SageMaker role")
    ap.add_argument("--csv", type=Path, required=True, help="the dataset (from src.monitor)")
    ap.add_argument("--label", help="names the S3 folders (default: baseline / the CSV name)")
    ap.add_argument("--baseline-label", default="baseline", help="which baseline a check uses")
    ap.add_argument("--injected", action="store_true", help="a synthetic copy (inject_drift)")
    ap.add_argument("--instance-type", default="ml.t3.xlarge", choices=PROCESSING_INSTANCE_TYPES)
    ap.add_argument("--max-runtime", type=int, default=1200, help="seconds (default 1200)")
    ap.add_argument("--show", action="store_true", help="print the request; start nothing")
    args = ap.parse_args(argv)

    cfg = get_config()
    label = args.label or ("baseline" if args.kind == "baseline" else args.csv.stem)
    zone = "injected" if args.injected else "datasets"
    dataset_prefix = f"monitoring/{zone}/{label}"
    if args.kind == "baseline":
        out_prefix, baseline_uri = f"monitoring/baseline/{label}", None
    else:
        out_prefix = f"monitoring/checks/{label}"
        baseline_uri = f"s3://{cfg.bucket}/monitoring/baseline/{args.baseline_label}/"

    name = job_name(cfg.project, f"monitor-{args.kind}")
    request = analyzer_request(
        name=name,
        role_arn=args.role_arn,
        image_uri=analyzer_image(cfg.region),
        instance_type=args.instance_type,
        max_runtime=args.max_runtime,
        dataset_uri=f"s3://{cfg.bucket}/{dataset_prefix}/",
        output_uri=f"s3://{cfg.bucket}/{out_prefix}/",
        baseline_uri=baseline_uri,
    )
    if args.show:
        print(json.dumps(request, indent=2))
        return 0
    if budget_hardstop_active(boto3.client("iam"), args.role_arn, cfg.project):
        raise SystemExit("budget hard-stop is active (deny policy attached); not starting a job")

    s3 = boto3.client("s3", region_name=cfg.region)
    s3.upload_file(str(args.csv), cfg.bucket, f"{dataset_prefix}/data.csv")
    sm = boto3.client("sagemaker", region_name=cfg.region)
    sm.create_processing_job(**request)
    print(f"started {name} on {args.instance_type}; results go to s3://{cfg.bucket}/{out_prefix}/")
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
    if args.kind == "baseline":
        key = f"{out_prefix}/constraints.json"
        constraints = json.loads(s3.get_object(Bucket=cfg.bucket, Key=key)["Body"].read())
        print(f"baseline: {len(constraints.get('features', []))} features constrained")
        return 0
    key = f"{out_prefix}/{VIOLATIONS}"
    try:
        report = json.loads(s3.get_object(Bucket=cfg.bucket, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        report = {}
    tag = "SYNTHETIC (injected) " if args.injected else ""
    print(f"{tag}violations for {label}:")
    for line in violation_lines(report):
        print("  ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
