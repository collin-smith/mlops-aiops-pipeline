#!/usr/bin/env python3
"""Create or update the training pipeline, run it, and report each step (Stages 3-4).

    ROLE=$(terraform -chdir=infra output -raw sagemaker_role_arn)
    uv run python scripts/pipeline.py --role-arn "$ROLE"                  # full run
    uv run python scripts/pipeline.py --role-arn "$ROLE" --max-train-rows 200000
    uv run python scripts/pipeline.py --role-arn "$ROLE" --show           # definition only

A run uploads src/ to s3://$MLOPS_BUCKET/code/<content hash>/ (unchanged code keeps its
location, so cached steps stay cached), creates or updates the pipeline from
src/pipeline/definition.py, starts it, and waits. Ctrl+C stops the run. At the end it
prints each step's status, whether it came from the cache, its billed time, the cost tag
on its job, the log of any failed step, the gate's decision, and the two reports.
A run the gate rejects ends Failed; that's the gate working, and it says so.

--show prints the definition and makes no AWS calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.config import aws_tags, get_config  # noqa: E402
from src.common.guards import budget_hardstop_active  # noqa: E402
from src.common.metrics import estimate_cost_usd  # noqa: E402
from src.common.sm_jobs import (  # noqa: E402
    PROCESSING_INSTANCE_TYPES,
    code_files,
    print_job_log,
    upload_files,
    xgboost_image,
)
from src.pipeline.definition import build_definition  # noqa: E402
from src.promote.fairness import summary_lines  # noqa: E402

DONE = ("Succeeded", "Failed", "Stopped")
DESCRIPTION = "validate -> train -> evaluate -> gate -> register for approval (D-039, D-041)"


def pipeline_name(project: str) -> str:
    return f"{project}-train"


def code_hash(files: dict[str, Path]) -> str:
    """A short hash of every uploaded path and its bytes."""
    h = hashlib.sha256()
    for key in sorted(files):
        h.update(key.encode())
        h.update(files[key].read_bytes())
    return h.hexdigest()[:12]


def latest_asof(s3, bucket: str) -> str:
    """The newest raw snapshot date; processed/311 is built from it (D-036)."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix="raw/311/asof=", Delimiter="/")
    dates = sorted(
        p["Prefix"].split("asof=")[1].rstrip("/") for p in resp.get("CommonPrefixes", [])
    )
    if not dates:
        raise SystemExit(f"no raw/311/asof=... snapshot in s3://{bucket}")
    return dates[-1]


def upsert_pipeline(sm, name: str, definition: dict, role_arn: str) -> str:
    body = json.dumps(definition)
    try:
        sm.describe_pipeline(PipelineName=name)
    except sm.exceptions.ResourceNotFound:
        sm.create_pipeline(
            PipelineName=name,
            PipelineDefinition=body,
            PipelineDescription=DESCRIPTION,
            RoleArn=role_arn,
            Tags=aws_tags(),  # the pipeline passes its tags on to every job it starts
        )
        return "created"
    sm.update_pipeline(
        PipelineName=name,
        PipelineDefinition=body,
        PipelineDescription=DESCRIPTION,
        RoleArn=role_arn,
    )
    return "updated"


def run_parameters(**values: str) -> list[dict]:
    return [{"Name": k, "Value": str(v)} for k, v in values.items()]


def list_steps(sm, execution_arn: str) -> list[dict]:
    pages = sm.get_paginator("list_pipeline_execution_steps").paginate(
        PipelineExecutionArn=execution_arn, SortOrder="Ascending"
    )
    return [s for page in pages for s in page["PipelineExecutionSteps"]]


def wait_for_execution(sm, execution_arn: str, poll_seconds: int = 20) -> str | None:
    """Print each step's status as it changes. Ctrl+C stops the run and returns None."""
    seen: dict[str, str] = {}
    try:
        while True:
            for s in list_steps(sm, execution_arn):
                if seen.get(s["StepName"]) != s["StepStatus"]:
                    seen[s["StepName"]] = s["StepStatus"]
                    print(f"  {time.strftime('%H:%M:%S')}  {s['StepName']:<9} {s['StepStatus']}")
            status = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)[
                "PipelineExecutionStatus"
            ]
            if status in DONE:
                return status
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        sm.stop_pipeline_execution(PipelineExecutionArn=execution_arn)
        print("stopping the run; its jobs stop with it")
        return None


def step_report(sm, steps: list[dict], project: str, instance_type: str) -> list[dict]:
    """One row per step: status, cache hit, job, billed seconds, estimated cost, cost tag."""
    rows = []
    for s in steps:
        job_arn = s.get("Metadata", {}).get("ProcessingJob", {}).get("Arn")
        row = {
            "step": s["StepName"],
            "status": s["StepStatus"],
            "cached": "CacheHitResult" in s,
            "job": job_arn.rsplit("/", 1)[-1] if job_arn else None,
        }
        if s.get("FailureReason"):
            row["failure_reason"] = s["FailureReason"]
        meta = s.get("Metadata", {})
        if "Condition" in meta:
            row["outcome"] = meta["Condition"].get("Outcome")
        if "RegisterModel" in meta:
            row["model_package"] = meta["RegisterModel"].get("Arn")
        if "Fail" in meta:
            row["reason"] = meta["Fail"].get("ErrorMessage")
        if job_arn and not row["cached"]:
            desc = sm.describe_processing_job(ProcessingJobName=row["job"])
            start, end = desc.get("ProcessingStartTime"), desc.get("ProcessingEndTime")
            if start and end:
                row["billed_seconds"] = round((end - start).total_seconds())
                row["usd"] = estimate_cost_usd(instance_type, row["billed_seconds"])
            tags = sm.list_tags(ResourceArn=job_arn)["Tags"]
            row["tagged"] = {"Key": "project", "Value": project} in tags
        rows.append(row)
    return rows


def read_report(sm, s3, job_name: str, file: str) -> dict:
    desc = sm.describe_processing_job(ProcessingJobName=job_name)
    uri = desc["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"].rstrip("/")
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return json.loads(s3.get_object(Bucket=bucket, Key=f"{key}/{file}")["Body"].read())


def verdict(status: str, rows: list[dict]) -> str:
    """What the run means, in one line: a gate rejection isn't a broken pipeline."""
    gate = next((r for r in rows if r["step"] == "Gate"), None)
    if status == "Succeeded" and gate and gate.get("outcome") == "True":
        package = next((r.get("model_package") for r in rows if r["step"] == "Register"), None)
        return f"PASSED the gate; registered {package} as PendingManualApproval"
    if gate and gate.get("outcome") == "False":
        return "REJECTED by the gate; nothing was registered"
    return f"run {status} before the gate decided"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--role-arn", required=True, help="the project SageMaker role (terraform output)"
    )
    ap.add_argument("--instance-type", default="ml.t3.xlarge", choices=PROCESSING_INSTANCE_TYPES)
    ap.add_argument("--asof", help="snapshot date (default: the newest raw/311 snapshot in S3)")
    ap.add_argument("--max-train-rows", type=int, default=0, help="sample the training rows")
    ap.add_argument("--image-version", default="3.2-0", help="built-in XGBoost image tag")
    ap.add_argument("--show", action="store_true", help="print the definition; no AWS calls")
    args = ap.parse_args(argv)

    cfg = get_config()
    files = code_files()
    code_uri = f"s3://{cfg.bucket}/code/{code_hash(files)}/"

    def definition(asof: str) -> dict:
        return build_definition(
            bucket=cfg.bucket,
            region=cfg.region,
            project=cfg.project,
            metric_namespace=cfg.metric_namespace,
            image_uri=xgboost_image(cfg.region, args.image_version),
            role_arn=args.role_arn,
            code_uri=code_uri,
            asof=asof,
            instance_type=args.instance_type,
        )

    if args.show:
        print(json.dumps(definition(args.asof or "YYYY-MM-DD"), indent=2))
        return 0

    if budget_hardstop_active(boto3.client("iam"), args.role_arn, cfg.project):
        raise SystemExit("budget hard-stop is active (deny policy attached); not starting a run")

    s3 = boto3.client("s3", region_name=cfg.region)
    sm = boto3.client("sagemaker", region_name=cfg.region)
    asof = args.asof or latest_asof(s3, cfg.bucket)
    upload_files(s3, cfg.bucket, code_uri.removeprefix(f"s3://{cfg.bucket}/").rstrip("/"), files)
    name = pipeline_name(cfg.project)
    print(f"{upsert_pipeline(sm, name, definition(asof), args.role_arn)} pipeline {name}")

    params = run_parameters(
        InstanceType=args.instance_type,
        AsOf=asof,
        CodeUri=code_uri,
        MaxTrainRows=args.max_train_rows,
    )
    arn = sm.start_pipeline_execution(PipelineName=name, PipelineParameters=params)[
        "PipelineExecutionArn"
    ]
    print(f"started {arn.rsplit('/', 1)[-1]}: snapshot {asof}, code {code_uri}")
    status = wait_for_execution(sm, arn)
    if status is None:
        return 1

    rows = step_report(sm, list_steps(sm, arn), cfg.project, args.instance_type)
    print(f"\nrun {status}: {verdict(status, rows)}")
    for r in rows:
        print("  " + json.dumps(r))
    billed = sum(r.get("usd", 0.0) for r in rows)
    print(f"estimated compute this run: ${billed:.3f} (cached steps cost nothing)")
    if any(r.get("tagged") is False for r in rows):
        print("WARNING: a step's job is missing the project cost tag; check before the next run")

    logs = boto3.client("logs", region_name=cfg.region)
    for r in rows:
        if r["status"] == "Failed" and r["job"]:
            print(f"\n{r['step']} log:")
            print_job_log(logs, r["job"])
    evaluate = next((r for r in rows if r["step"] == "Evaluate" and r["job"]), None)
    if evaluate and evaluate["status"] == "Succeeded":
        print(json.dumps(read_report(sm, s3, evaluate["job"], "evaluation.json"), indent=2))
        print("\n".join(summary_lines(read_report(sm, s3, evaluate["job"], "fairness.json"))))
    return 0 if status == "Succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
