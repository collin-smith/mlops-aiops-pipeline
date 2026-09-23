#!/usr/bin/env python3
"""Stage 6 real-time demo: serve the approved model from a SageMaker Serverless endpoint,
call it a few times, then delete everything it created.

This is the single, deliberate exception to the series' batch-only deploy (D-007, amended
by D-029). It lives in scripts/, not src/, because it is never part of the pipeline: run it
by hand, with your own credentials, once. The pipeline's SageMaker role still has no
CreateEndpoint* permission. The Budgets hard-stop only attaches its deny policy to the
project roles, not to your credentials, so this script checks for it and refuses to run
once the cap has been hit.

Why Serverless and not a provisioned endpoint: Serverless bills per invocation and has no
idle charge. Provisioned concurrency would bring the idle charge back, so this script never
sets it and ci.yml fails the build if the setting appears anywhere.

    ROLE=$(terraform -chdir=infra output -raw sagemaker_role_arn)
    python scripts/serverless_demo.py --role-arn "$ROLE" --payload sample_open_requests.csv --n 5
    python scripts/serverless_demo.py --cleanup-only   # if a previous run was interrupted

--payload is a headerless CSV of feature rows in the same format the Batch Transform job
reads (no label column). Everything the script creates is named ``<project>-serverless-demo``
and tagged ``project=<project>`` so the Budget sees it and scripts/nuke.sh sweeps it.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import aws_tags, get_config  # noqa: E402

# Serverless limits: memory 1024–6144 MB in 1 GB steps; concurrency 1–200.
MEMORY_MB = 2048
MAX_CONCURRENCY = 1


def resource_name(project: str) -> str:
    return f"{project}-serverless-demo"


def endpoint_config_request(
    name: str, model_name: str, project: str, *, memory_mb: int = MEMORY_MB
) -> dict:
    """The CreateEndpointConfig request: one Serverless variant, no provisioned concurrency."""
    if memory_mb not in range(1024, 6145, 1024):
        raise ValueError(f"memory_mb must be 1024–6144 in 1024 steps, got {memory_mb}")
    return {
        "EndpointConfigName": name,
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": memory_mb,
                    "MaxConcurrency": MAX_CONCURRENCY,
                },
            }
        ],
        "Tags": [{"Key": "project", "Value": project}],  # same shape as aws_tags()
    }


def read_payload_rows(path: Path, n: int) -> list[str]:
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} has no rows")
    try:
        for value in rows[0].split(","):
            float(value)
    except ValueError:
        msg = f"{path} has a non-numeric first row (a header?); expected bare features"
        raise SystemExit(msg) from None
    return rows[:n]


def summarize_latencies(ms: list[float]) -> dict[str, float]:
    """First call includes the cold start; the rest are warm."""
    warm = ms[1:]
    return {
        "cold_start_ms": round(ms[0], 1),
        "warm_median_ms": round(statistics.median(warm), 1) if warm else float("nan"),
        "calls": len(ms),
    }


def latest_approved_package(sm, group: str) -> str:
    resp = sm.list_model_packages(
        ModelPackageGroupName=group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    pkgs = resp["ModelPackageSummaryList"]
    if not pkgs:
        raise SystemExit(f"no Approved model package in {group} — approve one first (Stage 4)")
    return pkgs[0]["ModelPackageArn"]


def budget_hardstop_active(iam, role_arn: str, project: str) -> bool:
    """True if the Budgets action has attached its deny policy to the project role."""
    role = role_arn.rsplit("/", 1)[-1]
    attached = iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]
    return any(p["PolicyName"] == f"{project}-budget-hardstop-deny" for p in attached)


def cleanup(sm, name: str) -> None:
    """Delete endpoint, config, and model. Each step tolerates 'already gone'; any other
    failure (e.g. the endpoint is still Creating) is reported loudly rather than raised, so
    the remaining deletes still run."""
    for label, call, kwargs in (
        ("endpoint", sm.delete_endpoint, {"EndpointName": name}),
        ("endpoint config", sm.delete_endpoint_config, {"EndpointConfigName": name}),
        ("model", sm.delete_model, {"ModelName": name}),
    ):
        try:
            call(**kwargs)
            print(f"deleted {label} {name}")
        except ClientError as e:
            msg = e.response["Error"].get("Message", "")
            if e.response["Error"]["Code"] == "ValidationException" and "Could not find" in msg:
                print(f"{label} {name}: not found (already deleted)")
            else:
                print(f"!! could not delete {label} {name}: {msg}")
                print("!! re-run with --cleanup-only in a few minutes, or run scripts/nuke.sh")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--role-arn", help="execution role the model runs as (terraform output)")
    ap.add_argument("--payload", type=Path, help="headerless CSV of feature rows")
    ap.add_argument("--n", type=int, default=5, help="number of invocations (default 5)")
    ap.add_argument("--cleanup-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = get_config()
    name = resource_name(cfg.project)
    sm = boto3.client("sagemaker", region_name=cfg.region)

    if args.cleanup_only:
        cleanup(sm, name)
        return 0
    if not (args.role_arn and args.payload):
        ap.error("--role-arn and --payload are required unless --cleanup-only")

    rows = read_payload_rows(args.payload, args.n)
    if budget_hardstop_active(boto3.client("iam"), args.role_arn, cfg.project):
        raise SystemExit("budget hard-stop is active (deny policy attached); not creating anything")
    package_arn = latest_approved_package(sm, cfg.model_package_group)
    print(f"serving {package_arn}")

    runtime = boto3.client("sagemaker-runtime", region_name=cfg.region)
    tags = aws_tags()
    try:
        sm.create_model(
            ModelName=name,
            ExecutionRoleArn=args.role_arn,
            PrimaryContainer={"ModelPackageName": package_arn},
            Tags=tags,
        )
        sm.create_endpoint_config(**endpoint_config_request(name, name, cfg.project))
        sm.create_endpoint(EndpointName=name, EndpointConfigName=name, Tags=tags)
        print("waiting for endpoint to be InService (a few minutes)…")
        sm.get_waiter("endpoint_in_service").wait(
            EndpointName=name, WaiterConfig={"Delay": 15, "MaxAttempts": 40}
        )

        latencies: list[float] = []
        for row in rows:
            t0 = time.perf_counter()
            resp = runtime.invoke_endpoint(EndpointName=name, ContentType="text/csv", Body=row)
            latencies.append((time.perf_counter() - t0) * 1000)
            score = resp["Body"].read().decode().strip()
            print(f"  breach probability {score:>8}  ({latencies[-1]:.0f} ms)")
        print(summarize_latencies(latencies))
    finally:
        cleanup(sm, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
