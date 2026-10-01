#!/usr/bin/env python3
"""Stage 6 real-time demo: serve the shadow model from a SageMaker Serverless endpoint,
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

What it serves (D-044): the newest version in the shadow group, the model the gate rejected,
so every response says ``usage: shadow-not-for-use``. The built-in container's default
handler can't read this model's categorical features, so the model runs with
``src/deploy/inference.py``, packed as ``sourcedir.tar.gz``. The payload is the
``payload.json`` the batch scoring job wrote: a few scored rows, with the features the batch
job computed. A real-time caller couldn't compute the history features from one request,
which is the point the demo makes. The script checks each real-time score against the
batch score for the same row.

    ROLE=$(terraform -chdir=infra output -raw sagemaker_role_arn)
    aws s3 cp s3://$MLOPS_BUCKET/scored-reports/<score job>/payload.json .
    uv run python scripts/serverless_demo.py --role-arn "$ROLE" --payload payload.json --n 5
    uv run python scripts/serverless_demo.py --cleanup-only   # if a previous run was interrupted

Everything the script creates is named ``<project>-serverless-demo`` and tagged
``project=<project>`` so the Budget sees it and scripts/nuke.sh sweeps it.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import tarfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import aws_tags, get_config  # noqa: E402
from src.common.guards import budget_hardstop_active  # noqa: E402
from src.deploy import shadow  # noqa: E402

HANDLER = Path(__file__).resolve().parents[1] / "src" / "deploy" / "inference.py"
# agreement with the batch score; the two run the same booster on the same features
SCORE_TOLERANCE = 1e-4

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


def read_payload(path: Path, n: int) -> list[tuple[dict, float]]:
    """``(feature row, batch score)`` pairs from the scoring job's payload.json."""
    data = json.loads(path.read_text())
    if not {"rows", "batch_scores"} <= set(data):
        raise SystemExit(f"{path} isn't a scoring job payload.json (needs rows and batch_scores)")
    if not data["rows"]:
        raise SystemExit(f"{path} has no rows")
    return list(zip(data["rows"], data["batch_scores"], strict=True))[:n]


def handler_tarball(handler: Path = HANDLER) -> bytes:
    """sourcedir.tar.gz: the inference handler alone, at the archive's root."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(handler, arcname="inference.py")
    return buf.getvalue()


def container(desc: dict, source_uri: str, region: str) -> dict:
    """The shadow version's image and model, run with our handler."""
    (spec,) = desc["InferenceSpecification"]["Containers"]
    return {
        "Image": spec["Image"],
        "ModelDataUrl": spec["ModelDataUrl"],
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": source_uri,
            "SAGEMAKER_REGION": region,
        },
    }


def summarize_latencies(ms: list[float]) -> dict[str, float]:
    """First call includes the cold start; the rest are warm."""
    warm = ms[1:]
    return {
        "cold_start_ms": round(ms[0], 1),
        "warm_median_ms": round(statistics.median(warm), 1) if warm else float("nan"),
        "calls": len(ms),
    }


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
    ap.add_argument("--payload", type=Path, help="payload.json from the scoring job")
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

    rows = read_payload(args.payload, args.n)
    if budget_hardstop_active(boto3.client("iam"), args.role_arn, cfg.project):
        raise SystemExit("budget hard-stop is active (deny policy attached); not creating anything")
    desc = shadow.pick_version(sm, shadow.shadow_group(cfg.project), None)
    print(f"serving {desc['ModelPackageArn']} (usage={shadow.USAGE})")

    s3 = boto3.client("s3", region_name=cfg.region)
    key = f"code/{name}/sourcedir.tar.gz"
    s3.put_object(Bucket=cfg.bucket, Key=key, Body=handler_tarball())
    runtime = boto3.client("sagemaker-runtime", region_name=cfg.region)
    tags = aws_tags()
    try:
        sm.create_model(
            ModelName=name,
            ExecutionRoleArn=args.role_arn,
            PrimaryContainer=container(desc, f"s3://{cfg.bucket}/{key}", cfg.region),
            Tags=tags,
        )
        sm.create_endpoint_config(**endpoint_config_request(name, name, cfg.project))
        sm.create_endpoint(EndpointName=name, EndpointConfigName=name, Tags=tags)
        print("waiting for endpoint to be InService (a few minutes)…")
        sm.get_waiter("endpoint_in_service").wait(
            EndpointName=name, WaiterConfig={"Delay": 15, "MaxAttempts": 40}
        )

        latencies: list[float] = []
        mismatches = 0
        for row, batch in rows:
            t0 = time.perf_counter()
            resp = runtime.invoke_endpoint(
                EndpointName=name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps({"rows": [row]}),
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            body = json.loads(resp["Body"].read())
            (live,) = body["scores"]
            same = abs(live - batch) <= SCORE_TOLERANCE
            mismatches += not same
            print(
                f"  {body['usage']}  score {live:.4f}  batch {batch:.4f}  "
                f"{'match' if same else 'MISMATCH'}  ({latencies[-1]:.0f} ms)"
            )
        print(summarize_latencies(latencies))
        if mismatches:
            print(f"!! {mismatches} real-time score(s) differ from the batch job's")
    finally:
        cleanup(sm, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
