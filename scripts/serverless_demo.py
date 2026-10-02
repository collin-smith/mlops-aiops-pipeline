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

What it serves (D-044): the newest version in the shadow group, the model the gate rejected;
every line it prints says ``shadow-not-for-use``. It runs on the image's **default** handler,
which reads a headerless CSV of numbers. This model was trained on categorical columns, so
each category travels as its code in the model's own level order (``feature_schema.json``),
and a category the model never saw is sent empty. Codes in any other order give wrong scores
with no error, which is why the conversion uses the schema and a test checks it. A custom
handler (``SAGEMAKER_PROGRAM``) isn't an option on Serverless with this image: its script-mode
server writes ``/etc/sagemaker-nginx.conf`` and Serverless containers can't (first run,
2026-10-01: "PermissionError"). The endpoint serves a copy of the model that holds only
``xgboost-model``, so the default loader can't pick up the schema file instead.

The payload is the ``payload.json`` the batch scoring job wrote: a few scored rows, with the
features the batch job computed. A real-time caller couldn't compute the history features
from one request, which is the point the demo makes. Each real-time score is checked against
the batch score for the same row.

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
import math
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

MODEL_FILE = "xgboost-model"
# agreement with the batch score; the two run the same booster on the same features
SCORE_TOLERANCE = 1e-4
# Serverless with a failing container takes 10+ minutes to give up; wait up to 15
WAIT = {"Delay": 15, "MaxAttempts": 60}

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


def csv_row(row: dict, schema: dict) -> str:
    """One payload row as the default handler's CSV: columns in training order, each
    category as its code in the model's level order, unseen or missing values empty."""
    values = []
    for col in schema["columns"]:
        v = row.get(col)
        if col in schema["categories"]:
            levels = schema["categories"][col]
            v = levels.index(str(v)) if v is not None and str(v) in levels else None
        if v is None or (isinstance(v, float) and math.isnan(v)):
            values.append("")
        else:
            values.append(repr(float(v)) if isinstance(v, float) else str(v))
    return ",".join(values)


def read_artifact(model_tar: bytes) -> tuple[dict, bytes]:
    """The feature schema and the booster's bytes from a model.tar.gz."""
    with tarfile.open(fileobj=io.BytesIO(model_tar)) as tar:
        schema = json.load(tar.extractfile("feature_schema.json"))
        booster = tar.extractfile(MODEL_FILE).read()
    return schema, booster


def model_only_tarball(booster: bytes) -> bytes:
    """model.tar.gz with the booster alone, so the default loader has one file to load."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(MODEL_FILE)
        info.size = len(booster)
        tar.addfile(info, io.BytesIO(booster))
    return buf.getvalue()


def container(desc: dict, model_url: str) -> dict:
    """The shadow version's image, serving the model-only copy with the default handler."""
    (spec,) = desc["InferenceSpecification"]["Containers"]
    return {"Image": spec["Image"], "ModelDataUrl": model_url}


def summarize_latencies(ms: list[float]) -> dict[str, float]:
    """First call includes the cold start; the rest are warm."""
    warm = ms[1:]
    return {
        "cold_start_ms": round(ms[0], 1),
        "warm_median_ms": round(statistics.median(warm), 1) if warm else float("nan"),
        "calls": len(ms),
    }


def settle(sm, name: str, poll_seconds: int = 20, max_polls: int = 45) -> None:
    """Wait while the endpoint is still Creating or Updating: it can't be deleted until then.
    The first run's cleanup failed this way after a 10-minute wait (2026-10-01)."""
    for _ in range(max_polls):
        try:
            status = sm.describe_endpoint(EndpointName=name)["EndpointStatus"]
        except ClientError:
            return  # not there (never created, or already gone)
        if status not in ("Creating", "Updating"):
            return
        print(f"endpoint is {status}; waiting before deleting it…")
        time.sleep(poll_seconds)


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
    (spec,) = desc["InferenceSpecification"]["Containers"]
    bucket, key = spec["ModelDataUrl"].removeprefix("s3://").split("/", 1)
    schema, booster = read_artifact(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    model_key = f"code/{name}/model.tar.gz"  # the same booster bytes, alone in the archive
    s3.put_object(Bucket=cfg.bucket, Key=model_key, Body=model_only_tarball(booster))
    runtime = boto3.client("sagemaker-runtime", region_name=cfg.region)
    tags = aws_tags()
    try:
        sm.create_model(
            ModelName=name,
            ExecutionRoleArn=args.role_arn,
            PrimaryContainer=container(desc, f"s3://{cfg.bucket}/{model_key}"),
            Tags=tags,
        )
        sm.create_endpoint_config(**endpoint_config_request(name, name, cfg.project))
        sm.create_endpoint(EndpointName=name, EndpointConfigName=name, Tags=tags)
        print("waiting for endpoint to be InService (a few minutes)…")
        sm.get_waiter("endpoint_in_service").wait(EndpointName=name, WaiterConfig=WAIT)

        latencies: list[float] = []
        mismatches = 0
        for row, batch in rows:
            t0 = time.perf_counter()
            resp = runtime.invoke_endpoint(
                EndpointName=name, ContentType="text/csv", Body=csv_row(row, schema)
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            live = float(resp["Body"].read().decode().strip().split(",")[0])
            same = abs(live - batch) <= SCORE_TOLERANCE
            mismatches += not same
            print(
                f"  {shadow.USAGE}  score {live:.4f}  batch {batch:.4f}  "
                f"{'match' if same else 'MISMATCH'}  ({latencies[-1]:.0f} ms)"
            )
        print(summarize_latencies(latencies))
        if mismatches:
            print(f"!! {mismatches} real-time score(s) differ from the batch job's")
    finally:
        settle(sm, name)
        cleanup(sm, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
