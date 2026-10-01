#!/usr/bin/env python3
"""Register a rejected pipeline run's model in the shadow group, once (Stage 6, D-044).

    uv run python scripts/register_shadow.py vhhhe5vtr34a --dry-run   # print, change nothing
    uv run python scripts/register_shadow.py vhhhe5vtr34a

It reads the run's own Train and Evaluate jobs for the model and the two reports, checks
the gate rejected the run (a model that passed is already in the main group, awaiting the
approver), and registers it in ``<project>-breach-risk-shadow`` as ``Rejected``, marked
``usage = shadow-not-for-use``. A run that already has a shadow version isn't registered
again. Runs under your own credentials; registering is free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.config import get_config  # noqa: E402
from src.common.sm_jobs import xgboost_image  # noqa: E402
from src.deploy import shadow  # noqa: E402
from src.promote.fairness import summary_lines  # noqa: E402


def read_s3_json(s3, uri: str) -> dict:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("execution", help="pipeline execution id (e.g. vhhhe5vtr34a) or its ARN")
    ap.add_argument("--image-version", default="3.2-0", help="built-in XGBoost image tag")
    ap.add_argument("--dry-run", action="store_true", help="print the request, register nothing")
    args = ap.parse_args(argv)

    cfg = get_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    s3 = boto3.client("s3", region_name=cfg.region)
    account = boto3.client("sts", region_name=cfg.region).get_caller_identity()["Account"]
    exec_arn = shadow.execution_arn(cfg.region, account, f"{cfg.project}-train", args.execution)
    execution_id = exec_arn.rsplit("/", 1)[-1]
    group = shadow.shadow_group(cfg.project)

    outputs = shadow.run_outputs(sm, exec_arn)
    if outputs["gate_outcome"] != "False":
        raise SystemExit(
            f"run {execution_id}: gate outcome {outputs['gate_outcome']!r}, not a rejection. "
            "A model that passed belongs in the main group (scripts/approve.py)."
        )
    if execution_id in shadow.registered_executions(sm, group):
        raise SystemExit(f"run {execution_id} already has a version in {group}")

    fairness = read_s3_json(s3, outputs["fairness"])
    print("\n".join(summary_lines(fairness)))
    request = shadow.registration_request(
        group=group,
        image_uri=xgboost_image(cfg.region, args.image_version),
        outputs=outputs,
        params=shadow.run_parameters(sm, exec_arn),
        execution_id=execution_id,
        fairness=fairness,
    )
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0
    arn = sm.create_model_package(**request)["ModelPackageArn"]
    print(f"registered {arn}: Rejected, usage={shadow.USAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
