#!/usr/bin/env python3
"""Review a model version, and approve or reject it under the approver role (Stage 4).

    APPROVER=$(terraform -chdir=infra output -raw approver_role_arn)
    ROLE=$(terraform -chdir=infra output -raw sagemaker_role_arn)
    uv run python scripts/approve.py list
    uv run python scripts/approve.py review 3
    uv run python scripts/approve.py approve 3 --approver-role-arn "$APPROVER" --note "..."
    uv run python scripts/approve.py reject 3 --approver-role-arn "$APPROVER" --note "..."
    uv run python scripts/approve.py separation --training-role-arn "$ROLE" \
        --approver-role-arn "$APPROVER"

``review`` prints what a reviewer should read before deciding: the version's status, the
snapshot and code it came from, the gate it passed, evaluation.json, the per-sector
fairness table and the model card link. ``approve`` and ``reject`` assume the approver role
and record a note with the decision. ``approve`` refuses a version whose fairness report
fails, whatever the note says.

``separation`` asks IAM's policy simulator whether each role may approve, without
changing anything: the training role should come back denied, the approver allowed.

Everything reads your default credentials except the decision itself, which runs as the
approver role. All of it is free.
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
from src.pipeline.definition import model_package_group  # noqa: E402
from src.promote.fairness import summary_lines  # noqa: E402

DECISIONS = {"approve": "Approved", "reject": "Rejected"}


def package_arn(sm, group: str, version: str) -> str:
    """A full ARN passes through; a bare version number is looked up in the group."""
    if version.startswith("arn:"):
        return version
    for page in sm.get_paginator("list_model_packages").paginate(ModelPackageGroupName=group):
        for p in page["ModelPackageSummaryList"]:
            if str(p.get("ModelPackageVersion")) == version:
                return p["ModelPackageArn"]
    raise SystemExit(f"no version {version} in {group}")


def read_s3_json(s3, uri: str) -> dict:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def reports(desc: dict) -> tuple[str | None, str | None]:
    """The evaluation and fairness report URIs the Register step attached."""
    metrics = desc.get("ModelMetrics", {})
    evaluation = metrics.get("ModelQuality", {}).get("Statistics", {}).get("S3Uri")
    fairness = metrics.get("Bias", {}).get("Report", {}).get("S3Uri")
    return evaluation, fairness


def approval_blockers(fairness: dict | None) -> list[str]:
    """Reasons ``approve`` refuses. The gate already checked these; this checks again
    because a version can reach the registry by some other route than the gate."""
    if fairness is None:
        return ["no fairness report attached"]
    if not fairness.get("passed"):
        problems = [p for d in fairness["dimensions"].values() for p in d["problems"]]
        return [f"fairness check failed: {'; '.join(problems)}"]
    return []


def cmd_list(sm, s3, args, group: str) -> int:
    pages = sm.get_paginator("list_model_packages").paginate(
        ModelPackageGroupName=group, SortBy="CreationTime", SortOrder="Ascending"
    )
    rows = [p for page in pages for p in page["ModelPackageSummaryList"]]
    if not rows:
        print(f"{group}: no versions yet")
    for p in rows:
        print(
            f"  v{p.get('ModelPackageVersion')}  {p['ModelApprovalStatus']:<22} "
            f"{p['CreationTime']:%Y-%m-%d %H:%M}  {p.get('ModelPackageDescription', '')}"
        )
    return 0


def cmd_review(sm, s3, args, group: str) -> int:
    arn = package_arn(sm, group, args.version)
    desc = sm.describe_model_package(ModelPackageName=arn)
    print(f"{arn}\nstatus: {desc['ModelApprovalStatus']}")
    if desc.get("ApprovalDescription"):
        print(f"note:   {desc['ApprovalDescription']}")
    for k, v in desc.get("CustomerMetadataProperties", {}).items():
        print(f"{k + ':':<20}{v}")
    evaluation, fairness = reports(desc)
    if evaluation:
        print(json.dumps(read_s3_json(s3, evaluation)["binary_classification_metrics"], indent=2))
    if fairness:
        print("\n".join(summary_lines(read_s3_json(s3, fairness))))
    return 0


def approver_client(role_arn: str, region: str):
    creds = boto3.client("sts", region_name=region).assume_role(
        RoleArn=role_arn, RoleSessionName="model-approval", DurationSeconds=900
    )["Credentials"]
    return boto3.client(
        "sagemaker",
        region_name=region,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )


def cmd_decide(sm, s3, args, group: str) -> int:
    arn = package_arn(sm, group, args.version)
    desc = sm.describe_model_package(ModelPackageName=arn)
    status = DECISIONS[args.command]
    if args.command == "approve":
        _, fairness_uri = reports(desc)
        blockers = approval_blockers(read_s3_json(s3, fairness_uri) if fairness_uri else None)
        if blockers:
            raise SystemExit("not approving: " + "; ".join(blockers))
    approver = approver_client(args.approver_role_arn, get_config().region)
    approver.update_model_package(
        ModelPackageArn=arn, ModelApprovalStatus=status, ApprovalDescription=args.note
    )
    print(f"{arn}: {desc['ModelApprovalStatus']} -> {status}")
    return 0


def simulate(iam, role_arn: str, resource_arn: str) -> str:
    result = iam.simulate_principal_policy(
        PolicySourceArn=role_arn,
        ActionNames=["sagemaker:UpdateModelPackage"],
        ResourceArns=[resource_arn],
    )["EvaluationResults"][0]
    return result["EvalDecision"]


def cmd_separation(sm, s3, args, group: str) -> int:
    """Who may approve? Asked of IAM itself, so it tests the policies as deployed."""
    cfg = get_config()
    account = boto3.client("sts", region_name=cfg.region).get_caller_identity()["Account"]
    version = f"arn:aws:sagemaker:{cfg.region}:{account}:model-package/{group}/1"
    iam = boto3.client("iam")
    expected = {args.training_role_arn: "explicitDeny", args.approver_role_arn: "allowed"}
    ok = True
    for role, want in expected.items():
        got = simulate(iam, role, version)
        ok &= got == want
        print(f"  {role.rsplit('/', 1)[-1]:<28} sagemaker:UpdateModelPackage -> {got}")
    print("separation of duties holds" if ok else "SEPARATION BROKEN: check infra/registry.tf")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the group's versions and their status")
    review = sub.add_parser("review", help="what to read before deciding")
    review.add_argument("version", help="version number or model package ARN")
    for name in DECISIONS:
        p = sub.add_parser(name, help=f"set a version {DECISIONS[name]}, as the approver role")
        p.add_argument("version", help="version number or model package ARN")
        p.add_argument("--approver-role-arn", required=True)
        p.add_argument("--note", required=True, help="why; stored with the decision")
    sep = sub.add_parser("separation", help="IAM simulation: who may approve")
    sep.add_argument("--training-role-arn", required=True)
    sep.add_argument("--approver-role-arn", required=True)
    args = ap.parse_args(argv)

    cfg = get_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    s3 = boto3.client("s3", region_name=cfg.region)
    handler = {
        "list": cmd_list,
        "review": cmd_review,
        "approve": cmd_decide,
        "reject": cmd_decide,
        "separation": cmd_separation,
    }[args.command]
    return handler(sm, s3, args, model_package_group(cfg.project))


if __name__ == "__main__":
    raise SystemExit(main())
