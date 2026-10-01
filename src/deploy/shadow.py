"""The shadow model group (Stage 6, D-044): where a model the gate rejected is recorded
before it scores real requests, labelled not-for-use.

A rejected pipeline run registers nothing; it ends at the gate's Fail step. To shadow-score
with its model, ``scripts/register_shadow.py`` registers the run's artifact here, as
``Rejected`` (which is what the gate decided), with the run's reports attached. No project
role may change a version's status in this group (``infra/shadow.tf``), and the scoring
launcher refuses a version that somehow ends up ``Approved``.

These helpers make no AWS calls of their own; the clients are passed in.
"""

from __future__ import annotations

from src.pipeline.definition import MODEL_CARD_URL

USAGE = "shadow-not-for-use"
PIPELINE_STEPS = ("Train", "Evaluate")


def shadow_group(project: str) -> str:
    """Must match infra/shadow.tf."""
    return f"{project}-breach-risk-shadow"


def execution_arn(region: str, account: str, pipeline: str, execution_id: str) -> str:
    """A bare execution id (``vhhhe5vtr34a``) as the full ARN the API wants."""
    if execution_id.startswith("arn:"):
        return execution_id
    return f"arn:aws:sagemaker:{region}:{account}:pipeline/{pipeline}/execution/{execution_id}"


def run_outputs(sm, exec_arn: str) -> dict:
    """The run's model and reports, read from the jobs that actually ran.

    A cached step's output lives under the run that first computed it, not under this run's
    own prefix, so the paths come from each step's Processing job, not from a pattern.
    """
    steps = [
        s
        for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
            PipelineExecutionArn=exec_arn
        )
        for s in page["PipelineExecutionSteps"]
    ]
    by_name = {s["StepName"]: s for s in steps}
    gate = by_name.get("Gate", {}).get("Metadata", {}).get("Condition", {}).get("Outcome")
    uris = {}
    for name in PIPELINE_STEPS:
        step = by_name.get(name)
        if not step or step.get("StepStatus") != "Succeeded":
            raise SystemExit(f"{exec_arn}: step {name} didn't succeed, nothing to register")
        job = step["Metadata"]["ProcessingJob"]["Arn"].rsplit("/", 1)[-1]
        desc = sm.describe_processing_job(ProcessingJobName=job)
        (out,) = desc["ProcessingOutputConfig"]["Outputs"]
        uris[name] = out["S3Output"]["S3Uri"].rstrip("/")
    return {
        "gate_outcome": gate,
        "model": f"{uris['Train']}/model.tar.gz",
        "evaluation": f"{uris['Evaluate']}/evaluation.json",
        "fairness": f"{uris['Evaluate']}/fairness.json",
    }


def run_parameters(sm, exec_arn: str) -> dict[str, str]:
    pages = sm.get_paginator("list_pipeline_parameters_for_execution").paginate(
        PipelineExecutionArn=exec_arn
    )
    return {p["Name"]: p["Value"] for page in pages for p in page["PipelineParameters"]}


def registration_request(
    *,
    group: str,
    image_uri: str,
    outputs: dict,
    params: dict[str, str],
    execution_id: str,
    fairness: dict,
) -> dict:
    """CreateModelPackage for the shadow group: Rejected, with the reports and the lineage.

    No tags: AWS refuses them on a version ("Tags are not supported in Model Package
    versions"). The group carries the project tag, and a version costs nothing to keep.
    """
    # fairness.json stores each number as {"value": x}, the shape the gate's JsonGet reads
    ratios = {
        d: (v.get("recall_ratio") or {}).get("value")
        for d, v in fairness.get("dimensions", {}).items()
    }
    return {
        "ModelPackageGroupName": group,
        "ModelPackageDescription": (
            f"SHADOW, NOT FOR USE. Rejected by the gate; run {execution_id}, "
            f"snapshot {params.get('AsOf', '?')}"
        ),
        # what the gate decided; a shadow version is never approved (infra/shadow.tf)
        "ModelApprovalStatus": "Rejected",
        "InferenceSpecification": {
            "Containers": [{"Image": image_uri, "ModelDataUrl": outputs["model"]}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        "ModelMetrics": {
            "ModelQuality": {
                "Statistics": {"ContentType": "application/json", "S3Uri": outputs["evaluation"]}
            },
            "Bias": {"Report": {"ContentType": "application/json", "S3Uri": outputs["fairness"]}},
        },
        "CustomerMetadataProperties": {
            "usage": USAGE,
            "pipeline_execution": execution_id,
            "asof": params.get("AsOf", ""),
            "code_uri": params.get("CodeUri", ""),
            "gate": "rejected",
            "fairness_passed": str(bool(fairness.get("passed"))).lower(),
            **{f"recall_ratio_{d}": str(r) for d, r in ratios.items() if r is not None},
            "model_card": MODEL_CARD_URL,
        },
    }


def versions(sm, group: str) -> list[dict]:
    """The group's versions, newest first."""
    pages = sm.get_paginator("list_model_packages").paginate(
        ModelPackageGroupName=group, SortBy="CreationTime", SortOrder="Descending"
    )
    return [p for page in pages for p in page["ModelPackageSummaryList"]]


def registered_executions(sm, group: str) -> set[str]:
    """Pipeline runs that already have a version here, so a run is registered once."""
    found = set()
    for v in versions(sm, group):
        desc = sm.describe_model_package(ModelPackageName=v["ModelPackageArn"])
        run = desc.get("CustomerMetadataProperties", {}).get("pipeline_execution")
        if run:
            found.add(run)
    return found


def pick_version(sm, group: str, version: str | None) -> dict:
    """The version to score with (newest by default), described. Refuses an Approved one:
    a shadow version that reads Approved means someone changed it by hand."""
    rows = versions(sm, group)
    if not rows:
        raise SystemExit(f"{group} has no versions; run scripts/register_shadow.py first")
    if version is None:
        row = rows[0]
    else:
        matches = [r for r in rows if str(r.get("ModelPackageVersion")) == str(version)]
        if not matches:
            raise SystemExit(f"no version {version} in {group}")
        row = matches[0]
    desc = sm.describe_model_package(ModelPackageName=row["ModelPackageArn"])
    if desc["ModelApprovalStatus"] == "Approved":
        raise SystemExit(f"{row['ModelPackageArn']} is Approved; a shadow version never is. Stop.")
    if desc.get("CustomerMetadataProperties", {}).get("usage") != USAGE:
        raise SystemExit(f"{row['ModelPackageArn']} isn't marked {USAGE}; not scoring with it")
    return desc
