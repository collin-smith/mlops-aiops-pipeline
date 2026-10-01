"""Stage 6 shadow group, scoring launcher and views (D-044). No AWS calls."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.common import sm_jobs
from src.deploy import shadow, views

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("score_job", ROOT / "scripts" / "score_job.py")
score_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_job)

IMAGE = sm_jobs.xgboost_image("ca-central-1", "3.2-0")
PKG = "arn:aws:sagemaker:ca-central-1:123456789012:model-package/mlops-aiops-breach-risk-shadow/1"


def test_shadow_group_matches_terraform():
    tf = (ROOT / "infra" / "shadow.tf").read_text()
    assert '"${local.name}-breach-risk-shadow"' in tf
    assert shadow.shadow_group("mlops-aiops") == "mlops-aiops-breach-risk-shadow"


def test_execution_arn_from_a_bare_id():
    arn = shadow.execution_arn("ca-central-1", "123", "mlops-aiops-train", "vhhhe5vtr34a")
    assert arn.endswith(":123:pipeline/mlops-aiops-train/execution/vhhhe5vtr34a")
    assert arn.startswith("arn:aws:sagemaker:ca-central-1:")
    assert shadow.execution_arn("x", "y", "z", arn) == arn


class _FakeSM:
    """Two step listings (Train cached from an older run) and their jobs' outputs."""

    def __init__(self, gate="False", packages=(), statuses=None, usage=shadow.USAGE):
        self.gate, self.packages, self.usage = gate, list(packages), usage
        self.statuses = statuses or {}

    def get_paginator(self, op):
        return _Pager(self, op)

    def describe_processing_job(self, ProcessingJobName):  # noqa: N803
        prefix = {
            "train-job": "s3://b/pipeline-runs/OLDRUN/train",
            "eval-job": "s3://b/pipeline-runs/RUN/evaluate",
        }
        return {
            "ProcessingOutputConfig": {
                "Outputs": [{"S3Output": {"S3Uri": prefix[ProcessingJobName]}}]
            }
        }

    def describe_model_package(self, ModelPackageName):  # noqa: N803
        return {
            "ModelPackageArn": ModelPackageName,
            "ModelPackageVersion": int(ModelPackageName.rsplit("/", 1)[-1]),
            "ModelApprovalStatus": self.statuses.get(ModelPackageName, "Rejected"),
            "CustomerMetadataProperties": {"usage": self.usage, "pipeline_execution": "RUN"},
            "InferenceSpecification": {
                "Containers": [{"Image": IMAGE, "ModelDataUrl": "s3://b/m.tar.gz"}]
            },
        }


class _Pager:
    def __init__(self, sm, op):
        self.sm, self.op = sm, op

    def paginate(self, **kw):
        if self.op == "list_pipeline_execution_steps":

            def job(name):
                return {"ProcessingJob": {"Arn": f"arn:aws:sagemaker:r:a:processing-job/{name}"}}

            steps = [
                {"StepName": "Train", "StepStatus": "Succeeded", "Metadata": job("train-job")},
                {"StepName": "Evaluate", "StepStatus": "Succeeded", "Metadata": job("eval-job")},
                {
                    "StepName": "Gate",
                    "StepStatus": "Succeeded",
                    "Metadata": {"Condition": {"Outcome": self.sm.gate}},
                },
            ]
            return [{"PipelineExecutionSteps": steps}]
        if self.op == "list_model_packages":
            rows = [
                {"ModelPackageArn": a, "ModelPackageVersion": int(a.rsplit("/", 1)[-1])}
                for a in self.sm.packages
            ]
            return [{"ModelPackageSummaryList": rows}]
        raise AssertionError(self.op)


def test_run_outputs_follow_each_jobs_own_output_even_when_cached():
    out = shadow.run_outputs(_FakeSM(), "arn:exec/RUN")
    assert out["model"] == "s3://b/pipeline-runs/OLDRUN/train/model.tar.gz"  # not RUN's prefix
    assert out["fairness"] == "s3://b/pipeline-runs/RUN/evaluate/fairness.json"
    assert out["gate_outcome"] == "False"


def test_registration_is_rejected_and_marked_not_for_use():
    req = shadow.registration_request(
        group="g",
        image_uri=IMAGE,
        outputs={"model": "s3://m", "evaluation": "s3://e", "fairness": "s3://f"},
        params={"AsOf": "2026-09-23", "CodeUri": "s3://code/"},
        execution_id="RUN",
        fairness={
            "passed": False,
            # the real fairness.json shape: every number under "value"
            "dimensions": {
                "sector": {"recall_ratio": {"value": 0.7861}},
                "srg": {"recall_ratio": {"value": 0.7238}},
            },
        },
    )
    assert req["ModelApprovalStatus"] == "Rejected"
    meta = req["CustomerMetadataProperties"]
    assert meta["usage"] == "shadow-not-for-use" and meta["fairness_passed"] == "false"
    assert meta["recall_ratio_sector"] == "0.7861" and meta["recall_ratio_srg"] == "0.7238"
    assert req["ModelPackageDescription"].startswith("SHADOW, NOT FOR USE")
    assert req["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] == "s3://m"
    assert "Tags" not in req  # AWS refuses tags on a version; the group has them
    assert all(len(k) <= 128 and len(v) <= 256 for k, v in meta.items())


def test_a_run_is_registered_only_once():
    assert shadow.registered_executions(_FakeSM(packages=[PKG]), "g") == {"RUN"}


def test_pick_version_takes_the_newest_and_refuses_approved_or_unmarked():
    sm = _FakeSM(packages=[PKG[:-1] + "2", PKG])
    assert shadow.pick_version(sm, "g", None)["ModelPackageVersion"] == 2
    assert shadow.pick_version(sm, "g", "1")["ModelPackageVersion"] == 1
    with pytest.raises(SystemExit, match="Approved"):
        shadow.pick_version(_FakeSM(packages=[PKG], statuses={PKG: "Approved"}), "g", None)
    with pytest.raises(SystemExit, match="isn't marked"):
        shadow.pick_version(_FakeSM(packages=[PKG], usage="production"), "g", None)
    with pytest.raises(SystemExit, match="no versions"):
        shadow.pick_version(_FakeSM(), "g", None)


def test_shadow_versions_are_denied_to_the_approver():
    tf = (ROOT / "infra" / "shadow.tf").read_text()
    assert 'Effect   = "Deny"' in tf and "sagemaker:UpdateModelPackage" in tf
    assert "local.shadow_versions_arn" in tf


# --- the scoring launcher ---


def _request(**kw):
    args = dict(
        name="mlops-aiops-score-20261001-000000",
        role_arn="arn:aws:iam::123456789012:role/mlops-aiops-sagemaker",
        image_uri=IMAGE,
        instance_type="ml.t3.xlarge",
        max_runtime=1800,
        bucket="b",
        asof="2026-10-01",
        code_uri="s3://b/code/job/",
        model_uri="s3://b/pipeline-runs/RUN/train/model.tar.gz",
        model_ref=PKG,
        model_tag="v1",
    )
    args.update(kw)
    return score_job.processing_job_request(**args)


def test_launcher_mounts_the_snapshot_its_communities_and_the_shadow_model():
    req = _request()
    inputs = {i["InputName"]: i["S3Input"]["S3Uri"] for i in req["ProcessingInputs"]}
    assert inputs["data"] == "s3://b/processed/311/"
    assert inputs["communities"] == "s3://b/raw/communities/asof=2026-10-01/"
    assert inputs["model"] == "s3://b/pipeline-runs/RUN/train/model.tar.gz"
    args = req["AppSpecification"]["ContainerArguments"]
    assert args[args.index("--model-ref") + 1] == PKG


def test_launcher_writes_scores_to_the_table_partition_and_reports_elsewhere():
    config = _request()["ProcessingOutputConfig"]
    outputs = {o["OutputName"]: o["S3Output"] for o in config["Outputs"]}
    assert outputs["scores"]["S3Uri"] == "s3://b/scored/asof=2026-10-01/"
    assert outputs["scores"]["LocalPath"].endswith("/scores")
    assert outputs["report"]["S3Uri"].startswith("s3://b/scored-reports/")


def test_launcher_is_small_time_limited_and_tagged():
    req = _request()
    assert req["ProcessingResources"]["ClusterConfig"]["InstanceCount"] == 1
    assert req["StoppingCondition"]["MaxRuntimeInSeconds"] == 1800
    assert {"Key": "project", "Value": "mlops-aiops"} in req["Tags"]
    app = req["AppSpecification"]
    assert all(len(a) <= 256 for a in app["ContainerEntrypoint"] + app["ContainerArguments"])
    with pytest.raises(ValueError):
        _request(instance_type="ml.m5.large")
    with pytest.raises(ValueError):
        _request(max_runtime=7200)


def test_launcher_runs_the_scoring_module_not_a_pipeline_step():
    cmd = _request()["AppSpecification"]["ContainerEntrypoint"][2]
    assert "-m src.deploy.score" in cmd and "src.pipeline.step" not in cmd  # no new metrics


# --- the views ---


def test_outcomes_view_decides_late_on_time_or_undecided():
    sql = views.SHADOW_OUTCOMES
    assert sql.startswith("CREATE OR REPLACE VIEW shadow_outcomes")
    assert "FROM shadow_scores s" in sql and "JOIN processed_311 p" in sql
    for outcome in ("'on_time'", "'late'", "'undecided'"):
        assert outcome in sql
    assert "s.usage" in sql


def test_top_flagged_query_keeps_the_usage_column_first():
    sql = views.TOP_FLAGGED.format(asof="2026-10-01")
    assert sql.startswith("SELECT usage,") and "asof = '2026-10-01'" in sql


def test_create_views_runs_each_view():
    ran = []
    assert views.create_views(ran.append) == ["shadow_outcomes"]
    assert ran == [views.SHADOW_OUTCOMES]


def test_table_columns_match_the_scores_file():
    from src.deploy.score import OUTPUT_COLUMNS

    tf = (ROOT / "infra" / "shadow.tf").read_text()
    table = tf[tf.index('resource "aws_glue_catalog_table" "shadow_scores"') :]
    names = [
        line.split('"')[1]
        for line in table.splitlines()
        if line.strip().startswith("name =") and "shadow_scores" not in line
    ]
    assert names[1:] == OUTPUT_COLUMNS  # names[0] is the partition key, asof
    assert names[0] == "asof"
