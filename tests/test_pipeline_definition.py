"""Stage 3 pipeline definition and its launcher. The definition is data, so it's tested as
data: step order, what each step reads and writes, the cache key, limits. No AWS calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.common import sm_jobs
from src.pipeline import definition as d

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pipeline.py"
_spec = importlib.util.spec_from_file_location("pipeline_launcher", _PATH)
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)

ROLE = "arn:aws:iam::123456789012:role/mlops-aiops-sagemaker"


def _definition(**kw):
    args = dict(
        bucket="b",
        region="ca-central-1",
        project="mlops-aiops",
        metric_namespace="MLOpsAIOps/Pipeline",
        image_uri=sm_jobs.xgboost_image("ca-central-1", "3.2-0"),
        role_arn=ROLE,
        code_uri="s3://b/code/abc123/",
        asof="2026-09-23",
    )
    args.update(kw)
    return d.build_definition(**args)


def _steps(defn=None):
    return {s["Name"]: s for s in (defn or _definition())["Steps"]}


def test_three_processing_steps_in_order():
    steps = _steps()
    assert list(steps) == ["Validate", "Train", "Evaluate"]
    assert all(s["Type"] == "Processing" for s in steps.values())
    assert "DependsOn" not in steps["Validate"]
    assert steps["Train"]["DependsOn"] == ["Validate"]
    assert steps["Evaluate"]["DependsOn"] == ["Train"]


def test_every_step_is_cached_limited_and_small():
    for name, s in _steps().items():
        a = s["Arguments"]
        assert s["CacheConfig"] == {"Enabled": True, "ExpireAfter": "P30D"}
        assert a["StoppingCondition"]["MaxRuntimeInSeconds"] == d.MAX_RUNTIME[name] <= 3600
        cluster = a["ProcessingResources"]["ClusterConfig"]
        assert cluster["InstanceCount"] == 1
        assert cluster["InstanceType"] == {"Get": "Parameters.InstanceType"}
        assert a["RoleArn"] == ROLE


def test_evaluate_reads_the_model_train_wrote():
    inputs = {
        i["InputName"]: i["S3Input"] for i in _steps()["Evaluate"]["Arguments"]["ProcessingInputs"]
    }
    assert inputs["model"]["S3Uri"] == {
        "Get": "Steps.Train.ProcessingOutputConfig.Outputs['model'].S3Output.S3Uri"
    }
    assert inputs["model"]["LocalPath"] == d.IN_MODEL
    assert inputs["data"]["S3Uri"] == "s3://b/processed/311/"


def test_outputs_land_under_the_execution_id():
    for name, s in _steps().items():
        (out,) = s["Arguments"]["ProcessingOutputConfig"]["Outputs"]
        values = out["S3Output"]["S3Uri"]["Std:Join"]["Values"]
        assert values == [
            "s3://b/pipeline-runs",
            {"Get": "Execution.PipelineExecutionId"},
            name.lower(),
        ]


def test_cache_key_changes_with_code_and_snapshot():
    """A Processing step's cache key is its command, environment and input locations."""
    for s in _steps().values():
        a = s["Arguments"]
        code = next(i for i in a["ProcessingInputs"] if i["InputName"] == "code")
        assert code["S3Input"]["S3Uri"] == {"Get": "Parameters.CodeUri"}
        assert a["Environment"]["MLOPS_ASOF"] == {"Get": "Parameters.AsOf"}


def test_entrypoint_runs_the_step_through_the_metrics_wrapper():
    for name, s in _steps().items():
        app = s["Arguments"]["AppSpecification"]
        assert app["ContainerEntrypoint"] == d.ENTRYPOINT
        assert all(len(e) <= 256 for e in app["ContainerEntrypoint"])
        assert "python3 -m src.pipeline.step" in app["ContainerEntrypoint"][2]
        assert app["ContainerArguments"][0] == name.lower()


def test_parameters_carry_the_defaults():
    params = {p["Name"]: p["DefaultValue"] for p in _definition()["Parameters"]}
    assert params == {
        "InstanceType": "ml.t3.xlarge",
        "AsOf": "2026-09-23",
        "CodeUri": "s3://b/code/abc123/",
        "MaxTrainRows": "0",
    }
    assert all(p["Type"] == "String" for p in _definition()["Parameters"])


def test_definition_is_json_and_rejects_no_quota_instances():
    json.dumps(_definition())
    with pytest.raises(ValueError):
        _definition(instance_type="ml.m5.large")


# --- launcher -------------------------------------------------------------------------


def test_code_hash_is_stable_and_sees_changes(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    files = {"src/a.py": a, "src/b.py": b}
    first = launcher.code_hash(files)
    assert first == launcher.code_hash(dict(reversed(files.items())))
    b.write_text("y = 3\n")
    assert launcher.code_hash(files) != first


class _FakeS3:
    def __init__(self, prefixes):
        self.prefixes = prefixes

    def list_objects_v2(self, **kw):
        assert kw["Prefix"] == "raw/311/asof=" and kw["Delimiter"] == "/"
        return {"CommonPrefixes": [{"Prefix": p} for p in self.prefixes]}


def test_latest_asof_picks_the_newest_snapshot():
    s3 = _FakeS3(["raw/311/asof=2026-09-23/", "raw/311/asof=2026-10-14/"])
    assert launcher.latest_asof(s3, "b") == "2026-10-14"
    with pytest.raises(SystemExit):
        launcher.latest_asof(_FakeS3([]), "b")


class _FakeSM:
    class exceptions:  # noqa: N801 — mirrors boto3's client.exceptions
        class ResourceNotFound(Exception):
            pass

    def __init__(self, exists):
        self.exists, self.calls = exists, []

    def describe_pipeline(self, **kw):
        if not self.exists:
            raise self.exceptions.ResourceNotFound()

    def create_pipeline(self, **kw):
        self.calls.append(("create", kw))

    def update_pipeline(self, **kw):
        self.calls.append(("update", kw))


def test_upsert_creates_with_the_cost_tag_then_updates():
    sm = _FakeSM(exists=False)
    assert launcher.upsert_pipeline(sm, "p", {"Steps": []}, ROLE) == "created"
    kind, kw = sm.calls[0]
    assert {"Key": "project", "Value": "mlops-aiops"} in kw["Tags"]
    assert json.loads(kw["PipelineDefinition"]) == {"Steps": []}
    sm = _FakeSM(exists=True)
    assert launcher.upsert_pipeline(sm, "p", {"Steps": []}, ROLE) == "updated"
    assert sm.calls[0][0] == "update"


def test_run_parameters_are_strings():
    assert launcher.run_parameters(MaxTrainRows=0, AsOf="2026-09-23") == [
        {"Name": "MaxTrainRows", "Value": "0"},
        {"Name": "AsOf", "Value": "2026-09-23"},
    ]


def test_step_report_marks_cache_hits_and_skips_their_cost():
    steps = [
        {
            "StepName": "Validate",
            "StepStatus": "Succeeded",
            "CacheHitResult": {"SourcePipelineExecutionArn": "arn:old"},
            "Metadata": {"ProcessingJob": {"Arn": "arn:aws:sagemaker:r:1:processing-job/old-v"}},
        }
    ]
    (row,) = launcher.step_report(None, steps, "mlops-aiops", "ml.t3.xlarge")
    assert row == {"step": "Validate", "status": "Succeeded", "cached": True, "job": "old-v"}
