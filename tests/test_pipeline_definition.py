"""Stage 3-4 pipeline definition and its launcher. The definition is data, so it's tested as
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


def _processing():
    return {n: s for n, s in _steps().items() if s["Type"] == "Processing"}


def test_three_processing_steps_then_the_gate():
    steps = _steps()
    assert list(steps) == ["Validate", "Train", "Evaluate", "Gate"]
    assert list(_processing()) == ["Validate", "Train", "Evaluate"]
    assert "DependsOn" not in steps["Validate"]
    assert steps["Train"]["DependsOn"] == ["Validate"]
    assert steps["Evaluate"]["DependsOn"] == ["Train"]


def test_every_step_is_cached_limited_and_small():
    for name, s in _processing().items():
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
    for name, s in _processing().items():
        (out,) = s["Arguments"]["ProcessingOutputConfig"]["Outputs"]
        values = out["S3Output"]["S3Uri"]["Std:Join"]["Values"]
        assert values == [
            "s3://b/pipeline-runs",
            {"Get": "Execution.PipelineExecutionId"},
            name.lower(),
        ]


def test_cache_key_changes_with_code_and_snapshot():
    """A Processing step's cache key is its command, environment and input locations."""
    for s in _processing().values():
        a = s["Arguments"]
        code = next(i for i in a["ProcessingInputs"] if i["InputName"] == "code")
        assert code["S3Input"]["S3Uri"] == {"Get": "Parameters.CodeUri"}
        assert a["Environment"]["MLOPS_ASOF"] == {"Get": "Parameters.AsOf"}


def test_entrypoint_runs_the_step_through_the_metrics_wrapper():
    for name, s in _processing().items():
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


@pytest.mark.parametrize("step", ["Train", "Evaluate"])
def test_train_and_evaluate_read_the_communities_frozen_with_the_snapshot(step):
    a = _steps()[step]["Arguments"]
    inputs = {i["InputName"]: i["S3Input"] for i in a["ProcessingInputs"]}
    assert inputs["communities"]["S3Uri"] == {
        "Std:Join": {
            "On": "",
            "Values": ["s3://b/raw/communities/asof=", {"Get": "Parameters.AsOf"}, "/"],
        }
    }
    args = a["AppSpecification"]["ContainerArguments"]
    assert args[args.index("--communities") + 1] == inputs["communities"]["LocalPath"]


# --- the promotion gate (Stage 4) -----------------------------------------------------


def _gate():
    return _steps()["Gate"]


def test_gate_reads_both_reports_evaluate_declares():
    files = {f["PropertyFileName"]: f for f in _steps()["Evaluate"]["PropertyFiles"]}
    assert files["EvaluationReport"]["FilePath"] == "evaluation.json"
    assert files["FairnessReport"]["FilePath"] == "fairness.json"
    # a property file must name an output the step really has
    (out,) = _steps()["Evaluate"]["Arguments"]["ProcessingOutputConfig"]["Outputs"]
    assert {f["OutputName"] for f in files.values()} == {out["OutputName"]}

    conds = _gate()["Arguments"]["Conditions"]
    assert len(conds) == 4  # three floors, then champion / challenger
    for c, (key, floor) in zip(conds[:3], d.GATE.items(), strict=True):
        assert c["Type"] == "GreaterThanOrEqualTo"
        assert c["RightValue"] == floor
        get = c["LeftValue"]["Std:JsonGet"]
        file, path = d.GATE_PATHS[key]
        assert get["PropertyFile"] == {"Get": f"Steps.Evaluate.PropertyFiles.{file}"}
        assert file in files
        assert get["Path"] == path


def test_gate_paths_exist_in_the_reports_the_code_writes():
    """A typo'd path is the classic way a gate silently stops gating."""
    from src.pipeline.evaluate import build_report
    from src.promote.fairness import fairness_report
    from tests.test_fairness import _communities, _predictions

    reports = {
        "EvaluationReport": build_report(
            {"roc_auc": 0.66, "pr_auc": 0.33, "top_decile_lift": 2.0, "n_test": 10}, None
        ),
        "FairnessReport": fairness_report(_predictions(), _communities()),
    }
    for file, path in d.GATE_PATHS.values():
        node = reports[file]
        for part in path.split("."):
            node = node[part]
        assert isinstance(node, float)


def test_champion_check_passes_with_no_champion_or_a_real_win():
    (check,) = [c for c in _gate()["Arguments"]["Conditions"] if c["Type"] == "Or"]
    no_champion, wins = check["Arguments"]["Conditions"]
    assert no_champion["Type"] == "Equals" and no_champion["RightValue"] == 0
    assert no_champion["LeftValue"]["Std:JsonGet"]["Path"] == "champion.exists"
    assert wins["Type"] == "GreaterThanOrEqualTo" and wins["RightValue"] == 0.005
    assert wins["LeftValue"]["Std:JsonGet"]["Path"] == "champion.margin.value"
    args = _steps()["Evaluate"]["Arguments"]["AppSpecification"]["ContainerArguments"]
    assert args[args.index("--champion-group") + 1] == "mlops-aiops-breach-risk"


def test_champion_paths_exist_as_numbers_with_and_without_a_champion():
    from src.promote import champion

    for report in (
        champion.comparison(0.37, None, None),
        champion.comparison(0.37, {"ModelPackageArn": "arn:mp/1"}, 0.36),
    ):
        node = {"champion": report}
        for _, path in d.CHAMPION_PATHS.values():
            value = node
            for part in path.split("."):
                value = value[part]
            assert isinstance(value, int | float)


def test_gate_thresholds_are_code_not_run_parameters():
    params = {p["Name"] for p in _definition()["Parameters"]}
    assert not params & {"MinPrAuc", "MinRecallRatio", "MinGroupLift"}
    assert d.GATE == {"pr_auc": 0.25, "recall_ratio": 0.8, "min_group_lift": 1.5}
    assert all(isinstance(c["RightValue"], float) for c in _gate()["Arguments"]["Conditions"][:3])


def test_pass_registers_for_manual_approval_and_fail_registers_nothing():
    a = _gate()["Arguments"]
    (register,) = a["IfSteps"]
    (rejected,) = a["ElseSteps"]
    assert register["Type"] == "RegisterModel"
    assert rejected["Type"] == "Fail"
    r = register["Arguments"]
    assert r["ModelPackageGroupName"] == "mlops-aiops-breach-risk"
    assert r["ModelApprovalStatus"] == "PendingManualApproval"
    (container,) = r["InferenceSpecification"]["Containers"]
    assert container["ModelDataUrl"]["Std:Join"]["Values"] == [
        {"Get": "Steps.Train.ProcessingOutputConfig.Outputs['model'].S3Output.S3Uri"},
        "model.tar.gz",
    ]
    bias = r["ModelMetrics"]["Bias"]["Report"]["S3Uri"]["Std:Join"]["Values"]
    assert bias[-1] == "fairness.json"
    meta = r["CustomerMetadataProperties"]
    assert meta["gate"] == (
        "pr_auc>=0.25;recall_ratio>=0.8;min_group_lift>=1.5;beats_champion_by>=0.005"
    )
    assert meta["asof"] == {"Get": "Parameters.AsOf"}
    # the failure reason carries the numbers that failed
    values = rejected["Arguments"]["ErrorMessage"]["Std:Join"]["Values"]
    assert sum("Std:JsonGet" in v for v in values if isinstance(v, dict)) == 4


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


def _gate_steps(outcome: str) -> list[dict]:
    steps = [
        {
            "StepName": "Gate",
            "StepStatus": "Succeeded",
            "Metadata": {"Condition": {"Outcome": outcome}},
        }
    ]
    if outcome == "True":
        arn = "arn:aws:sagemaker:r:1:model-package/mlops-aiops-breach-risk/1"
        steps.append(
            {
                "StepName": "Register",
                "StepStatus": "Succeeded",
                "Metadata": {"RegisterModel": {"Arn": arn}},
            }
        )
    else:
        steps.append(
            {
                "StepName": "Rejected",
                "StepStatus": "Failed",
                "Metadata": {"Fail": {"ErrorMessage": "recall_ratio 0.55"}},
            }
        )
    return steps


def test_verdict_tells_a_rejection_from_a_broken_run():
    passed = launcher.step_report(None, _gate_steps("True"), "mlops-aiops", "ml.t3.xlarge")
    assert passed[1]["model_package"].endswith("breach-risk/1")
    assert launcher.verdict("Succeeded", passed).startswith("PASSED the gate; registered")

    rejected = launcher.step_report(None, _gate_steps("False"), "mlops-aiops", "ml.t3.xlarge")
    assert rejected[1]["reason"] == "recall_ratio 0.55"
    assert launcher.verdict("Failed", rejected) == "REJECTED by the gate; nothing was registered"

    assert launcher.verdict("Failed", []) == "run Failed before the gate decided"


def test_fairness_summary_marks_groups_that_are_not_gated():
    from src.promote.fairness import fairness_report, summary_lines
    from tests.test_fairness import _communities, _predictions

    lines = summary_lines(fairness_report(_predictions(), _communities()))
    assert lines[0].startswith("fairness PASS")
    assert any("NORTHWEST" in line for line in lines)
