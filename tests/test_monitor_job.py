"""Stage 7 Layer A launcher (D-045): one-off analyzer jobs with the cost controls, no metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "monitor_job.py"
_spec = importlib.util.spec_from_file_location("monitor_job", _PATH)
monitor_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor_job)

ROLE = "arn:aws:iam::123456789012:role/mlops-aiops-sagemaker"
BUCKET = "s3://mlops-aiops-datalake-x"


def _request(**kw):
    args = {
        "name": "job",
        "role_arn": ROLE,
        "image_uri": monitor_job.analyzer_image("ca-central-1"),
        "instance_type": "ml.t3.xlarge",
        "max_runtime": 1200,
        "dataset_uri": f"{BUCKET}/monitoring/datasets/x/",
        "output_uri": f"{BUCKET}/monitoring/checks/x/",
    }
    return monitor_job.analyzer_request(**{**args, **kw})


@pytest.mark.parametrize("baseline", [None, f"{BUCKET}/monitoring/baseline/baseline/"])
def test_every_request_switches_off_cloudwatch_metrics(baseline):
    env = _request(baseline_uri=baseline)["Environment"]
    assert env["publish_cloudwatch_metrics"] == "Disabled"
    assert json.loads(env["dataset_format"])["csv"]["header"] is True


def test_request_is_one_small_tagged_instance_with_a_time_limit():
    req = _request()
    cluster = req["ProcessingResources"]["ClusterConfig"]
    assert cluster["InstanceCount"] == 1
    assert cluster["InstanceType"] == "ml.t3.xlarge"
    assert req["StoppingCondition"]["MaxRuntimeInSeconds"] == 1200
    assert {"Key": "project", "Value": "mlops-aiops"} in req["Tags"]
    assert req["AppSpecification"]["ImageUri"].endswith("/sagemaker-model-monitor-analyzer")


def test_baseline_run_has_no_baseline_input_and_a_check_does():
    suggest = _request()
    assert [i["InputName"] for i in suggest["ProcessingInputs"]] == ["dataset"]
    assert "baseline_constraints" not in suggest["Environment"]
    check = _request(baseline_uri=f"{BUCKET}/monitoring/baseline/baseline/")
    assert [i["InputName"] for i in check["ProcessingInputs"]] == ["dataset", "baseline"]
    env = check["Environment"]
    assert env["baseline_constraints"].startswith(monitor_job.IN_BASELINE)
    assert env["baseline_statistics"].startswith(monitor_job.IN_BASELINE)


def test_refuses_instances_without_quota_and_unbounded_runs():
    with pytest.raises(ValueError, match="quota"):
        _request(instance_type="ml.m5.xlarge")
    with pytest.raises(ValueError, match="max_runtime"):
        _request(max_runtime=7200)


def test_violation_lines():
    report = {"violations": [{"feature_name": "service_name", "constraint_check_type": "x"}]}
    assert monitor_job.violation_lines(report) == ["service_name: x"]
    assert monitor_job.violation_lines({}) == ["no violations"]


def test_no_schedule_is_ever_created():
    assert "create_monitoring_schedule" not in _PATH.read_text()
