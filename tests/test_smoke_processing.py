"""Processing smoke test (D-039): the request builder carries the cost controls."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_processing.py"
_spec = importlib.util.spec_from_file_location("smoke_processing", _PATH)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)

ROLE = "arn:aws:iam::123456789012:role/mlops-aiops-sagemaker"
CODE = "s3://mlops-aiops-datalake-x/code/smoke/"
IMAGE = "example.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost:1.7-1"


def test_request_is_one_small_tagged_instance_with_a_time_limit():
    req = smoke.processing_job_request("job", ROLE, IMAGE, "ml.t3.medium", 600, CODE)
    cluster = req["ProcessingResources"]["ClusterConfig"]
    assert cluster["InstanceCount"] == 1
    assert cluster["InstanceType"] == "ml.t3.medium"
    assert req["StoppingCondition"]["MaxRuntimeInSeconds"] == 600
    assert {"Key": "project", "Value": "mlops-aiops"} in req["Tags"]
    assert "ProcessingOutputConfig" not in req
    (code,) = req["ProcessingInputs"]
    assert code["S3Input"]["S3Uri"] == CODE


def test_entrypoint_arguments_fit_the_api_limit():
    # CreateProcessingJob caps each ContainerEntrypoint item at 256 characters; the first
    # version inlined the whole probe and was rejected (2026-09-28).
    req = smoke.processing_job_request("job", ROLE, IMAGE, "ml.t3.medium", 600, CODE)
    entry = req["AppSpecification"]["ContainerEntrypoint"]
    assert all(len(arg) <= 256 and "\n" not in arg for arg in entry)
    assert entry[1].endswith(smoke.PROBE_FILE.name) and smoke.PROBE_FILE.exists()


@pytest.mark.parametrize("itype", ["ml.m5.large", "ml.g5.xlarge"])
def test_request_rejects_instance_types_without_quota(itype):
    with pytest.raises(ValueError):
        smoke.processing_job_request("job", ROLE, IMAGE, itype, 600, CODE)


@pytest.mark.parametrize("seconds", [30, 3600])
def test_request_rejects_runtime_outside_smoke_range(seconds):
    with pytest.raises(ValueError):
        smoke.processing_job_request("job", ROLE, IMAGE, "ml.t3.medium", seconds, CODE)


def test_job_name_is_valid_and_prefixed():
    name = smoke.job_name("mlops-aiops", datetime(2026, 9, 28, 17, 5, 9, tzinfo=UTC))
    assert name == "mlops-aiops-smoke-20260928-170509"
    assert len(name) <= 63


def test_summarize_reports_billed_seconds_and_failure():
    start = datetime(2026, 9, 28, tzinfo=UTC)
    desc = {
        "ProcessingJobStatus": "Failed",
        "FailureReason": "ResourceLimitExceeded",
        "ProcessingStartTime": start,
        "ProcessingEndTime": start + timedelta(seconds=74),
    }
    assert smoke.summarize(desc) == {
        "status": "Failed",
        "failure_reason": "ResourceLimitExceeded",
        "billed_seconds": 74,
    }
    assert smoke.summarize({"ProcessingJobStatus": "InProgress"}) == {"status": "InProgress"}
