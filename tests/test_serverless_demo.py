"""Stage 6 serverless demo — the parts that must hold without touching AWS (D-029).

The request builder is the cost control: one Serverless variant, never provisioned
concurrency. The rest checks the payload guard and the latency summary.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serverless_demo.py"
_spec = importlib.util.spec_from_file_location("serverless_demo", _PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def test_endpoint_config_is_serverless_only():
    req = demo.endpoint_config_request("n", "m", "mlops-aiops")
    (variant,) = req["ProductionVariants"]
    assert variant["ServerlessConfig"] == {"MemorySizeInMB": 2048, "MaxConcurrency": 1}
    assert "ProvisionedConcurrency" not in variant["ServerlessConfig"]
    assert "InstanceType" not in variant  # an instance type would make it a billed-24/7 endpoint
    assert {"Key": "project", "Value": "mlops-aiops"} in req["Tags"]  # Budget filter keys on this


@pytest.mark.parametrize("mb", [512, 1500, 8192])
def test_endpoint_config_rejects_invalid_memory(mb):
    with pytest.raises(ValueError):
        demo.endpoint_config_request("n", "m", "p", memory_mb=mb)


def test_read_payload_rows_limits_and_rejects_header(tmp_path):
    good = tmp_path / "rows.csv"
    good.write_text("1,0.5,1e-3\n2,0.1,4\n3,0.9,5\n")
    assert demo.read_payload_rows(good, 2) == ["1,0.5,1e-3", "2,0.1,4"]

    header = tmp_path / "header.csv"
    header.write_text("service_name,hour\n1,2\n")
    with pytest.raises(SystemExit):
        demo.read_payload_rows(header, 5)


def test_summarize_latencies_splits_cold_from_warm():
    s = demo.summarize_latencies([4200.0, 80.0, 60.0, 70.0])
    assert s == {"cold_start_ms": 4200.0, "warm_median_ms": 70.0, "calls": 4}


def test_resource_name_is_predictable_for_cleanup():
    assert demo.resource_name("mlops-aiops") == "mlops-aiops-serverless-demo"


class _FakeIAM:
    def __init__(self, names):
        self.names = names

    def list_attached_role_policies(self, RoleName):  # noqa: N803 — boto3 kwarg name
        assert RoleName == "mlops-aiops-sagemaker"
        return {"AttachedPolicies": [{"PolicyName": n} for n in self.names]}


def test_budget_hardstop_detected_from_attached_policy():
    arn = "arn:aws:iam::123456789012:role/mlops-aiops-sagemaker"
    assert demo.budget_hardstop_active(
        _FakeIAM(["mlops-aiops-budget-hardstop-deny"]), arn, "mlops-aiops"
    )
    assert not demo.budget_hardstop_active(_FakeIAM(["something-else"]), arn, "mlops-aiops")


class _FakeSM:
    """delete_endpoint fails with the given message; the other deletes succeed."""

    def __init__(self, endpoint_error):
        self.endpoint_error = endpoint_error
        self.deleted = []

    def delete_endpoint(self, EndpointName):  # noqa: N803
        err = {"Error": {"Code": "ValidationException", "Message": self.endpoint_error}}
        raise ClientError(err, "DeleteEndpoint")

    def delete_endpoint_config(self, EndpointConfigName):  # noqa: N803
        self.deleted.append("config")

    def delete_model(self, ModelName):  # noqa: N803
        self.deleted.append("model")


def test_cleanup_reports_already_gone(capsys):
    sm = _FakeSM("Could not find endpoint x")
    demo.cleanup(sm, "x")
    assert "not found (already deleted)" in capsys.readouterr().out
    assert sm.deleted == ["config", "model"]


def test_cleanup_flags_a_stuck_endpoint_and_still_deletes_the_rest(capsys):
    sm = _FakeSM("Cannot update in-progress endpoint")
    demo.cleanup(sm, "x")
    out = capsys.readouterr().out
    assert "!! could not delete endpoint" in out and "--cleanup-only" in out
    assert sm.deleted == ["config", "model"]
