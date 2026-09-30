"""Stage 4 approval tool: the checks it makes before a decision. No AWS calls."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.promote.fairness import fairness_report
from tests.test_fairness import _communities, _predictions

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "approve.py"
_spec = importlib.util.spec_from_file_location("approve", _PATH)
approve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(approve)

GROUP = "mlops-aiops-breach-risk"


class _Pages:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **kw):
        assert kw["ModelPackageGroupName"] == GROUP
        return self.pages


class _FakeSM:
    def __init__(self, versions):
        self.versions = versions

    def get_paginator(self, name):
        assert name == "list_model_packages"
        summaries = [
            {"ModelPackageVersion": v, "ModelPackageArn": f"arn:mp/{GROUP}/{v}"}
            for v in self.versions
        ]
        return _Pages([{"ModelPackageSummaryList": summaries}])


def test_package_arn_resolves_a_version_number():
    sm = _FakeSM([1, 2, 3])
    assert approve.package_arn(sm, GROUP, "2") == f"arn:mp/{GROUP}/2"
    assert approve.package_arn(sm, GROUP, "arn:given") == "arn:given"
    with pytest.raises(SystemExit):
        approve.package_arn(sm, GROUP, "9")


def test_reports_finds_what_the_register_step_attached():
    desc = {
        "ModelMetrics": {
            "ModelQuality": {"Statistics": {"S3Uri": "s3://b/e/evaluation.json"}},
            "Bias": {"Report": {"S3Uri": "s3://b/e/fairness.json"}},
        }
    }
    assert approve.reports(desc) == ("s3://b/e/evaluation.json", "s3://b/e/fairness.json")
    assert approve.reports({}) == (None, None)


def test_approve_refuses_a_failed_or_missing_fairness_report():
    good = fairness_report(_predictions(), _communities())
    bad = fairness_report(_predictions(blind_sector="NORTHWEST"), _communities())
    assert approve.approval_blockers(good) == []
    (reason,) = approve.approval_blockers(bad)
    assert "NORTHWEST" in reason
    assert approve.approval_blockers(None) == ["no fairness report attached"]


class _FakeIAM:
    def __init__(self, decisions):
        self.decisions = decisions

    def simulate_principal_policy(self, **kw):
        assert kw["ActionNames"] == ["sagemaker:UpdateModelPackage"]
        return {"EvaluationResults": [{"EvalDecision": self.decisions[kw["PolicySourceArn"]]}]}


def test_simulate_reports_iams_decision():
    iam = _FakeIAM({"arn:train": "explicitDeny", "arn:approver": "allowed"})
    assert approve.simulate(iam, "arn:train", "arn:mp") == "explicitDeny"
    assert approve.simulate(iam, "arn:approver", "arn:mp") == "allowed"
