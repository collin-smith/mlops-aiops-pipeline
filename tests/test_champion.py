"""Stage 5 champion / challenger: who the champion is, and what counts as a win. No AWS."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.promote import champion


class _FakeSM:
    def __init__(self, approved):
        self.approved, self.asked = approved, None

    def list_model_packages(self, **kw):
        self.asked = kw
        return {"ModelPackageSummaryList": [{"ModelPackageArn": a} for a in self.approved[:1]]}

    def describe_model_package(self, ModelPackageName):  # noqa: N803 — boto3's name
        url = f"s3://b/pipeline-runs/{ModelPackageName}/train/model.tar.gz"
        return {
            "ModelPackageArn": ModelPackageName,
            "InferenceSpecification": {"Containers": [{"ModelDataUrl": url}]},
        }


class _FakeS3:
    def __init__(self):
        self.downloaded = []

    def download_file(self, bucket, key, dest):
        self.downloaded.append((bucket, key))
        Path(dest).write_bytes(b"model")


def test_only_an_approved_version_can_be_champion():
    sm = _FakeSM(["arn:mp/3"])
    assert champion.latest_approved(sm, "g")["ModelPackageArn"] == "arn:mp/3"
    assert sm.asked["ModelApprovalStatus"] == "Approved"
    assert sm.asked["SortOrder"] == "Descending"
    assert champion.latest_approved(_FakeSM([]), "g") is None


def test_no_champion_means_nothing_to_beat():
    out = champion.compare(_FakeSM([]), _FakeS3(), "g", 0.37, score_with=lambda d: 1 / 0)
    assert out == {"exists": 0, "arn": None, "pr_auc": None, "margin": {"value": 0.0}}


def test_the_champion_is_downloaded_and_scored_on_the_same_rows():
    s3 = _FakeS3()
    out = champion.compare(_FakeSM(["arn:mp/2"]), s3, "g", 0.374, score_with=lambda d: 0.36)
    assert s3.downloaded == [("b", "pipeline-runs/arn:mp/2/train/model.tar.gz")]
    assert out["exists"] == 1 and out["pr_auc"] == 0.36
    assert out["margin"]["value"] == pytest.approx(0.014)


@pytest.mark.parametrize(("challenger", "wins"), [(0.365, True), (0.364, False), (0.36, False)])
def test_a_win_has_to_clear_the_guardband(challenger, wins):
    margin = champion.comparison(challenger, {"ModelPackageArn": "a"}, 0.36)["margin"]["value"]
    assert (margin >= champion.GUARDBAND) is wins


def test_a_registry_error_propagates():
    class Broken(_FakeSM):
        def list_model_packages(self, **kw):
            raise RuntimeError("AccessDenied")

    with pytest.raises(RuntimeError):
        champion.compare(Broken([]), _FakeS3(), "g", 0.37, score_with=lambda d: 0.3)
