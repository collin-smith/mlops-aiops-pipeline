"""Stage 2 training step and its SageMaker launcher (D-039). No AWS calls.

The end-to-end test needs xgboost and scikit-learn (the `ml` extra) and skips without them.
"""

from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import sm_jobs
from src.pipeline import train

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_job.py"
_spec = importlib.util.spec_from_file_location("train_job", _PATH)
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)

ROLE = "arn:aws:iam::123456789012:role/mlops-aiops-sagemaker"
IMAGE = sm_jobs.xgboost_image("ca-central-1", "3.2-0")


def _request(**kw):
    args = dict(
        name="job",
        role_arn=ROLE,
        image_uri=IMAGE,
        instance_type="ml.t3.xlarge",
        max_runtime=3600,
        code_uri="s3://b/code/job/",
        data_uri="s3://b/processed/311/",
        output_uri="s3://b/model-artifacts/job/",
    )
    args.update(kw)
    return launcher.processing_job_request(**args)


def test_top_decile_metrics_match_hand_computation():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1] * 10)
    p = np.where(np.arange(100) % 10 == 9, 0.9, 0.1)  # flags exactly the positives at index 9
    p[np.arange(100) % 10 == 0] = 0.05
    m = train.top_decile_metrics(y, p)
    assert m["base_rate"] == 0.2
    assert m["top_decile_rate"] == 1.0
    assert m["top_decile_lift"] == 5.0
    assert m["top_decile_recall"] == 0.5


def test_load_requests_drops_hive_partition_columns(tmp_path, requests_frame):
    df = requests_frame.assign(year=2024, month=1)
    df.to_parquet(tmp_path, partition_cols=["year", "month"])
    loaded = train.load_requests(tmp_path)
    assert len(loaded) == len(requests_frame)
    assert not {"year", "month"} & set(loaded.columns)


class _FakeBooster:
    def save_model(self, path):
        Path(path).write_text("{}")


def test_write_artifacts_packs_model_schema_and_thresholds(tmp_path):
    schema = {"columns": ["a"], "categories": {"service_name": ["x"]}}
    thresholds = pd.Series({"Pothole": 3.0}, name="threshold_days")
    tar_path = train.write_artifacts(_FakeBooster(), schema, thresholds, {"roc_auc": 0.6}, tmp_path)
    with tarfile.open(tar_path) as tar:
        assert sorted(tar.getnames()) == ["feature_schema.json", "thresholds.csv", "xgboost-model"]
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"roc_auc": 0.6}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.json", "model.tar.gz"]


def test_train_end_to_end_on_the_fixture(tmp_path, requests_frame):
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    data = tmp_path / "data"
    data.mkdir()
    requests_frame.to_parquet(data / "part-0.parquet")
    assert train.main(["--data", str(data), "--out", str(tmp_path / "out")]) == 0
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert {"roc_auc", "pr_auc", "top_decile_lift", "n_train", "n_test"} <= set(metrics)


def test_launcher_request_mounts_code_and_data_and_writes_model_artifacts():
    req = _request()
    inputs = {i["InputName"]: i["S3Input"] for i in req["ProcessingInputs"]}
    assert inputs["code"]["LocalPath"] == launcher.IN_CODE
    assert inputs["data"]["S3Uri"] == "s3://b/processed/311/"
    (out,) = req["ProcessingOutputConfig"]["Outputs"]
    assert out["S3Output"]["S3Uri"] == "s3://b/model-artifacts/job/"
    assert req["ProcessingResources"]["ClusterConfig"]["InstanceCount"] == 1
    assert req["StoppingCondition"]["MaxRuntimeInSeconds"] == 3600
    assert {"Key": "project", "Value": "mlops-aiops"} in req["Tags"]


@pytest.mark.parametrize("rows", [0, 1_000_000])
def test_launcher_entrypoint_fits_the_api_limit(rows):
    entry = _request(max_train_rows=rows)["AppSpecification"]["ContainerEntrypoint"]
    assert all(len(arg) <= 256 for arg in entry)
    assert ("--max-train-rows" in entry[-1]) == bool(rows)


def test_launcher_rejects_no_quota_instance_and_silly_runtime():
    with pytest.raises(ValueError):
        _request(instance_type="ml.m5.large")
    with pytest.raises(ValueError):
        _request(max_runtime=60)


def test_launcher_uploads_src_and_requirements_without_caches():
    files = launcher.code_files()
    assert "src/pipeline/train.py" in files
    assert launcher.REQUIREMENTS in files and files[launcher.REQUIREMENTS].exists()
    assert not any("__pycache__" in k for k in files)


def test_xgboost_image_is_pinned_per_region():
    assert IMAGE == "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost:3.2-0"
    with pytest.raises(ValueError):
        sm_jobs.xgboost_image("ca-central-1", "9.9-9")


class _FakeLogs:
    """First page empty with a token (as FilterLogEvents really does), then the lines."""

    class exceptions:  # noqa: N801 — mirrors boto3's client.exceptions
        class ResourceNotFoundException(Exception):
            pass

    def get_paginator(self, op):
        assert op == "filter_log_events"
        return self

    def paginate(self, **kw):
        assert kw["logStreamNamePrefix"] == "job"
        return iter([{"events": [], "nextToken": "t"}, {"events": [{"message": "SMOKE OK\n"}]}])


def test_print_job_log_reads_past_an_empty_first_page(capsys):
    sm_jobs.print_job_log(_FakeLogs(), "job")
    assert "SMOKE OK" in capsys.readouterr().out
