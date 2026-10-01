"""Pipeline-step smoke tests — Stage 3 (D-016).

These run the step code against a tiny local fixture with no AWS calls, to catch path,
shape and I/O-contract breaks before a pipeline run that costs money. The end-to-end
test needs xgboost and scikit-learn (the `ml` extra) and skips without them.
"""

from __future__ import annotations

import json
import sys
import types

import pandas as pd
import pytest

from src.features.build_features import align_to_schema
from src.pipeline import evaluate, step


def test_train_then_evaluate_reproduces_training_metrics(tmp_path, requests_frame):
    """The artifact alone gives the numbers training recorded: nothing it needs is missing."""
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    from src.pipeline import train

    data, model, out = tmp_path / "data", tmp_path / "model", tmp_path / "eval"
    comms = tmp_path / "communities"
    data.mkdir()
    comms.mkdir()
    requests_frame.to_parquet(data / "part-0.parquet")
    (comms / evaluate.COMMUNITIES_FILE).write_text(
        json.dumps([{"comm_code": "X", "sector": "CENTRE", "srg": "ESTABLISHED"}])
    )
    assert train.main(["--data", str(data), "--communities", str(comms), "--out", str(model)]) == 0
    args = ["--model", str(model), "--data", str(data), "--communities", str(comms)]
    assert evaluate.main([*args, "--out", str(out)]) == 0

    report = json.loads((out / evaluate.REPORT_NAME).read_text())
    assert report["matches_training"] is True
    # the paths the Stage 4 gate reads
    assert 0 <= report["binary_classification_metrics"]["pr_auc"]["value"] <= 1
    assert report["n_test"] > 0
    fair = json.loads((out / "fairness.json").read_text())
    # one community is one measurable group at most: the check can't vouch, so it fails
    assert fair["passed"] is False
    assert fair["dimensions"]["sector"]["groups"][0]["group"] == "CENTRE"


def test_report_flags_an_artifact_that_disagrees_with_training():
    metrics = {"roc_auc": 0.66, "pr_auc": 0.33, "top_decile_lift": 2.04, "n_test": 10}
    same = evaluate.build_report(
        metrics, {"roc_auc": 0.66, "pr_auc": 0.33, "top_decile_lift": 2.04}
    )
    assert same["matches_training"] is True
    off = evaluate.build_report(metrics, {"roc_auc": 0.70, "pr_auc": 0.33, "top_decile_lift": 2.04})
    assert off["matches_training"] is False
    assert off["difference_from_training"]["roc_auc"] == pytest.approx(0.04)
    assert "matches_training" not in evaluate.build_report(metrics, None)


def test_align_to_schema_orders_columns_and_uses_training_levels():
    schema = {
        "columns": ["req_month", "service_name"],
        "categories": {"service_name": ["Pothole", "Tree"]},
    }
    features = pd.DataFrame(
        {"service_name": pd.Categorical(["Tree", "Graffiti"]), "req_month": [1, 2], "extra": [0, 0]}
    )
    out = align_to_schema(features, schema)
    assert list(out.columns) == ["req_month", "service_name"]
    assert list(out["service_name"].cat.categories) == ["Pothole", "Tree"]
    assert out["service_name"].cat.codes.tolist() == [1, -1]  # an unseen level is missing


def _fake_step(monkeypatch, body):
    mod = types.ModuleType("fake_step")
    mod.main = body
    monkeypatch.setitem(sys.modules, "fake_step", mod)
    monkeypatch.setitem(step.STEPS, "fake", "fake_step")
    from src.common import metrics

    monkeypatch.setattr(metrics, "_DISABLED", True)  # log the metrics, don't send them


def test_step_runner_passes_arguments_and_exits_zero(monkeypatch):
    seen = []
    _fake_step(monkeypatch, lambda argv: seen.append(argv))
    assert step.main(["fake", "--x", "1"]) == 0
    assert seen == [["--x", "1"]]


def test_step_runner_turns_a_failure_into_a_nonzero_exit(monkeypatch):
    def boom(argv):
        raise RuntimeError("bad data")

    _fake_step(monkeypatch, boom)
    assert step.main(["fake"]) == 1


def test_step_runner_still_succeeds_when_metrics_cannot_be_sent(monkeypatch):
    from src.common import metrics

    _fake_step(monkeypatch, lambda argv: None)

    def denied(*a, **kw):
        raise PermissionError("AccessDenied: cloudwatch:PutMetricData")

    monkeypatch.setattr(metrics, "emit_run_metric", denied)
    assert step.main(["fake"]) == 0


def test_step_runner_knows_the_three_steps():
    assert set(step.STEPS) >= {"validate", "train", "evaluate"}
    with pytest.raises(SystemExit):
        step.main(["nope"])
