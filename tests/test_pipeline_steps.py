"""Pipeline-step smoke tests — Stage 3 (D-016).

These run the Processing-step *entrypoints* (`preprocess`, `evaluate`) against a tiny
local fixture with **no AWS calls and no SageMaker**, to catch path / shape / I/O-contract
breaks before a $-costing pipeline run. They are skipped until Stage 3 lands the
entrypoints; the import guard keeps CI green in the meantime.

Run: `pytest tests/test_pipeline_steps.py`
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_PREPROCESS = importlib.util.find_spec("src.pipeline.preprocess") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_PREPROCESS, reason="Stage 3 pipeline entrypoints not implemented yet"
)


def test_preprocess_writes_train_and_test(tmp_path, requests_frame):
    """preprocess.main() reads raw Parquet from an input dir and writes
    train.csv / test.csv to an output dir, with the label column present and no
    leaky columns."""
    from src.features.build_labels import LEAKY_COLUMNS
    from src.pipeline import preprocess

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    out_dir.mkdir()
    requests_frame.to_parquet(in_dir / "part-0.parquet")

    preprocess.main(["--input", str(in_dir), "--output", str(out_dir)])

    import pandas as pd

    train = pd.read_csv(out_dir / "train.csv")
    test = pd.read_csv(out_dir / "test.csv")
    assert "breach" in train.columns
    assert len(train) and len(test)
    assert not (set(train.columns) & (set(LEAKY_COLUMNS) - {"breach"}))


def test_evaluate_emits_metric_json(tmp_path):
    """evaluate.main() writes evaluation.json with the metric path the Stage 4
    ConditionStep reads (`binary_classification_metrics.pr_auc.value`)."""
    import json

    from src.pipeline import evaluate

    # a trivially-perfect prediction file so the numbers are deterministic
    (tmp_path / "test").mkdir()
    (tmp_path / "model").mkdir()
    (tmp_path / "out").mkdir()
    import pandas as pd

    pd.DataFrame({"breach": [0, 1, 0, 1], "pred": [0.1, 0.9, 0.2, 0.8]}).to_csv(
        tmp_path / "test" / "predictions.csv", index=False
    )

    evaluate.main(["--predictions", str(tmp_path / "test"), "--output", str(tmp_path / "out")])

    report = json.loads((tmp_path / "out" / "evaluation.json").read_text())
    assert report["binary_classification_metrics"]["pr_auc"]["value"] > 0.5
