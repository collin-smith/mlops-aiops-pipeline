"""Stage 3 evaluate step: score the test year with the saved model artifact.

The train step already reports test metrics, so this step's job is to prove the artifact
stands on its own. It unpacks ``model.tar.gz``, rebuilds the test year from the raw data,
shapes the features with ``feature_schema.json`` (as batch scoring will in Stage 6), and
scores them with the saved booster. If its numbers don't match the ones training recorded,
something the model depends on didn't make it into the artifact, and the step fails.

    python -m src.pipeline.evaluate --model <dir with model.tar.gz> --data <parquet dir> --out <dir>

It writes ``evaluation.json`` in the layout SageMaker's model-quality reports use
(``binary_classification_metrics.<name>.value``), which the Stage 4 condition step reads.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

from src.features.build_features import align_to_schema, build_features
from src.features.build_labels import assert_no_leakage, build_training_labels
from src.pipeline.train import evaluate, load_requests

REPORT_NAME = "evaluation.json"
# metrics the artifact must reproduce from training's metrics.json
CHECKED = ("roc_auc", "pr_auc", "top_decile_lift")
TOLERANCE = 1e-3


class ArtifactMismatchError(RuntimeError):
    """The saved model doesn't reproduce the metrics recorded at training time."""


def load_artifact(model_dir: Path):
    """The booster and feature schema from ``model.tar.gz``."""
    import xgboost as xgb

    with tempfile.TemporaryDirectory() as tmp, tarfile.open(model_dir / "model.tar.gz") as tar:
        tar.extractall(tmp, filter="data")
        booster = xgb.Booster()
        booster.load_model(str(Path(tmp) / "xgboost-model"))
        schema = json.loads((Path(tmp) / "feature_schema.json").read_text())
    return booster, schema


def predict(booster, features: pd.DataFrame):
    import xgboost as xgb

    return booster.predict(xgb.DMatrix(features, enable_categorical=True))


def build_report(metrics: dict, training: dict | None) -> dict:
    report = {
        "binary_classification_metrics": {
            k: {"value": v} for k, v in metrics.items() if k != "n_test"
        },
        "n_test": metrics["n_test"],
    }
    if training is not None:
        diffs = {k: round(abs(metrics[k] - training[k]), 6) for k in CHECKED if k in training}
        report["matches_training"] = all(d <= TOLERANCE for d in diffs.values())
        report["difference_from_training"] = diffs
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True, help="directory holding model.tar.gz")
    ap.add_argument("--data", type=Path, required=True, help="processed Parquet directory")
    ap.add_argument("--out", type=Path, required=True, help=f"where {REPORT_NAME} goes")
    args = ap.parse_args(argv)

    booster, schema = load_artifact(args.model)
    _, test, _ = build_training_labels(load_requests(args.data))
    X_test = align_to_schema(build_features(test, keep_key=False), schema)
    assert_no_leakage(X_test)
    metrics = evaluate(test["breach"].to_numpy(), predict(booster, X_test))

    training_file = args.model / "metrics.json"
    training = json.loads(training_file.read_text()) if training_file.exists() else None
    report = build_report(metrics, training)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_NAME).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

    if report.get("matches_training") is False:
        diffs = report["difference_from_training"]
        raise ArtifactMismatchError(f"the artifact doesn't reproduce training's metrics: {diffs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
