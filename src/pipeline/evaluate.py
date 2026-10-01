"""Stage 3 evaluate step: score the test year with the saved model artifact.

The train step already reports test metrics, so this step's job is to prove the artifact
stands on its own. It unpacks ``model.tar.gz``, rebuilds the test year from the raw data,
shapes the features with ``feature_schema.json`` (as batch scoring will in Stage 6), and
scores them with the saved booster. If its numbers don't match the ones training recorded,
something the model depends on didn't make it into the artifact, and the step fails.

    python -m src.pipeline.evaluate --model <dir with model.tar.gz> --data <parquet dir> \
        --communities <dir with communities.json> --out <dir>

It writes two reports, which the Stage 4 gate reads:

* ``evaluation.json``, in the layout SageMaker's model-quality reports use
  (``binary_classification_metrics.<name>.value``)
* ``fairness.json``, the per-sector check (D-035, ``src/promote/fairness.py``) on the
  same test-year scores

With ``--champion-group`` it also scores the latest approved model on the same rows and
adds the comparison to ``evaluation.json`` as ``champion`` (D-043, ``src/promote/champion.py``).

A failed fairness check doesn't fail this step. The gate reads the report and decides,
the same way it decides on PR-AUC, so a rejection is a gate outcome and not a crash.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import align_to_schema
from src.features.build_labels import assert_no_leakage, build_training_labels
from src.features.history import sector_lookup
from src.pipeline.train import evaluate, featurize, load_requests
from src.promote import champion, fairness

REPORT_NAME = "evaluation.json"
COMMUNITIES_FILE = "communities.json"
# metrics the artifact must reproduce from training's metrics.json
CHECKED = ("roc_auc", "pr_auc", "top_decile_lift")
TOLERANCE = 1e-3


class ArtifactMismatchError(RuntimeError):
    """The saved model doesn't reproduce the metrics recorded at training time."""


def load_artifact(model_dir: Path):
    """The booster, feature schema and label thresholds from ``model.tar.gz``."""
    import xgboost as xgb

    with tempfile.TemporaryDirectory() as tmp, tarfile.open(model_dir / "model.tar.gz") as tar:
        tar.extractall(tmp, filter="data")
        booster = xgb.Booster()
        booster.load_model(str(Path(tmp) / "xgboost-model"))
        schema = json.loads((Path(tmp) / "feature_schema.json").read_text())
        thresholds = pd.read_csv(Path(tmp) / "thresholds.csv", index_col=0).iloc[:, 0]
    return booster, schema, thresholds


def predict(booster, features: pd.DataFrame):
    import xgboost as xgb

    return booster.predict(xgb.DMatrix(features, enable_categorical=True))


def score_model(model_dir: Path, test: pd.DataFrame, raw: pd.DataFrame, sectors) -> np.ndarray:
    """A saved model's scores for ``test``, with the features its own schema asks for and
    its own label thresholds (a Stage 2/3 artifact has no feature_set: it's the baseline)."""
    booster, schema, thresholds = load_artifact(model_dir)
    feature_set = schema.get("feature_set", "baseline")
    X = align_to_schema(featurize(test, raw, thresholds, sectors, feature_set), schema)
    assert_no_leakage(X)
    return predict(booster, X)


def champion_comparison(group: str | None, pr_auc: float, test, raw, sectors) -> dict:
    """Against the approved champion, on the same rows (D-043). No group, no lookup."""
    if not group:
        return champion.comparison(pr_auc, None, None)
    import boto3
    from sklearn.metrics import average_precision_score

    y = test["breach"].to_numpy()

    def champion_pr_auc(model_dir: Path) -> float:
        return round(
            float(average_precision_score(y, score_model(model_dir, test, raw, sectors))), 4
        )

    return champion.compare(
        boto3.client("sagemaker"), boto3.client("s3"), group, pr_auc, champion_pr_auc
    )


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
    ap.add_argument(
        "--communities", type=Path, required=True, help=f"directory holding {COMMUNITIES_FILE}"
    )
    ap.add_argument(
        "--out", type=Path, required=True, help=f"where {REPORT_NAME} and fairness.json go"
    )
    ap.add_argument(
        "--champion-group",
        help="Model Package Group to find the approved champion in (omit to skip, e.g. locally)",
    )
    args = ap.parse_args(argv)

    communities = fairness.load_communities(args.communities / COMMUNITIES_FILE)
    raw = load_requests(args.data)
    _, test, _ = build_training_labels(raw)
    sectors = sector_lookup(communities)
    scores = score_model(args.model, test, raw, sectors)
    metrics = evaluate(test["breach"].to_numpy(), scores)
    fair = fairness.fairness_report(test[["breach", "comm_code"]].assign(pred=scores), communities)

    training_file = args.model / "metrics.json"
    training = json.loads(training_file.read_text()) if training_file.exists() else None
    report = build_report(metrics, training)
    report["champion"] = champion_comparison(
        args.champion_group, metrics["pr_auc"], test, raw, sectors
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_NAME).write_text(json.dumps(report, indent=2))
    # strict JSON: the gate's JsonGet can't parse NaN
    (args.out / fairness.REPORT_NAME).write_text(json.dumps(fair, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)
    print(
        json.dumps(
            {k: fair[k] for k in ("passed", "recall_ratio", "min_group_lift")}
            | {d: v["problems"] for d, v in fair["dimensions"].items()}
        ),
        flush=True,
    )

    if report.get("matches_training") is False:
        diffs = report["difference_from_training"]
        raise ArtifactMismatchError(f"the artifact doesn't reproduce training's metrics: {diffs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
