"""Stage 2 training step: labels, features, XGBoost, evaluation, and the model artifact.

The same model as ``notebooks/02_baseline_model.ipynb``, packaged to run inside a
SageMaker Processing job (D-039; launched by ``scripts/train_job.py``) or locally:

    python -m src.pipeline.train --data data/processed/311 --out /tmp/model

It writes to ``--out``:
  model.tar.gz   xgboost-model, feature_schema.json (column order and the category levels
                 scoring must align to), thresholds.csv (per-category breach thresholds)
  metrics.json   test-year ROC-AUC, PR-AUC, and the top-decile lift the series reports

Training and evaluation share one step for now. Stage 3 splits them into pipeline steps.
"""

from __future__ import annotations

import argparse
import json
import platform
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL, align_categories, build_features
from src.features.build_labels import assert_no_leakage, build_training_labels

# notebooks/02_baseline_model.ipynb, unchanged (D-037's 2.0x baseline).
PARAMS = {
    "n_estimators": 400,
    "max_depth": 7,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "max_cat_to_onehot": 1,
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": 42,
}
PARTITION_COLUMNS = ("year", "month")


def load_requests(data_dir: Path) -> pd.DataFrame:
    """Read the processed Parquet snapshot; the hive partition columns aren't data."""
    raw = pd.read_parquet(data_dir)
    return raw.drop(columns=[c for c in PARTITION_COLUMNS if c in raw.columns])


def top_decile_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """How the riskiest tenth of requests compares with all of them."""
    base = float(y.mean())
    top = p >= np.quantile(p, 0.9)
    top_rate = float(y[top].mean())
    return {
        "base_rate": round(base, 4),
        "top_decile_rate": round(top_rate, 4),
        "top_decile_lift": round(top_rate / base, 3) if base else float("nan"),
        "top_decile_recall": round(float(y[top].sum() / y.sum()), 4) if y.sum() else 0.0,
    }


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        **top_decile_metrics(y, p),
        "n_test": len(y),
    }


def feature_schema(features: pd.DataFrame) -> dict:
    return {
        "columns": list(features.columns),
        "categories": {c: [str(v) for v in features[c].cat.categories] for c in CATEGORICAL},
    }


def write_artifacts(
    booster, schema: dict, thresholds: pd.Series, metrics: dict, out_dir: Path
) -> Path:
    """model.tar.gz (what the registry points at) and metrics.json (what the gate reads)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "model.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        booster.save_model(str(staging / "xgboost-model"))
        (staging / "feature_schema.json").write_text(json.dumps(schema, indent=2))
        thresholds.to_csv(staging / "thresholds.csv")
        with tarfile.open(tar_path, "w:gz") as tar:
            for f in sorted(staging.iterdir()):
                tar.add(f, arcname=f.name)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return tar_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="processed Parquet directory")
    ap.add_argument("--out", type=Path, required=True, help="where the artifacts go")
    ap.add_argument(
        "--max-train-rows", type=int, default=0, help="random sample of the training rows (0 = all)"
    )
    args = ap.parse_args(argv)

    import sklearn
    import xgboost as xgb

    print(
        f"python {platform.python_version()} | pandas {pd.__version__} | "
        f"xgboost {xgb.__version__} | sklearn {sklearn.__version__}",
        flush=True,
    )
    t0 = time.time()
    raw = load_requests(args.data)
    train, test, thresholds = build_training_labels(raw)
    if args.max_train_rows and len(train) > args.max_train_rows:
        train = train.sample(n=args.max_train_rows, random_state=42).sort_index()
    X_train = build_features(train, keep_key=False)
    # categories the model never saw become missing; XGBoost routes them down its default branch
    X_test = align_categories(build_features(test, keep_key=False), X_train)
    assert_no_leakage(X_train)
    assert_no_leakage(X_test)
    y_train, y_test = train["breach"].to_numpy(), test["breach"].to_numpy()
    print(
        f"{len(raw):,} requests; train {len(y_train):,} ({y_train.mean():.1%} late), "
        f"test {len(y_test):,} ({y_test.mean():.1%} late); prep {time.time() - t0:.0f}s",
        flush=True,
    )

    t1 = time.time()
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t1
    metrics = evaluate(y_test, model.predict_proba(X_test)[:, 1])
    metrics.update(
        n_train=len(y_train),
        fit_seconds=round(fit_seconds, 1),
        xgboost=xgb.__version__,
        params=PARAMS,
    )
    write_artifacts(model.get_booster(), feature_schema(X_train), thresholds, metrics, args.out)
    print(json.dumps({k: v for k, v in metrics.items() if k != "params"}), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
