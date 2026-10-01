"""Inference handler for the Stage 6 Serverless demo (D-029, D-044).

The built-in XGBoost container's default handler reads a numeric CSV with no feature types.
This model was trained on pandas categoricals (``enable_categorical``), so a category sent
as a number would be read as a number, and the score would be quietly wrong. This handler
takes JSON records of the model's own features, gives each categorical column the training
levels in ``feature_schema.json`` (as ``align_to_schema`` does for batch scoring), and
scores with the booster.

It is packed on its own (``sourcedir.tar.gz``) and runs inside AWS's image, where ``src``
isn't installed, so it imports nothing from this repo. A test checks its alignment matches
``src.features.build_features.align_to_schema``.

Request:  {"rows": [{"service_name": "...", "cat_late_90d": 0.21, ...}, ...]}
Response: {"usage": "shadow-not-for-use", "scores": [0.31, ...]}
"""

from __future__ import annotations

import json
import os

import pandas as pd

USAGE = "shadow-not-for-use"
JSON = "application/json"


def align(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = frame.reindex(columns=schema["columns"]).copy()
    for c, levels in schema["categories"].items():
        values = out[c].astype("string")
        out[c] = pd.Categorical(values.where(values.isin(levels)), categories=levels)
    for c in out.columns:
        if c not in schema["categories"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def model_fn(model_dir: str):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "xgboost-model"))
    with open(os.path.join(model_dir, "feature_schema.json")) as f:
        schema = json.load(f)
    return booster, schema


def input_fn(body, content_type: str) -> pd.DataFrame:
    if content_type != JSON:
        raise ValueError(f"send {JSON} rows of the model's features, not {content_type}")
    data = json.loads(body)
    return pd.DataFrame.from_records(data["rows"])


def predict_fn(frame: pd.DataFrame, model) -> list[float]:
    import xgboost as xgb

    booster, schema = model
    scores = booster.predict(xgb.DMatrix(align(frame, schema), enable_categorical=True))
    return [round(float(s), 6) for s in scores]


def output_fn(scores: list[float], accept: str):
    return json.dumps({"usage": USAGE, "scores": scores}), JSON
