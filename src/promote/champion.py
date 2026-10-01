"""Champion / challenger (Stage 5, D-043): the newly trained model has to beat the one
already approved, on the same test year, by more than a guardband.

The champion is the latest **Approved** version in the Model Package Group, not the last
model trained: an unapproved model is nobody's baseline. Both models score the same rows,
the challenger's test year, each with the features its own schema asks for, so a champion
trained on the baseline feature set is judged fairly against a challenger on the new one.

The guardband stops noise from promoting: a challenger that wins by 0.001 PR-AUC hasn't
shown anything a different random seed couldn't undo.

The Evaluate step writes the result into ``evaluation.json`` as ``champion``:

    {"exists": 1, "arn": "...", "pr_auc": 0.36, "margin": {"value": 0.014}}
    {"exists": 0, "arn": null, "pr_auc": null, "margin": {"value": 0.0}}

and the gate passes this check if ``exists`` is 0 or ``margin`` ≥ ``GUARDBAND``. Both are
always numbers, so the gate's JSON reads never meet a null.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

GUARDBAND = 0.005


def latest_approved(sm, group: str) -> dict | None:
    """The newest Approved version in ``group``, described, or None."""
    resp = sm.list_model_packages(
        ModelPackageGroupName=group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    summaries = resp.get("ModelPackageSummaryList", [])
    if not summaries:
        return None
    return sm.describe_model_package(ModelPackageName=summaries[0]["ModelPackageArn"])


def download_model(s3, model_data_url: str, into: Path) -> Path:
    bucket, key = model_data_url.removeprefix("s3://").split("/", 1)
    into.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(into / "model.tar.gz"))
    return into


def comparison(challenger_pr_auc: float, champion: dict | None, champion_pr_auc: float | None):
    if champion is None:
        return {"exists": 0, "arn": None, "pr_auc": None, "margin": {"value": 0.0}}
    margin = round(challenger_pr_auc - champion_pr_auc, 4)
    return {
        "exists": 1,
        "arn": champion["ModelPackageArn"],
        "pr_auc": champion_pr_auc,
        "margin": {"value": margin},
    }


def compare(sm, s3, group: str, challenger_pr_auc: float, score_with) -> dict:
    """``score_with(model_dir)`` returns a saved model's PR-AUC on the challenger's test
    year. Any error reading the registry propagates: no comparison, no pass."""
    champion = latest_approved(sm, group)
    if champion is None:
        return comparison(challenger_pr_auc, None, None)
    url = champion["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    with tempfile.TemporaryDirectory() as tmp:
        champion_pr_auc = score_with(download_model(s3, url, Path(tmp)))
    return comparison(challenger_pr_auc, champion, champion_pr_auc)
