"""Stage 4 fairness check. Synthetic predictions over City planning sectors; no AWS calls."""

import json

import numpy as np
import pandas as pd

from src.promote.fairness import REPORT_NAME, FairnessThresholds, fairness_report, main

SECTORS = {
    "NORTHWEST": "ESTABLISHED",
    "NORTHEAST": "DEVELOPING",
    "SOUTHEAST": "DEVELOPING",
    "CENTRE": "ESTABLISHED",
}


def _communities() -> pd.DataFrame:
    # two communities per sector
    rows = []
    for sector, srg in SECTORS.items():
        for k in range(2):
            rows.append(
                {"comm_code": f"{sector[:2]}{k}", "name": f"{sector} {k}", "sector": sector,
                 "srg": srg, "class": "Residential"}
            )  # fmt: skip
    return pd.DataFrame(rows)


def _predictions(blind_sector: str | None = None, per_community: int = 1_500, seed: int = 7):
    """Scores carry real signal (breach + noise), except in ``blind_sector``, where they
    are pure noise: the model can't tell a slow request from a quick one there."""
    rng = np.random.default_rng(seed)
    frames = []
    for code in _communities()["comm_code"]:
        breach = (rng.random(per_community) < 0.25).astype(int)
        noise = rng.normal(0, 0.7, per_community)
        signal = 0 if blind_sector and code.startswith(blind_sector[:2]) else breach
        frames.append(pd.DataFrame({"comm_code": code, "breach": breach, "pred": signal + noise}))
    return pd.concat(frames, ignore_index=True)


def _reject(token):
    raise ValueError(f"non-JSON constant {token}")


def test_even_model_passes():
    report = fairness_report(_predictions(), _communities())
    assert report["passed"], report
    assert report["recall_ratio"]["value"] >= 0.8
    assert report["min_group_lift"]["value"] >= 1.5


def test_model_blind_in_one_sector_fails_and_names_it():
    report = fairness_report(_predictions(blind_sector="NORTHWEST"), _communities())
    assert not report["passed"]
    sector = report["dimensions"]["sector"]
    assert not sector["passed"]
    assert sector["lowest_recall_group"] == "NORTHWEST"
    assert sector["recall_ratio"]["value"] < 0.8
    # the srg rollup dilutes it (NORTHWEST is half of ESTABLISHED) but still shows it
    srg = report["dimensions"]["srg"]
    assert srg["lowest_recall_group"] == "ESTABLISHED"


def test_small_groups_are_reported_not_gated():
    preds = _predictions()
    comms = pd.concat(
        [_communities(), pd.DataFrame([{"comm_code": "TINY", "sector": "WEST", "srg": "FUTURE"}])],
        ignore_index=True,
    )
    # 40 rows where the model is useless: too few to measure, so not held against it
    tiny = pd.DataFrame({"comm_code": "TINY", "breach": [1, 0] * 20, "pred": -5.0})
    report = fairness_report(pd.concat([preds, tiny], ignore_index=True), comms)
    assert report["passed"]
    west = next(g for g in report["dimensions"]["sector"]["groups"] if g["group"] == "WEST")
    assert west["gated"] is False
    assert west["rows"] == 40


def test_requests_without_a_community_are_never_gated():
    preds = _predictions()
    no_comm = pd.DataFrame({"comm_code": None, "breach": [1, 0] * 1_000, "pred": -5.0})
    report = fairness_report(pd.concat([preds, no_comm], ignore_index=True), _communities())
    assert report["passed"]
    groups = {g["group"]: g for g in report["dimensions"]["sector"]["groups"]}
    assert groups["UNKNOWN"]["gated"] is False
    assert groups["UNKNOWN"]["rows"] == 2_000


def test_fewer_than_two_measurable_groups_fails_closed():
    preds = _predictions()
    preds = preds[preds["comm_code"].str.startswith("NO")]  # NORTHWEST only
    report = fairness_report(preds, _communities())
    assert not report["passed"]
    assert report["recall_ratio"]["value"] is None
    assert "only 1 group" in report["dimensions"]["sector"]["problems"][0]


def test_main_writes_json_the_condition_step_can_read(tmp_path):
    preds_dir, out_dir = tmp_path / "preds", tmp_path / "out"
    preds_dir.mkdir()
    _predictions(blind_sector="SOUTHEAST").to_csv(preds_dir / "predictions.csv", index=False)
    comm_file = tmp_path / "communities.json"
    comm_file.write_text(_communities().to_json(orient="records"))

    main(
        ["--predictions", str(preds_dir), "--communities", str(comm_file),
         "--output", str(out_dir)],
        thresholds=FairnessThresholds(),
    )  # fmt: skip
    # the pipeline's JsonGet needs strict JSON, so refuse NaN / Infinity when parsing
    report = json.loads((out_dir / REPORT_NAME).read_text(), parse_constant=_reject)
    assert report["passed"] is False
    assert isinstance(report["recall_ratio"]["value"], float)
    assert isinstance(report["min_group_lift"]["value"], float)
