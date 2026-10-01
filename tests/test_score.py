"""Stage 6 shadow scoring (D-044): which rows get a score, the flag, the output's labels, and
the Serverless handler agreeing with the batch job. No AWS calls.

The end-to-end tests need xgboost and scikit-learn (the `ml` extra) and skip without them.
"""

from __future__ import annotations

import json
import tarfile

import pandas as pd
import pytest

from src.deploy import inference, score
from src.features.build_features import align_to_schema
from src.features.build_labels import overdue_open

THRESHOLDS = pd.Series({"Pothole Repair": 5.0, "Tree Concern": 60.0, "__global__": 10.0})
ASOF = pd.Timestamp("2026-09-24", tz="UTC")


def _req(i, cat, filed_days_ago, closed=False):
    t = ASOF - pd.Timedelta(days=filed_days_ago)
    return {
        "service_request_id": f"R{i}",
        "requested_date": t.isoformat(),
        "closed_date": (t + pd.Timedelta(days=1)).isoformat() if closed else None,
        "service_name": cat,
        "comm_code": "X",
    }


def _frame():
    return pd.DataFrame(
        [
            _req(1, "Pothole Repair", 2),  # open, inside 5 days -> scored
            _req(2, "Pothole Repair", 9),  # open, past its deadline -> already late
            _req(3, "Pothole Repair", 2, closed=True),  # closed -> decided
            _req(4, "Tree Concern", 30),  # open, inside 60 days -> scored
            _req(5, "Brand New Category", 3),  # unseen: the 10-day global threshold -> scored
            _req(6, "Brand New Category", 12),  # unseen, past 10 days -> already late
        ]
    )


def test_scoring_rows_are_open_and_inside_their_deadline():
    rows = score.scoring_rows(_frame(), THRESHOLDS, ASOF)
    assert rows["service_request_id"].tolist() == ["R1", "R4", "R5"]
    assert rows.set_index("service_request_id")["category_seen"].to_dict() == {
        "R1": True,
        "R4": True,
        "R5": False,
    }
    assert (rows["deadline"] >= ASOF).all()


def test_scoring_rows_and_overdue_open_split_the_open_requests_exactly():
    frame = _frame()
    scored = set(score.scoring_rows(frame, THRESHOLDS, ASOF)["service_request_id"])
    late = set(overdue_open(frame, THRESHOLDS, ASOF)["service_request_id"])
    still_open = set(frame.loc[frame["closed_date"].isna(), "service_request_id"])
    assert scored.isdisjoint(late)
    assert scored | late == still_open


def test_snapshot_time_is_the_day_after_the_newest_request():
    assert score.snapshot_time(_frame()) == pd.Timestamp("2026-09-23", tz="UTC")


def test_context_frame_reaches_back_a_month_before_the_oldest_scored_row():
    frame = pd.concat([_frame(), pd.DataFrame([_req(9, "Pothole Repair", 400, closed=True)])])
    rows = score.scoring_rows(frame, THRESHOLDS, ASOF)
    context = score.context_frame(frame, rows)
    assert "R9" not in set(context["service_request_id"])  # far older than any scored row
    assert set(rows["service_request_id"]) <= set(context["service_request_id"])


def test_flag_takes_the_top_tenth_of_every_sector():
    rows = []
    for sector, offset in (("CENTRE", 0.5), ("WEST", 0.0)):  # CENTRE's scores all run hotter
        rows += [{"sector": sector, "pred": offset + i / 1000} for i in range(100)]
    flagged = score.flag(pd.DataFrame(rows))
    assert flagged.groupby("sector")["flagged"].sum().to_dict() == {"CENTRE": 10, "WEST": 10}
    top = flagged.loc[flagged["rank_in_sector"] == 1, "pred"].tolist()
    assert top == [0.599, 0.099]


def test_handler_alignment_matches_batch_alignment():
    schema = {
        "columns": ["service_name", "cat_late_90d", "req_dow"],
        "categories": {"service_name": ["Pothole Repair", "Tree Concern"]},
    }
    frame = pd.DataFrame(
        {
            "req_dow": [1, 2, 3],
            "service_name": ["Tree Concern", "Unseen", None],
            "cat_late_90d": [0.2, None, 0.5],
        }
    )
    batch = align_to_schema(frame, schema)
    # what the endpoint gets: the same rows after a trip through JSON
    sent = pd.DataFrame.from_records(json.loads(frame.to_json(orient="records")))
    live = inference.align(sent, schema)
    pd.testing.assert_frame_equal(live, batch, check_dtype=False)
    assert batch["service_name"].cat.codes.tolist() == [1, -1, -1]  # unseen -> missing


def test_handler_refuses_csv():
    with pytest.raises(ValueError):
        inference.input_fn("1,2,3", "text/csv")


# --- end to end on the fixture: train a challenger, score with its artifact ---


@pytest.fixture
def scored_run(tmp_path, requests_frame):
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    from src.pipeline import train

    newest = pd.to_datetime(requests_frame["requested_date"], utc=True).max()
    fresh = pd.DataFrame(
        [
            {
                **requests_frame.iloc[i % 2].to_dict(),
                "service_request_id": f"NEW-{i}",
                "requested_date": (newest - pd.Timedelta(days=i % 3)).isoformat(),
                "closed_date": None,
                "status_description": "Open",
            }
            for i in range(30)
        ]
    )
    frame = pd.concat([requests_frame, fresh], ignore_index=True)
    data, comms, model = tmp_path / "data", tmp_path / "comms", tmp_path / "model"
    data.mkdir()
    comms.mkdir()
    frame.to_parquet(data / "part-0.parquet")
    lookup = [{"comm_code": "X", "sector": "CENTRE", "srg": "ESTABLISHED"}]
    (comms / "communities.json").write_text(json.dumps(lookup))
    args = ["--data", str(data), "--communities", str(comms), "--out", str(model)]
    assert train.main(args) == 0
    return {"data": data, "comms": comms, "model": model, "frame": frame, "lookup": lookup}


def _score(run, **kw):
    when = pd.Timestamp("2026-10-01T00:00:00", tz="UTC").to_pydatetime()
    return score.score(
        run["model"], run["frame"], pd.DataFrame(run["lookup"]), "arn:test/1", scored_at=when, **kw
    )


def test_every_scored_row_is_labelled_not_for_use(scored_run):
    out, summary, _ = _score(scored_run)
    assert list(out.columns) == score.OUTPUT_COLUMNS
    assert (out["usage"] == "shadow-not-for-use").all()
    assert (out["model_ref"] == "arn:test/1").all()
    assert set(out["service_request_id"]) <= {f"NEW-{i}" for i in range(30)} | set(
        scored_run["frame"]["service_request_id"]
    )
    assert summary["rows_scored"] == len(out) > 0
    assert summary["rows_flagged"] == int(out["flagged"].sum()) >= 1


def test_scoring_twice_gives_identical_output(scored_run):
    a, _, _ = _score(scored_run)
    b, _, _ = _score(scored_run)
    pd.testing.assert_frame_equal(a, b)


def test_the_serverless_handler_reproduces_the_batch_scores(scored_run, tmp_path):
    _, _, payload = _score(scored_run)
    unpacked = tmp_path / "unpacked"
    with tarfile.open(scored_run["model"] / "model.tar.gz") as tar:
        tar.extractall(unpacked, filter="data")
    model = inference.model_fn(str(unpacked))
    frame = inference.input_fn(json.dumps({"rows": payload["rows"]}), "application/json")
    body, content_type = inference.output_fn(inference.predict_fn(frame, model), "application/json")
    assert content_type == "application/json"
    response = json.loads(body)
    assert response["usage"] == "shadow-not-for-use"
    assert response["scores"] == pytest.approx(payload["batch_scores"], abs=1e-5)


def test_main_writes_only_parquet_where_the_table_reads(scored_run, tmp_path):
    out = tmp_path / "out"
    args = [
        "--model", str(scored_run["model"]), "--data", str(scored_run["data"]),
        "--communities", str(scored_run["comms"]), "--out", str(out),
        "--model-ref", "arn:test/1", "--model-tag", "v1",
    ]  # fmt: skip
    assert score.main(args) == 0
    assert [p.name for p in (out / "scores").iterdir()] == ["scores-v1.parquet"]
    assert sorted(p.name for p in (out / "report").iterdir()) == ["payload.json", "summary.json"]
    written = pd.read_parquet(out / "scores" / "scores-v1.parquet")
    assert str(written["deadline"].dtype).startswith("datetime64[ms")
