import pandas as pd
import pytest

from src.common.config import LEAKY_COLUMNS
from src.features.build_features import (
    FEATURE_COLUMNS,
    add_backlog_features,
    add_calendar_features,
    build_features,
)


def test_build_features_emits_only_allowlisted_columns(requests_frame):
    feats = build_features(requests_frame)
    assert set(feats.columns) - {"service_request_id"} == set(FEATURE_COLUMNS)
    assert not (set(feats.columns) & set(LEAKY_COLUMNS))


def test_calendar_features_are_in_range(requests_frame):
    out = add_calendar_features(requests_frame)
    assert out["req_month"].between(1, 12).all()
    assert out["req_dow"].between(0, 6).all()
    assert set(out["req_is_weekend"].unique()) <= {0, 1}
    assert set(out["req_is_holiday_week"].unique()) <= {0, 1}


def test_backlog_count_excludes_self_and_future():
    # three requests, same category, 5 days apart
    df = pd.DataFrame(
        {
            "service_request_id": ["a", "b", "c"],
            "requested_date": [
                "2022-06-01T00:00:00",
                "2022-06-06T00:00:00",
                "2022-06-11T00:00:00",
            ],
            "service_name": ["Snow Removal"] * 3,
            "comm_name": ["BELTLINE"] * 3,
        }
    )
    out = add_backlog_features(df)
    # first row: nothing before it; second: 1 before; third: 2 before
    assert list(out["cat_open_30d"]) == [0, 1, 2]


def test_backlog_count_respects_30_day_window():
    df = pd.DataFrame(
        {
            "service_request_id": ["a", "b"],
            "requested_date": ["2022-01-01T00:00:00", "2022-03-01T00:00:00"],
            "service_name": ["Weed Concern"] * 2,
            "comm_name": ["BOWNESS"] * 2,
        }
    )
    out = add_backlog_features(df)
    # 59 days apart -> outside the 30-day window
    assert list(out["cat_open_30d"]) == [0, 0]


def test_backlog_is_partitioned_by_key():
    df = pd.DataFrame(
        {
            "service_request_id": ["a", "b", "c"],
            "requested_date": ["2022-06-01", "2022-06-02", "2022-06-03"],
            "service_name": ["Pothole", "Tree", "Pothole"],
            "comm_name": ["A", "B", "A"],
        }
    )
    out = add_backlog_features(df)
    # only the two potholes see each other
    assert list(out["cat_open_30d"]) == [0, 0, 1]


def test_athena_helper_uses_the_capped_project_workgroup(monkeypatch):
    """Queries must run in infra/'s workgroup (2 GB scan cap), never AWS's default 'primary'."""
    from src.common import athena
    from src.common.config import get_config

    captured = {}

    class _Stop(Exception):
        pass

    class _FakeAthena:
        def start_query_execution(self, **kwargs):
            captured.update(kwargs)
            raise _Stop

    monkeypatch.setattr(athena.boto3, "client", lambda *a, **k: _FakeAthena())
    with pytest.raises(_Stop):
        athena.run_query("SELECT 1")
    assert captured["WorkGroup"] == get_config().athena_workgroup == "mlops-aiops"


def test_aws_tags_carry_the_project_tag_the_budgets_filter_on():
    from src.common.config import aws_tags

    assert aws_tags() == [{"Key": "project", "Value": "mlops-aiops"}]
