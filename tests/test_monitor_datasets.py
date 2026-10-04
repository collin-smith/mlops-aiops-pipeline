"""Stage 7 Layer A inputs (D-045): the intake window, the baseline's counts, the injected copy."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_labels import build_training_labels
from src.monitor import datasets, inject_drift
from src.pipeline.train import featurize, recent

ASOF = pd.Timestamp("2026-10-01", tz="UTC")


def test_intake_is_every_request_filed_in_the_window_open_or_closed():
    rows = [
        {"requested_date": (ASOF - pd.Timedelta(days=d)).isoformat(), "closed_date": c}
        for d, c in ((1, None), (10, "2026-09-25"), (29, None), (31, "2026-09-02"))
    ]
    out = datasets.intake_rows(pd.DataFrame(rows), ASOF, days=30)
    assert len(out) == 3


def test_baseline_counts_match_featurising_the_whole_split(requests_frame):
    """Sampling after featurising: a sampled baseline must carry the full split's counts."""
    sectors = pd.Series({"X": "EAST"})
    train, _, thresholds = build_training_labels(requests_frame)
    full = featurize(recent(train, 2), requests_frame, thresholds, sectors, "challenger")
    table = datasets.baseline_table(requests_frame, thresholds, sectors, sample=40)
    assert len(table) == 40
    assert list(table.columns) == datasets.CSV_COLUMNS
    sampled = full.sample(n=40, random_state=datasets.SEED).sort_index()
    assert table["cat_open_30d"].tolist() == sampled["cat_open_30d"].tolist()


def _intake():
    names = ["Roads - Pothole Maintenance"] * 10 + ["Bylaw - Noise Concerns"] * 90
    return pd.DataFrame({"service_name": names, "req_month": 10})


def test_injection_sets_the_share_and_keeps_the_size():
    out, note = inject_drift.inject(_intake(), share=0.3)
    assert len(out) == 100
    assert note["synthetic"] is True
    assert note["share_before"] == 0.1 and note["share_after"] == 0.3
    assert out["service_name"].str.contains("Pothole").mean() == 0.3


def test_injection_refuses_a_pattern_with_no_rows():
    with pytest.raises(SystemExit, match="no rows"):
        inject_drift.inject(_intake(), pattern="Volcano")


def test_department_is_the_prefix_or_other():
    names = pd.Series(
        ["Roads - Pothole Maintenance", "WRS - Cart Management", "Zoo - Penguins", None, "Law"]
    )
    assert datasets.department(names).tolist() == ["Roads", "WRS", "Other", "Other", "Law"]
    assert len(datasets.DEPARTMENTS) == 30
