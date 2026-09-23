"""Community map — the classification logic, tested without matplotlib or the network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "community_map.py"
_spec = importlib.util.spec_from_file_location("community_map", _PATH)
cmap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmap)


def test_quantile_breaks_split_communities_evenly():
    breaks = cmap.quantile_breaks([float(v) for v in range(1, 11)], k=5)
    assert breaks == [2.0, 4.0, 6.0, 8.0]
    counts = [0] * 5
    for v in range(1, 11):
        counts[cmap.classify(float(v), breaks)] += 1
    assert counts == [2, 2, 2, 2, 2]


def test_quantile_breaks_needs_enough_communities():
    with pytest.raises(ValueError):
        cmap.quantile_breaks([1.0, 2.0], k=5)


def test_filter_min_n_drops_thin_communities():
    metric = {"AAA": (0.3, 500), "BBB": (0.9, 12)}
    assert cmap.filter_min_n(metric, 200) == {"AAA": 0.3}


def test_exterior_rings_handles_polygon_and_multipolygon():
    ring = [[0, 0], [1, 0], [1, 1], [0, 0]]
    hole = [[0.2, 0.2], [0.3, 0.2], [0.3, 0.3], [0.2, 0.2]]
    assert cmap.exterior_rings({"type": "Polygon", "coordinates": [ring, hole]}) == [
        [(0, 0), (1, 0), (1, 1), (0, 0)]
    ]
    multi = {"type": "MultiPolygon", "coordinates": [[ring], [ring]]}
    assert len(cmap.exterior_rings(multi)) == 2
    assert cmap.exterior_rings({"type": "Point", "coordinates": [0, 0]}) == []


def test_load_metrics_csv_skips_blank_rows(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("comm_code,value,n\nAAA,0.25,300\n,0.5,10\nBBB,,40\n")
    assert cmap.load_metrics_csv(p) == {"AAA": (0.25, 300)}
