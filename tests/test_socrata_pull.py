"""Ingest snapshot handling: processed/311 holds exactly one snapshot, locally and in S3.

A second pull used to add a second copy of every month: `write_to_dataset` names each file
with a fresh uuid, and `snapshot` uploaded without deleting. No AWS calls; S3 is faked.
"""

import argparse
import gzip
import json

import pandas as pd
import pytest

from src.ingest import socrata_pull as sp


def _write_raw(root, asof, n, id_prefix="SR"):
    raw = root / "raw" / "311" / f"asof={asof}"
    raw.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw / "part-00000.ndjson.gz", "wt") as f:
        for i in range(n):
            month = 1 + i % 3
            row = {
                "service_request_id": f"{id_prefix}-{i:04d}",
                "requested_date": f"2025-0{month}-1{i % 9}T00:00:00.000",
                "closed_date": f"2025-0{month}-2{i % 9}T00:00:00.000",
                "service_name": "Pothole Repair",
            }
            f.write(json.dumps(row) + "\n")


def _rows(root):
    return len(pd.read_parquet(root / "processed" / "311"))


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "DATA_ROOT", tmp_path)
    return tmp_path


def test_rebuilding_processed_replaces_it_instead_of_adding_a_copy(data_root):
    _write_raw(data_root, "2026-09-23", 30)
    sp.cmd_to_parquet(argparse.Namespace(asof="2026-09-23"))
    assert _rows(data_root) == 30

    # the same snapshot again, then a newer one: still one copy, of the latest
    sp.cmd_to_parquet(argparse.Namespace(asof="2026-09-23"))
    assert _rows(data_root) == 30
    _write_raw(data_root, "2026-10-07", 36)
    sp.cmd_to_parquet(argparse.Namespace(asof="2026-10-07"))
    assert _rows(data_root) == 36


def test_stale_keys_are_the_ones_not_just_uploaded():
    remote = [
        "processed/311/year=2025/month=1/old.parquet",
        "processed/311/year=2025/month=1/new.parquet",
    ]
    assert sp.stale_keys(remote, {"processed/311/year=2025/month=1/new.parquet"}) == [
        "processed/311/year=2025/month=1/old.parquet"
    ]


class FakeS3:
    """Just enough of the boto3 S3 client for snapshot and replay."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def upload_file(self, path, bucket, key):
        with open(path, "rb") as f:
            self.objects[key] = f.read()

    def download_file(self, bucket, key, path):
        with open(path, "wb") as f:
            f.write(self.objects[key])

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)

    def get_paginator(self, name):
        store = self.objects

        class Pager:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in sorted(store) if k.startswith(Prefix)]}

        return Pager()


def test_snapshot_mirrors_processed_and_keeps_raw_history(data_root, monkeypatch):
    stale = "processed/311/year=2025/month=1/from-an-older-build.parquet"
    old_raw = "raw/311/asof=2026-09-01/part-00000.ndjson.gz"
    s3 = FakeS3({stale: b"x", old_raw: b"x"})
    monkeypatch.setattr(sp, "_s3", lambda: s3)

    _write_raw(data_root, "2026-09-23", 30)
    sp.cmd_to_parquet(argparse.Namespace(asof="2026-09-23"))
    sp.cmd_snapshot(argparse.Namespace(asof="2026-09-23"))

    assert stale not in s3.objects  # processed/ is mirrored
    assert old_raw in s3.objects  # raw/ history is never deleted
    assert any(k.startswith("processed/311/year=2025/") for k in s3.objects)


def test_replay_rebuilds_processed_from_that_dates_raw_files(data_root, monkeypatch):
    # S3 holds two raw snapshots; processed/ there is the newer build
    _write_raw(data_root, "2026-09-23", 30)
    _write_raw(data_root, "2026-10-07", 36, id_prefix="NEW")
    s3 = FakeS3()
    for asof in ("2026-09-23", "2026-10-07"):
        f = data_root / "raw" / "311" / f"asof={asof}" / "part-00000.ndjson.gz"
        s3.objects[f"raw/311/asof={asof}/part-00000.ndjson.gz"] = f.read_bytes()
    monkeypatch.setattr(sp, "_s3", lambda: s3)
    monkeypatch.setattr(sp, "get_config", lambda: type("C", (), {"bucket": "b"})())

    sp.cmd_to_parquet(argparse.Namespace(asof="2026-10-07"))  # local processed = newer build
    sp.cmd_replay(argparse.Namespace(asof="2026-09-23"))

    ids = pd.read_parquet(data_root / "processed" / "311")["service_request_id"]
    assert len(ids) == 30
    assert ids.str.startswith("SR-").all()
