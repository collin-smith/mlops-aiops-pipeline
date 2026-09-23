"""Minimal Athena query helper — run SQL, wait, return rows as a DataFrame.

Deliberately tiny. Queries in this project scan MB (partitioned Parquet), so a
simple poll loop is fine and avoids an awswrangler dependency.
"""

from __future__ import annotations

import time

import boto3
import pandas as pd

from src.common.config import get_config

_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


def run_query(sql: str, *, database: str | None = None, poll_seconds: float = 1.0) -> pd.DataFrame:
    cfg = get_config()
    client = boto3.client("athena", region_name=cfg.region)

    start = client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database or cfg.glue_database},
        ResultConfiguration={"OutputLocation": cfg.athena_output},
        WorkGroup=cfg.athena_workgroup,
    )
    qid = start["QueryExecutionId"]

    while True:
        info = client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = info["Status"]["State"]
        if state in _TERMINAL:
            break
        time.sleep(poll_seconds)

    if state != "SUCCEEDED":
        reason = info["Status"].get("StateChangeReason", "unknown")
        raise RuntimeError(f"Athena query {qid} {state}: {reason}")

    scanned_mb = info["Statistics"].get("DataScannedInBytes", 0) / 1e6
    rows: list[list[str]] = []
    header: list[str] | None = None
    paginator = client.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=qid):
        for r in page["ResultSet"]["Rows"]:
            values = [c.get("VarCharValue") for c in r["Data"]]
            if header is None:
                header = values
            else:
                rows.append(values)

    df = pd.DataFrame(rows, columns=header or [])
    df.attrs["scanned_mb"] = round(scanned_mb, 2)
    df.attrs["query_execution_id"] = qid
    return df
