"""Pipeline health from SageMaker's own job history (Stage 7 Layer B, D-045).

The source is ``ListProcessingJobs`` + ``DescribeProcessingJob``: free, and it covers every
job the project has run (smoke tests, training, pipeline steps, scoring, the analyzer),
where the Stage 3 metrics cover three pipeline runs. Each job is reduced to its kind, its
billed seconds, an estimated cost, and **seconds per million input rows**, which is how a
slow job is told apart from a bigger snapshot.

The rule is stated, not fitted: a job is flagged if it failed, or if its seconds per
million rows (billed seconds, for kinds that don't read the snapshot) are more than twice
the median of the other jobs of its kind
(``detect.ratio_flags``). There are a handful of jobs per kind, and an EWMA on three points
would be theatre. ``detect.ewma_flags`` takes over once a kind has ten.
"""

from __future__ import annotations

import re

import pandas as pd

from src.common.metrics import estimate_cost_usd
from src.monitor import detect

# Rows in each snapshot ``processed/311`` has held (validation reports, cost log). A job
# is matched to the newest snapshot taken on or before its start. Add new ones with
# ``scripts/pipeline_health.py --snapshot <asof>=<rows>``.
SNAPSHOT_ROWS = {"2026-09-23": 2_911_486, "2026-10-01": 2_921_625}
# Only these kinds read the whole snapshot, so only they are judged per million rows. The
# others (smoke tests, the analyzer reading a sampled CSV) are judged on billed seconds.
SNAPSHOT_KINDS = ("train", "score", "pipeline-")
_STAMP = re.compile(r"-\d{8}-\d{6}$")


def job_kind(name: str, project: str) -> str | None:
    """What a job was for, from its name; None if it isn't this project's.

    Launcher jobs are ``<project>-<kind>-<YYYYmmdd-HHMMSS>``; pipeline steps are
    ``pipelines-<execution>-<Step>-<suffix>``.
    """
    if name.startswith(f"{project}-"):
        return _STAMP.sub("", name[len(project) + 1 :]) or None
    if name.startswith("pipelines-"):
        parts = name.split("-")
        return f"pipeline-{parts[2].lower()}" if len(parts) >= 4 else "pipeline"
    return None


def job_records(sm, project: str) -> pd.DataFrame:
    """One row per project Processing job in this region, from SageMaker's history."""
    rows = []
    for page in sm.get_paginator("list_processing_jobs").paginate():
        for item in page["ProcessingJobSummaries"]:
            name = item["ProcessingJobName"]
            kind = job_kind(name, project)
            if kind is None:
                continue
            d = sm.describe_processing_job(ProcessingJobName=name)
            start, end = d.get("ProcessingStartTime"), d.get("ProcessingEndTime")
            cluster = d["ProcessingResources"]["ClusterConfig"]
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "status": d["ProcessingJobStatus"],
                    "created": d["CreationTime"],
                    "billed_seconds": (end - start).total_seconds() if start and end else None,
                    "instance_type": cluster["InstanceType"],
                }
            )
    return pd.DataFrame(rows)


def rows_at(created: pd.Series, snapshots: dict[str, int]) -> pd.Series:
    """The row count of the newest snapshot on or before each job (NaN before the first)."""
    snaps = sorted((pd.Timestamp(k, tz="UTC"), v) for k, v in snapshots.items())
    created = pd.to_datetime(created, utc=True)

    def lookup(t):
        rows = [v for asof, v in snaps if asof <= t]
        return rows[-1] if rows else float("nan")

    return created.map(lookup)


def assess(jobs: pd.DataFrame, snapshots: dict[str, int] | None = None) -> pd.DataFrame:
    """``jobs`` plus cost, seconds per million rows, and a ``flag`` with its reason."""
    out = jobs.sort_values("created").reset_index(drop=True).copy()
    out["usd"] = [
        estimate_cost_usd(t, s) if pd.notna(s) else float("nan")
        for t, s in zip(out["instance_type"], out["billed_seconds"], strict=True)
    ]
    reads_snapshot = out["kind"].str.startswith(SNAPSHOT_KINDS)
    out["rows"] = rows_at(out["created"], snapshots or SNAPSHOT_ROWS).where(reads_snapshot)
    out["sec_per_mrows"] = out["billed_seconds"] / (out["rows"] / 1e6)
    # what each job is judged on: per million rows if it reads the snapshot, else seconds
    measure = out["sec_per_mrows"].where(reads_snapshot, out["billed_seconds"])
    done = out["status"] == "Completed"
    slow = pd.Series(False, index=out.index)
    judged = out.loc[done & measure.notna()]
    slow.loc[judged.index] = detect.ratio_flags(measure[judged.index], judged["kind"])
    failed = out["status"] == "Failed"
    out["flag"] = slow | failed
    out["reason"] = ""
    out.loc[slow, "reason"] = f"> {detect.RATIO_FACTOR:g}× its kind's median"
    out.loc[failed, "reason"] = "failed"
    return out


def alert_text(assessed: pd.DataFrame) -> str | None:
    """The SNS message for flagged jobs, or None when there's nothing to say."""
    flagged = assessed.loc[assessed["flag"]]
    if flagged.empty:
        return None
    lines = [f"{len(flagged)} job(s) flagged by the Stage 7 pipeline-health check:"]
    for r in flagged.itertuples():
        secs = f"{r.billed_seconds:.0f} s" if pd.notna(r.billed_seconds) else "no billed time"
        lines.append(f"- {r.name} ({r.kind}): {r.reason}; {secs}")
    lines.append("A flag is a reason to look, not a diagnosis.")
    return "\n".join(lines)
