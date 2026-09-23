"""CloudWatch custom-metric emitters.

Stage 3 onward, every pipeline run emits its own operational metrics here — run
duration, rows processed, and a dollar cost estimate. Stage 7's anomaly detector
reads this same namespace back out of CloudWatch, so the AIOps layer is only as
good as what we emit now. Keep the metric count small (~8): CloudWatch custom
metrics are $0.30/metric/month and are quietly the largest line in the cost model.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import boto3

from src.common.config import get_config

log = logging.getLogger(__name__)

# Rough on-demand USD/hour for the instances this project uses. Spot is ~70% less;
# pass spot=True where the job actually ran on spot. These are deliberately
# approximate — the point is a consistent, trackable estimate, not an invoice.
INSTANCE_HOURLY_USD = {
    "ml.m5.large": 0.115,
    "ml.m5.xlarge": 0.230,
    "ml.t3.medium": 0.050,
    "glue.dpu": 0.44,  # per DPU-hour
}

_DISABLED = os.environ.get("MLOPS_METRICS_DISABLED") == "1"


def _client():
    cfg = get_config()
    return boto3.client("cloudwatch", region_name=cfg.region)


def emit_run_metric(
    name: str,
    value: float,
    unit: str = "None",
    *,
    stage: str,
    dimensions: dict[str, str] | None = None,
) -> None:
    """Put a single datapoint into the project namespace.

    ``stage`` becomes a dimension so Stage 7 can baseline each stage separately
    ("preprocess is normally 90s") rather than lumping the whole pipeline together.
    """
    cfg = get_config()
    dims = [{"Name": "Stage", "Value": stage}]
    for k, v in (dimensions or {}).items():
        dims.append({"Name": k, "Value": str(v)})

    if _DISABLED:
        log.info("metric (disabled) %s=%s %s dims=%s", name, value, unit, dims)
        return

    _client().put_metric_data(
        Namespace=cfg.metric_namespace,
        MetricData=[
            {
                "MetricName": name,
                "Value": float(value),
                "Unit": unit,
                "Timestamp": datetime.now(UTC),
                "Dimensions": dims,
            }
        ],
    )
    log.info("metric %s=%s %s dims=%s", name, value, unit, dims)


def estimate_cost_usd(
    instance_type: str, seconds: float, *, count: int = 1, spot: bool = False
) -> float:
    hourly = INSTANCE_HOURLY_USD.get(instance_type, 0.0)
    if spot:
        hourly *= 0.30
    return round(hourly * (seconds / 3600.0) * count, 4)


def emit_cost_estimate(
    instance_type: str,
    seconds: float,
    *,
    stage: str,
    count: int = 1,
    spot: bool = False,
) -> float:
    """Estimate and emit the dollar cost of a job. Returns the estimate."""
    usd = estimate_cost_usd(instance_type, seconds, count=count, spot=spot)
    emit_run_metric(
        "RunCostUsd",
        usd,
        unit="None",
        stage=stage,
        dimensions={"InstanceType": instance_type, "Spot": str(spot)},
    )
    emit_run_metric("RunDurationSeconds", seconds, unit="Seconds", stage=stage)
    return usd


class timed_stage:
    """Context manager: on exit, emit duration + cost for a stage.

    >>> with timed_stage("preprocess", instance_type="ml.m5.large", spot=True):
    ...     run_processing_job()
    """

    def __init__(self, stage: str, *, instance_type: str = "ml.m5.large", spot: bool = False):
        self.stage = stage
        self.instance_type = instance_type
        self.spot = spot
        self._start = 0.0

    def __enter__(self) -> timed_stage:
        self._start = datetime.now(UTC).timestamp()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        seconds = datetime.now(UTC).timestamp() - self._start
        emit_run_metric("StepFailure", 1.0 if exc_type else 0.0, stage=self.stage)
        emit_cost_estimate(self.instance_type, seconds, stage=self.stage, spot=self.spot)
