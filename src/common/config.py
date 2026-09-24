"""Central configuration.

Everything that varies by environment or that another module might need to agree on
(resource names, the S3 zone layout, the label definition, the CloudWatch namespace)
lives here so there is exactly one source of truth.

Values come from environment variables (see ``.env.example``); the defaults are the
"just works locally" fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

# --- Label definition (Stage 2 uses this; frozen here so tests and the pipeline agree) ---
TRAIN_WINDOW_YEARS = int(os.environ.get("MLOPS_TRAIN_WINDOW_YEARS", "5"))
BREACH_PERCENTILE = float(os.environ.get("MLOPS_BREACH_PERCENTILE", "0.75"))
HOLDOUT_YEARS = int(os.environ.get("MLOPS_HOLDOUT_YEARS", "1"))
# Tickets closed in a stale bulk close (a backlog purge; see src.pipeline.validate):
#   "flag"    keep the breach label, but leave them out of the per-category thresholds
#   "exclude" drop them from training and evaluation
#   "keep"    treat them like any other closed ticket
PURGE_HANDLING = os.environ.get("MLOPS_PURGE_HANDLING", "flag")

# Columns that reveal the outcome. These must never reach the model. The leakage
# guard in ``src.features.build_labels`` asserts against this list.
LEAKY_COLUMNS: tuple[str, ...] = (
    "updated_date",
    "closed_date",
    "status_description",
    "days_to_close",
    "breach",
    "purge_closed",  # derived from closed_date
    "censored",  # "still open at the snapshot", i.e. derived from closed_date
)

# Columns known at intake — the only inputs the model is allowed to see.
INTAKE_COLUMNS: tuple[str, ...] = (
    "service_request_id",
    "requested_date",
    "service_name",
    "agency_responsible",
    "comm_code",
    "comm_name",
    "source",
    "longitude",
    "latitude",
)

# All columns we request from Socrata (intake + outcome + geo).
SOCRATA_SELECT_COLUMNS: tuple[str, ...] = INTAKE_COLUMNS + (
    "updated_date",
    "closed_date",
    "status_description",
    "location_type",
)


@dataclass(frozen=True)
class Config:
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "ca-central-1"))
    project: str = field(default_factory=lambda: os.environ.get("MLOPS_PROJECT", "mlops-aiops"))

    bucket: str = field(
        default_factory=lambda: os.environ.get("MLOPS_BUCKET", "mlops-aiops-datalake-local")
    )
    athena_output: str = field(
        default_factory=lambda: os.environ.get(
            "MLOPS_ATHENA_OUTPUT", "s3://mlops-aiops-athena-results-local/"
        )
    )
    # Must match infra/ aws_athena_workgroup (named after var.project). The 2 GB per-query
    # scan cap lives on that workgroup; AWS's default "primary" workgroup has no cap.
    athena_workgroup: str = field(
        default_factory=lambda: os.environ.get("MLOPS_ATHENA_WORKGROUP", "mlops-aiops")
    )
    glue_database: str = field(
        default_factory=lambda: os.environ.get("MLOPS_GLUE_DATABASE", "mlops_aiops")
    )
    metric_namespace: str = field(
        default_factory=lambda: os.environ.get("MLOPS_METRIC_NAMESPACE", "MLOpsAIOps/Pipeline")
    )
    # Stage 4 registers into this group; Stage 6 scores the latest Approved package in it.
    model_package_group: str = field(
        default_factory=lambda: os.environ.get(
            "MLOPS_MODEL_PACKAGE_GROUP", "mlops-aiops-breach-risk"
        )
    )

    socrata_domain: str = field(
        default_factory=lambda: os.environ.get("SOCRATA_DOMAIN", "data.calgary.ca")
    )
    socrata_resource: str = field(
        default_factory=lambda: os.environ.get("SOCRATA_311_RESOURCE", "iahh-g8bj")
    )
    socrata_app_token: str = field(default_factory=lambda: os.environ.get("SOCRATA_APP_TOKEN", ""))
    # Community District Boundaries: comm_code -> sector / srg, for the Stage 4 fairness check.
    socrata_communities_resource: str = field(
        default_factory=lambda: os.environ.get("SOCRATA_COMMUNITIES_RESOURCE", "surr-xmvs")
    )

    # --- Derived S3 zone URIs ---
    @property
    def raw_prefix(self) -> str:
        return f"s3://{self.bucket}/raw/311"

    @property
    def processed_prefix(self) -> str:
        return f"s3://{self.bucket}/processed/311"

    @property
    def model_artifacts_prefix(self) -> str:
        return f"s3://{self.bucket}/model-artifacts"

    @property
    def scored_prefix(self) -> str:
        return f"s3://{self.bucket}/scored"

    def raw_asof(self, asof: str) -> str:
        return f"{self.raw_prefix}/asof={asof}"

    @property
    def socrata_base_url(self) -> str:
        return f"https://{self.socrata_domain}/resource/{self.socrata_resource}.json"

    @property
    def socrata_communities_url(self) -> str:
        return f"https://{self.socrata_domain}/resource/{self.socrata_communities_resource}.json"

    @property
    def socrata_metadata_url(self) -> str:
        return f"https://{self.socrata_domain}/api/views/{self.socrata_resource}.json"


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()


def aws_tags() -> list[dict[str, str]]:
    """Tags for resources created from Python (SageMaker jobs, models, endpoints).

    Terraform's provider default_tags don't reach anything launched through boto3 or the
    SageMaker SDK, and an untagged job drops out of the per-project cost view (D-031).
    """
    return [{"Key": "project", "Value": get_config().project}]
