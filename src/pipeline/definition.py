"""The training pipeline as a SageMaker Pipelines definition (Stages 3 and 4).

Three Processing steps, in order (D-039: the account has no Training Job quota):

    Validate  the data-validation gate (D-034); a failed check stops the run before any
              training spend
    Train     the Stage 2 model; writes model.tar.gz and metrics.json
    Evaluate  scores the test year with the saved artifact; writes evaluation.json and
              fairness.json

then the promotion gate (Stage 4, D-041):

    Gate      a Condition step on the two reports: PR-AUC, the sector recall ratio and
              the lowest group lift must all clear ``GATE``, and the model must beat the
              approved champion by ``GUARDBAND`` if there is one (D-043)
    Register  (if it passes) a new version in the Model Package Group, status
              PendingManualApproval; only the approver role can approve it
    Rejected  (if it doesn't) a Fail step, so the run ends Failed with the numbers in
              its reason, and nothing is registered

The gate's thresholds are constants here, not pipeline parameters. Anyone who can start a
run can set a parameter, so a parameter would let the gate be lowered for one run.
Changing a threshold takes a commit.

The definition is plain JSON (schema 2020-12-01), built here without the SageMaker SDK,
so it can be read, diffed and unit-tested as data. ``scripts/pipeline.py`` creates or
updates the pipeline from it and starts runs.

Step caching is on. A Processing step's cache key is its container command, its
environment and its input locations, not the bytes behind them. So the code goes to a
folder named after a hash of its contents (new code, new location, a re-run), and the
snapshot date goes into every step's environment (a new snapshot is a re-run too).
"""

from __future__ import annotations

from src.common.sm_jobs import PROCESSING_INSTANCE_TYPES
from src.promote.champion import GUARDBAND
from src.promote.fairness import FairnessThresholds

IN_CODE = "/opt/ml/processing/input/code"
IN_DATA = "/opt/ml/processing/input/data"
IN_MODEL = "/opt/ml/processing/input/model"
IN_COMMUNITIES = "/opt/ml/processing/input/communities"
OUT = "/opt/ml/processing/output"
REQUIREMENTS = "src/pipeline/job_requirements.txt"

# Each step's hard time limit, in seconds. Full-data times on ml.t3.xlarge (2026-09-28):
# train 239 s billed. Validate and evaluate do less work than train.
MAX_RUNTIME = {"Validate": 900, "Train": 3600, "Evaluate": 1200}
CACHE = {"Enabled": True, "ExpireAfter": "P30D"}

# bash -c <script> <$0> <args...>: installs what the image lacks, then runs the step with
# the job's ContainerArguments. Each entrypoint element must stay under 256 characters.
_BOOT = (
    f'cd {IN_CODE} && pip install -q -r {REQUIREMENTS} && exec python3 -m src.pipeline.step "$@"'
)
ENTRYPOINT = ["bash", "-c", _BOOT, "step"]


# The promotion gate (D-041). Chance-level PR-AUC is the test year's base rate, 0.18; the
# Stage 2 model scores 0.327. The fairness floors are D-035's.
_FAIR = FairnessThresholds()
GATE = {
    "pr_auc": 0.25,
    "recall_ratio": _FAIR.min_recall_ratio,
    "min_group_lift": _FAIR.min_group_lift,
}
# JsonGet paths into the Evaluate step's two reports
# champion / challenger (D-043): passes when there's no approved champion, or the margin
# clears the guardband. Both values are always numbers in evaluation.json.
CHAMPION_PATHS = {
    "champion_exists": ("EvaluationReport", "champion.exists"),
    "champion_margin": ("EvaluationReport", "champion.margin.value"),
}
GATE_PATHS = {
    "pr_auc": ("EvaluationReport", "binary_classification_metrics.pr_auc.value"),
    "recall_ratio": ("FairnessReport", "recall_ratio.value"),
    "min_group_lift": ("FairnessReport", "min_group_lift.value"),
}
PROPERTY_FILES = [
    {"PropertyFileName": name, "OutputName": "evaluation", "FilePath": file}
    for name, file in (("EvaluationReport", "evaluation.json"), ("FairnessReport", "fairness.json"))
]
MODEL_CARD_URL = "https://github.com/collin-smith/mlops-aiops-pipeline/blob/main/docs/model-card.md"


def model_package_group(project: str) -> str:
    """Must match infra/registry.tf."""
    return f"{project}-breach-risk"


def param(name: str) -> dict:
    return {"Get": f"Parameters.{name}"}


def run_prefix(bucket: str, step: str) -> dict:
    """s3://<bucket>/pipeline-runs/<execution id>/<step>/, resolved when the run starts."""
    return {
        "Std:Join": {
            "On": "/",
            "Values": [
                f"s3://{bucket}/pipeline-runs",
                {"Get": "Execution.PipelineExecutionId"},
                step.lower(),
            ],
        }
    }


def json_get(key: str) -> dict:
    """The gate's value for ``key``, read from the Evaluate step's report at run time."""
    file, path = {**GATE_PATHS, **CHAMPION_PATHS}[key]
    return {
        "Std:JsonGet": {
            "PropertyFile": {"Get": f"Steps.Evaluate.PropertyFiles.{file}"},
            "Path": path,
        }
    }


def output_file(step: str, output: str, file: str) -> dict:
    """The S3 URI of ``file`` inside a step's output, resolved at run time."""
    uri = {"Get": f"Steps.{step}.ProcessingOutputConfig.Outputs['{output}'].S3Output.S3Uri"}
    return {"Std:Join": {"On": "/", "Values": [uri, file]}}


def _gate(*, image_uri: str, group: str) -> dict:
    conditions = [
        {"Type": "GreaterThanOrEqualTo", "LeftValue": json_get(k), "RightValue": v}
        for k, v in GATE.items()
    ]
    conditions.append(
        {
            "Type": "Or",
            "Arguments": {
                "Conditions": [
                    {"Type": "Equals", "LeftValue": json_get("champion_exists"), "RightValue": 0},
                    {
                        "Type": "GreaterThanOrEqualTo",
                        "LeftValue": json_get("champion_margin"),
                        "RightValue": GUARDBAND,
                    },
                ]
            },
        }
    )
    register = {
        "Name": "Register",
        "Type": "RegisterModel",
        "Arguments": {
            "ModelPackageGroupName": group,
            "ModelPackageDescription": {
                "Std:Join": {
                    "On": " ",
                    "Values": [
                        "snapshot",
                        param("AsOf"),
                        "run",
                        {"Get": "Execution.PipelineExecutionId"},
                    ],
                }
            },
            # a person approves it, under a different role (infra/registry.tf)
            "ModelApprovalStatus": "PendingManualApproval",
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": image_uri,
                        "ModelDataUrl": output_file("Train", "model", "model.tar.gz"),
                    }
                ],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            },
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": output_file("Evaluate", "evaluation", "evaluation.json"),
                    }
                },
                "Bias": {
                    "Report": {
                        "ContentType": "application/json",
                        "S3Uri": output_file("Evaluate", "evaluation", "fairness.json"),
                    }
                },
            },
            # what a reviewer needs to trace the version back to its data and code
            "CustomerMetadataProperties": {
                "asof": param("AsOf"),
                "code_uri": param("CodeUri"),
                "pipeline_execution": {"Get": "Execution.PipelineExecutionId"},
                "gate": ";".join(f"{k}>={v}" for k, v in GATE.items())
                + f";beats_champion_by>={GUARDBAND}",
                "model_card": MODEL_CARD_URL,
            },
        },
    }
    values = ["The promotion gate rejected this model. Needed:"]
    values += [f"{k} >= {v}," for k, v in GATE.items()]
    values.append("got:")
    values.append(f"and beats any approved champion by >= {GUARDBAND},")
    for k in GATE:
        values += [f"{k}", json_get(k)]
    values += ["champion_margin", json_get("champion_margin")]
    values.append("(see evaluation.json and fairness.json)")
    rejected = {
        "Name": "Rejected",
        "Type": "Fail",
        "Arguments": {"ErrorMessage": {"Std:Join": {"On": " ", "Values": values}}},
    }
    return {
        "Name": "Gate",
        "Type": "Condition",
        "DependsOn": ["Evaluate"],
        "Arguments": {"Conditions": conditions, "IfSteps": [register], "ElseSteps": [rejected]},
    }


def _input(name: str, uri, local: str) -> dict:
    return {
        "InputName": name,
        "AppManaged": False,
        "S3Input": {
            "S3Uri": uri,
            "LocalPath": local,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
        },
    }


def _step(
    name: str,
    args: list,
    inputs: list[dict],
    output: str,
    *,
    bucket: str,
    image_uri: str,
    role_arn: str,
    env: dict,
    depends_on: list[str],
) -> dict:
    step = {
        "Name": name,
        "Type": "Processing",
        "CacheConfig": CACHE,
        "Arguments": {
            "AppSpecification": {
                "ImageUri": image_uri,
                "ContainerEntrypoint": ENTRYPOINT,
                "ContainerArguments": args,
            },
            "ProcessingInputs": inputs,
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": output,
                        "AppManaged": False,
                        "S3Output": {
                            "S3Uri": run_prefix(bucket, name),
                            "LocalPath": OUT,
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": param("InstanceType"),
                    "VolumeSizeInGB": 5,
                }
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME[name]},
            "RoleArn": role_arn,
            "Environment": env,
        },
    }
    if depends_on:
        step["DependsOn"] = depends_on
    return step


def build_definition(
    *,
    bucket: str,
    region: str,
    project: str,
    metric_namespace: str,
    image_uri: str,
    role_arn: str,
    code_uri: str,
    asof: str,
    instance_type: str = "ml.t3.xlarge",
) -> dict:
    """The pipeline definition. The keyword values become the parameters' defaults."""
    group = model_package_group(project)
    if instance_type not in PROCESSING_INSTANCE_TYPES:
        raise ValueError(
            f"{instance_type} has no Processing quota here; use one of {PROCESSING_INSTANCE_TYPES}"
        )
    data_uri = f"s3://{bucket}/processed/311/"
    env = {
        "PYTHONUNBUFFERED": "1",
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
        "MLOPS_PROJECT": project,
        "MLOPS_METRIC_NAMESPACE": metric_namespace,
        "MLOPS_INSTANCE_TYPE": param("InstanceType"),
        # part of every step's cache key: a new snapshot re-runs the whole pipeline
        "MLOPS_ASOF": param("AsOf"),
    }
    code = _input("code", param("CodeUri"), IN_CODE)
    data = _input("data", data_uri, IN_DATA)
    model_uri = {"Get": "Steps.Train.ProcessingOutputConfig.Outputs['model'].S3Output.S3Uri"}
    # the community lookup frozen with the same snapshot (socrata_pull.py communities)
    communities_uri = {
        "Std:Join": {
            "On": "",
            "Values": [f"s3://{bucket}/raw/communities/asof=", param("AsOf"), "/"],
        }
    }
    communities = _input("communities", communities_uri, IN_COMMUNITIES)
    common = {"bucket": bucket, "image_uri": image_uri, "role_arn": role_arn, "env": env}

    steps = [
        _step(
            "Validate",
            ["validate", "--input", IN_DATA, "--output", OUT, "--asof", param("AsOf")],
            [code, data],
            "validation",
            depends_on=[],
            **common,
        ),
        _step(
            "Train",
            ["train", "--data", IN_DATA, "--communities", IN_COMMUNITIES, "--out", OUT,
             "--max-train-rows", param("MaxTrainRows")],
            [code, data, communities],
            "model",
            depends_on=["Validate"],
            **common,
        ),
        _step(
            "Evaluate",
            ["evaluate", "--model", IN_MODEL, "--data", IN_DATA,
             "--communities", IN_COMMUNITIES, "--out", OUT, "--champion-group", group],
            [code, data, _input("model", model_uri, IN_MODEL), communities],
            "evaluation",
            depends_on=["Train"],
            **common,
        ),
    ]  # fmt: skip
    steps[-1]["PropertyFiles"] = PROPERTY_FILES
    steps.append(_gate(image_uri=image_uri, group=group))
    return {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [
            {"Name": "InstanceType", "Type": "String", "DefaultValue": instance_type},
            {"Name": "AsOf", "Type": "String", "DefaultValue": asof},
            {"Name": "CodeUri", "Type": "String", "DefaultValue": code_uri},
            {"Name": "MaxTrainRows", "Type": "String", "DefaultValue": "0"},
        ],
        "Steps": steps,
    }
