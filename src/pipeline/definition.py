"""Stage 3: the training pipeline as a SageMaker Pipelines definition.

Three Processing steps, in order (D-039: the account has no Training Job quota):

    Validate  the data-validation gate (D-034); a failed check stops the run before any
              training spend
    Train     the Stage 2 model; writes model.tar.gz and metrics.json
    Evaluate  scores the test year with the saved artifact and writes evaluation.json

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

IN_CODE = "/opt/ml/processing/input/code"
IN_DATA = "/opt/ml/processing/input/data"
IN_MODEL = "/opt/ml/processing/input/model"
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
            ["train", "--data", IN_DATA, "--out", OUT, "--max-train-rows", param("MaxTrainRows")],
            [code, data],
            "model",
            depends_on=["Validate"],
            **common,
        ),
        _step(
            "Evaluate",
            ["evaluate", "--model", IN_MODEL, "--data", IN_DATA, "--out", OUT],
            [code, data, _input("model", model_uri, IN_MODEL)],
            "evaluation",
            depends_on=["Train"],
            **common,
        ),
    ]
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
