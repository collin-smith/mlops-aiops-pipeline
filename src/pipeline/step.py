"""Run one pipeline step and report it to CloudWatch (Stage 3).

Every Processing step in the pipeline runs through here:

    python -m src.pipeline.step validate --input ... --output ...

It calls the step's own ``main()`` inside ``timed_stage``, so each run leaves a
``StepFailure`` (0 or 1), a duration and a cost estimate in the ``MLOpsAIOps/Pipeline``
namespace, with the step's name as the ``Stage`` dimension. The Stage 1 alarm watches
``StepFailure``; Stage 7's anomaly detection reads the rest. A step that raises exits
non-zero, which fails the Processing job and stops the pipeline.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

from src.common.metrics import timed_stage

STEPS = {
    "validate": "src.pipeline.validate",
    "train": "src.pipeline.train",
    "evaluate": "src.pipeline.evaluate",
}

log = logging.getLogger("step")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in STEPS:
        raise SystemExit(f"usage: python -m src.pipeline.step {{{','.join(STEPS)}}} [args...]")
    name, rest = argv[0], argv[1:]
    module = importlib.import_module(STEPS[name])
    instance_type = os.environ.get("MLOPS_INSTANCE_TYPE", "local")
    try:
        with timed_stage(name, instance_type=instance_type):
            module.main(rest)
    except Exception as e:
        log.exception("%s failed: %s", name, e)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
