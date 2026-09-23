# notebooks/

Stage 2 lives here:

- `01_explore_311.ipynb` — monthly request volume by category, seasonal decomposition.
  The shareable civic insight (snow/ice, weeds, cart/waste each have distinct annual
  curves; backlog spikes lead breach months).
- `02_baseline_model.ipynb` — Athena → CSV in S3 → SageMaker Training Job (XGBoost
  built-in) on `ml.m5.large` spot. Time-based split. Reports ROC-AUC + PR-AUC.

Run on a SageMaker Studio space with **1-hour idle auto-shutdown**, or launch the jobs
via the SDK from local. Never leave a notebook app running — see `docs/cost-log.md`.
