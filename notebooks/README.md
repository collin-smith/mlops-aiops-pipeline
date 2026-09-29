# notebooks/

Stage 2 lives here. Both notebooks run **locally** on the Stage 1 snapshot, with no AWS
calls and no cost, so every number in the article can be reproduced:

- `01_explore_311.ipynb`: how each category's requests spread over the year (snow and
  ice, weeds, potholes and carts have very different curves), and whether a category's
  busy months are also its late months. That's true for some categories and the reverse
  for others. Charts are saved to `figures/01-*.png`.
- `02_baseline_model.ipynb`: the censoring-aware breach label, intake-time features, the
  leakage guard, and a local XGBoost baseline (ROC-AUC 0.66; the riskiest tenth runs late
  2× as often as requests overall; D-037). Chart: `figures/02-late-by-risk-decile.png`.
  Its last section, the same model as a SageMaker Training Job, is added once the
  account's training quota is approved.

To run them, you need the Stage 1 snapshot under `code/data/`. Either pull it
(`uv run python -m src.ingest.socrata_pull pull`, then `to-parquet`) or restore a frozen one from S3:

```bash
uv run python -m src.ingest.socrata_pull replay --asof 2026-09-23
uv sync --extra ml
uv run --with jupyter jupyter lab notebooks/
```

The model notebook takes about 3 minutes on a laptop for 2.9 million requests.

If you run anything on SageMaker Studio instead, use a space with **1-hour idle
auto-shutdown**, and never leave a notebook app running. See `docs/cost-log.md`.
