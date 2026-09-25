# Cost log

Running AWS spend receipt for the series. Updated at the end of every build session.
Screenshotted into the Stage 6 article. Target: ~$8 expected, **$25 hard cap** on this project's
tagged spend before credits, plus a $15/$30 account-wide safety net (D-038; enforced
by the Budgets action — see `infra/budgets.tf`).

How to check month-to-date:

```bash
aws ce get-cost-and-usage --region us-east-1 \
  --time-period "Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d)" \
  --granularity MONTHLY --metrics UnblendedCost \
  --filter '{"Tags":{"Key":"project","Values":["mlops-aiops"]}}' \
  --query 'ResultsByTime[0].Total.UnblendedCost'
```

(`scripts/nuke.sh` prints this at the end of each run.)

## Ledger

| Date | Session | Actions | Session est. | MTD actual | Notes |
|---|---|---|---|---|---|
| 2026-09-09 | planning + scaffold | repo scaffold, code, Terraform written. **No AWS resources applied yet.** | $0.00 | $0.00 | Build not started on AWS. |
| 2026-09-09 | exploratory data pass | aggregate SoQL queries + ~230k-row sample from the public Socrata API; local separability check. See `docs/exploratory-findings.md`. | $0.00 | $0.00 | Public open data, no auth, no AWS. |
| 2026-09-23 | Stage 1 build | Terraform base stack applied (33 resources, three applies: clock skew, then the existing anomaly monitor), 2,911,486-row pull (local), Parquet (local), 147 MB snapshot to S3, 2 crawler runs (the second after the `table_prefix` fix), 3 Athena queries (≈12 MB scanned, 30 MB billed), 4 Cost Explorer queries. | ≈ $0.35 (2 crawls ≈ $0.30 at the 10-min minimum; Cost Explorer $0.04; S3 and Athena < $0.01) | **$0.25 usage** (before credits; confirmed 2026-09-24 by service: Cost Explorer $0.11, Glue $0.09, S3 $0.05) | The account is on the AWS Free plan: usage is offset by credits, so the net shows $0.00. The account also had ≈ $3.58 of unrelated usage since 09-01 (an earlier EKS/Bedrock experiment). `nuke.sh --check` CLEAN. |

## Budget by stage (from `output/04-cost-and-serverless.md`)

| Line | Est. |
|---|---|
| Socrata pull (local) | < $0.50 |
| Glue crawler (~6 runs) | ~$0.20 |
| SageMaker Processing (~20, on-demand — Processing has no spot) | ~$0.60 |
| SageMaker Training XGBoost (~20, spot) | ~$0.80 |
| Batch Transform (~10) | ~$0.30 |
| Model Monitor (~8, then deleted) | ~$1.00 |
| RCF train + score (~3) | ~$0.30 |
| S3 storage (~1 GB, 2 mo) | ~$0.10 |
| Athena (~50 queries, partitioned) | < $0.25 |
| CloudWatch custom metrics (~8) | ~$3.00 |
| Serverless Inference demo (Stage 6, once, D-029) | < $0.05 |
| Civic scorecard (Stage 7, S3 CSV, not metrics — D-028) | ~$0.00 |
| **Expected total** | **~$8** |

CloudWatch custom metrics ($0.30/metric/month) are the largest line — keep the metric
count near 8 and delete the namespace usage after Stage 8.
