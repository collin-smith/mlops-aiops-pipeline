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
| 2026-09-28 | SageMaker smoke test (D-039) | 2 Processing jobs, 119 s billed each (ml.t3.medium, then ml.t3.xlarge), built-in XGBoost image, a 1 KB probe script in S3. | ≈ $0.01 | not yet in Cost Explorer | The Training Job quota was denied, so training moves to Processing on ml.t3 (D-039). A job this short is almost all start-up: expect ~2 min billed per job before any work. |
| 2026-09-28 | Stage 2 training on SageMaker (D-039) | 1 Processing job, ml.t3.xlarge, 239 s billed, full 2.06M-row training set; src/ (~100 KB) uploaded to code/, model.tar.gz (3.3 MB) written to model-artifacts/. | ≈ $0.015 | not yet in Cost Explorer | 2.04× top-decile lift. Start-up plus the pip install took about 100 s of the 239. |
| 2026-09-29 | Stage 3 pipeline, first run (D-040) | Pipeline `mlops-aiops-train` created; run `clk3bbv23tm2`: 3 Processing jobs on ml.t3.xlarge, 89 + 249 + 94 = 432 s billed; src/ uploaded to code/45a76756e827/. Then an identical run (`x4jspzywfbrc`): all 3 steps cache hits, 0 s billed. | ≈ $0.026 | not yet in Cost Explorer | Each step emits 3 custom metrics (StepFailure, RunDurationSeconds, RunCostUsd), so 9 in all. CloudWatch prorates custom metrics by the hour they're sent. |
| 2026-09-30 | Stage 4 gate + registry, first run (D-041) | `terraform apply`: model package group, approver role + policy, no-self-approval policy; training and CI roles lose `UpdateModelPackage`. IAM simulator check (free). Run `wm2m10zouect`: 3 Processing jobs on ml.t3.xlarge, 89 + 249 + 114 = 452 s billed, then Gate (False) and Rejected; nothing registered. | ≈ $0.028 | project tag $0.15 (lags) | The registry, the gate, and the Fail step bill nothing. Full re-run because the code hash changed. |
| 2026-10-01 | Stage 6 shadow scoring (D-044) | `terraform apply`: shadow model package group, approver deny, Glue table `shadow_scores` (3 added, all free). Shadow version 1 registered (free; two refused requests first, $0). Fresh snapshot 2026-10-01: 2,921,625 rows pulled locally, uploaded to S3 (the first `snapshot` attempt hit the placeholder bucket and wrote nothing), 107 stale objects removed; 2 crawler runs. Scoring job `mlops-aiops-score-20261001-235814`: ml.t3.xlarge, 104 s billed. View `shadow_outcomes` created (DDL, free). ★ Athena query: 155.75 KB scanned (10 MB billed minimum). Serverless demo: two attempts, both Failed before serving (the image can't write `/etc` on Serverless), so nothing billed; dropped (D-044). | ≈ $0.03 (scoring $0.0064; crawlers ≈ $0.02; S3 and Athena < $0.01) | not yet in Cost Explorer | One crawler run only re-read the old snapshot (the upload had failed); harmless. |
| 2026-09-30 | Stage 5 first automated retrain (D-042, D-043) | Local challenger experiments ($0). `terraform apply` ×2: CI role permissions, then its OIDC trust (immutable subject). GitHub Actions retrain runs: #1 failed at OIDC sign-in, #2 at `UpdatePipeline` validation (no jobs started, $0); #3 run `vhhhe5vtr34a`: 89 + 219 + 119 = 427 s billed on ml.t3.xlarge, gate False (fairness), nothing registered. | ≈ $0.026 | not yet in Cost Explorer | GitHub Actions minutes are free on a public repo. No new CloudWatch metrics (D-043). |

## Budget by stage (from `output/04-cost-and-serverless.md`)

| Line | Est. |
|---|---|
| Socrata pull (local) | < $0.50 |
| Glue crawler (~6 runs) | ~$0.20 |
| SageMaker Processing (~20, on-demand — Processing has no spot) | ~$0.60 |
| SageMaker training as Processing on ml.t3.xlarge (~20, on-demand; D-039, no spot) | ~$1.00 |
| Batch scoring as Processing (~10; D-039, replaces Batch Transform) | ~$0.30 |
| Model Monitor (~8, then deleted) | ~$1.00 |
| Pipeline anomaly check (EWMA in Lambda; RCF needs a Training Job, D-039) | ~$0.00 |
| S3 storage (~1 GB, 2 mo) | ~$0.10 |
| Athena (~50 queries, partitioned) | < $0.25 |
| CloudWatch custom metrics (~8) | ~$3.00 |
| Serverless Inference demo (Stage 6, D-029): tried twice, never served, dropped (D-044) | $0.00 |
| Civic scorecard (Stage 7, S3 CSV, not metrics — D-028) | ~$0.00 |
| **Expected total** | **~$8** |

CloudWatch custom metrics ($0.30/metric/month) are the largest line — keep the metric
count near 8 and delete the namespace usage after Stage 8.
