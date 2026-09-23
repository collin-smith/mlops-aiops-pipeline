# infra/ — base Terraform stack

One stack, shared by every stage. Later stages add resources here behind their own
files; they do **not** get their own stack (keeps IAM coherent).

## What Stage 1 creates

| Resource | Purpose |
|---|---|
| S3 `…-datalake-<suffix>` | raw / processed / model-artifacts / scored zones (prefixes) |
| S3 `…-athena-results-<suffix>` | Athena query output, 7-day expiry |
| Athena workgroup | 2 GB per-query scan cap, CloudWatch metrics on |
| Glue Catalog database | `mlops_aiops` |
| Glue crawler `…-processed-311` | on-demand only, no schedule |
| Glue job `…-json-to-parquet` | optional Spark ETL (local `to-parquet` is the default path) |
| IAM `…-sagemaker` | scoped to job-based compute — **no** endpoint permissions |
| IAM `…-glue` | crawler + ETL |
| IAM `…-github-actions` + OIDC provider | Stage 5 retrain workflow |
| SNS `…-alerts` + email sub | Budgets + alarm fan-out |
| CloudWatch log group + `StepFailure` alarm | MLOps-side smoke detector |
| AWS Budgets `…-series-total` | account-wide, cumulative from `series_start`: warn at $15, hard-stop at $25 |
| AWS Budgets `…-account-monthly` | account-wide, monthly: FORECAST warning at $10 |
| Cost Anomaly Detection monitor + subscription | unusual daily spend → alerts topic, immediately (`anomaly.tf`) |
| Budgets **action** | at $25 ACTUAL, auto-attaches a deny policy to the pipeline roles |

## Apply

```bash
cp terraform.tfvars.example terraform.tfvars   # edit
terraform init
terraform apply
```

Then copy the outputs into the repo `.env`:

```bash
terraform output -raw datalake_bucket          # -> MLOPS_BUCKET
terraform output -raw athena_output_location   # -> MLOPS_ATHENA_OUTPUT
terraform output -raw glue_database            # -> MLOPS_GLUE_DATABASE
```

**Confirm the SNS subscription** — AWS emails a confirmation link to `alert_email`;
until you click it the budget/alarm notifications go nowhere.

## The budget hard-stop

`enable_budget_hardstop = true` (default) creates `aws_budgets_budget_action.hardstop`.
When ACTUAL spend for the whole account, cumulative since `series_start`, crosses $25, AWS Budgets
assumes `…-budget-action` and attaches `…-budget-hardstop-deny` to the SageMaker, Glue,
and GitHub-Actions roles. New training / processing / transform / crawler / pipeline
runs are then denied until you detach it:

```bash
for r in mlops-aiops-sagemaker mlops-aiops-glue mlops-aiops-github-actions; do
  aws iam detach-role-policy --role-name "$r" \
    --policy-arn "$(terraform output -raw budget_denyall_arn 2>/dev/null || echo arn:aws:iam::ACCOUNT:policy/mlops-aiops-budget-hardstop-deny)"
done
```

Set `enable_budget_hardstop = false` for alert-only behaviour.

> Budgets data can lag actual spend by several hours. The hard-stop is a backstop,
> not a real-time circuit breaker. `scripts/nuke.sh` after every build session is the
> primary control.

## No VPC (by design — D-020)

This stack provisions no `aws_vpc`, no subnets, no NAT. Every service is regional/managed
and SageMaker jobs run without `VpcConfig`. That is a deliberate, budget-driven choice
(interface endpoints for the hardened pattern would be ~$44/month). The VPC-hardened
variant — private subnets across 2 AZs, VPC endpoints for S3/ECR/SageMaker/CloudWatch/STS,
no IGW/NAT — is in `vpc.tf.example` (rename to `vpc.tf` to use) and drawn in
`../output/stages/architecture-network.drawio`.

## Teardown

```bash
terraform destroy
```

Per-stage teardown deltas are in each stage's `runbook.md`. `terraform destroy` will
not delete a non-empty S3 bucket — empty the data-lake bucket first if you want it gone
(`aws s3 rm s3://<bucket> --recursive`), or keep it (storage is ~$0.03/mo for 1 GB).
