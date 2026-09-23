output "datalake_bucket" {
  description = "Set MLOPS_BUCKET to this."
  value       = aws_s3_bucket.datalake.bucket
}

output "athena_results_bucket" {
  value = aws_s3_bucket.athena_results.bucket
}

output "athena_output_location" {
  description = "Set MLOPS_ATHENA_OUTPUT to this."
  value       = "s3://${aws_s3_bucket.athena_results.bucket}/"
}

output "glue_database" {
  description = "Set MLOPS_GLUE_DATABASE to this."
  value       = aws_glue_catalog_database.main.name
}

output "glue_crawler" {
  value = aws_glue_crawler.processed_311.name
}

output "athena_workgroup" {
  value = aws_athena_workgroup.primary.name
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "github_actions_role_arn" {
  description = "Add as the AWS_ROLE_ARN secret in the GitHub repo (Stage 5)."
  value       = aws_iam_role.github_actions.arn
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}

output "budget_hardstop_enabled" {
  value = var.enable_budget_hardstop
}

output "budget_denyall_arn" {
  description = "Policy the Budgets action attaches at the hard-stop threshold; detach to resume."
  value       = aws_iam_policy.budget_denyall.arn
}
