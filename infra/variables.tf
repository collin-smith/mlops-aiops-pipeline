variable "region" {
  type    = string
  default = "ca-central-1"
}

variable "project" {
  description = "Resource name prefix and cost-allocation tag value. Must match MLOPS_PROJECT."
  type        = string
  default     = "mlops-aiops"
}

variable "bucket_suffix" {
  description = "Random-ish suffix to make the S3 bucket name globally unique (e.g. your initials + 4 digits)."
  type        = string
}

variable "alert_email" {
  description = "Address that receives Budgets + CloudWatch alarm notifications."
  type        = string
}

variable "github_repo" {
  description = "owner/name of the GitHub repo allowed to assume the OIDC deploy role."
  type        = string
  default     = "collin-smith/mlops-aiops-pipeline"
}

variable "budget_warn_usd" {
  description = "Series-total ACTUAL spend tagged to this project that triggers the warning email."
  type        = number
  default     = 15
}

variable "budget_hardstop_usd" {
  description = "At this series-total ACTUAL spend tagged to this project, the Budgets action attaches a deny policy to the pipeline roles."
  type        = number
  default     = 25
}

variable "account_warn_usd" {
  description = "Whole-account ACTUAL spend this month (before credits) that triggers the first safety-net email."
  type        = number
  default     = 15
}

variable "account_alert_usd" {
  description = "Whole-account ACTUAL spend this month (before credits) that triggers the second safety-net email. Also the budget's limit."
  type        = number
  default     = 30
}

variable "budget_forecast_warn_usd" {
  description = "Warn when this month's FORECAST account spend passes this (D-030)."
  type        = number
  default     = 10
}

variable "existing_anomaly_monitor_arn" {
  description = "Leave empty to create a Cost Anomaly Detection services monitor. AWS allows one per account; if the account already has one (e.g. AWS's default), put its ARN here."
  type        = string
  default     = ""
}

variable "series_start" {
  description = "Start of the cumulative series budget (YYYY-MM-DD_HH:MM, UTC). First of the month the build starts; spend before it isn't counted."
  type        = string
  default     = "2026-09-01_00:00"
}

variable "enable_budget_hardstop" {
  description = "false = alerts only. true = auto-attach the deny policy at the hard-stop threshold."
  type        = bool
  default     = true
}
