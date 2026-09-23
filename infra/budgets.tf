# Cost guardrail (D-011, reworked by D-031). Two budgets, both inside the free tier:
#
#   1. series_total — ACCOUNT-WIDE (no tag filter), cumulative from var.series_start for a
#      year, so it never resets mid-series. Carries the $15 warning and the Budgets ACTION
#      that attaches a deny policy to the SageMaker + Glue + GitHub-Actions roles at $25.
#      Account-wide on purpose: some costs can't carry the project tag (CloudWatch custom
#      metrics), SDK-launched jobs only carry it if the code passes Tags, and the tag
#      counts nothing until it is activated as a cost-allocation tag. The account is
#      dedicated to this series (plus a near-$0 static site), so account spend is a safe,
#      slightly conservative proxy. The deny policy only touches this project's roles.
#   2. account_monthly — also account-wide, monthly, carrying the early FORECAST warning
#      at $10 (D-030). Monthly because a forecast on the annual series_total budget would
#      extrapolate a whole year and fire on noise. Per-project spend is read from Cost
#      Explorer's tag filter instead (scripts/nuke.sh prints it), which needs the `project`
#      tag activated (Stage 1 runbook step 2).
#
# Forecasts need a few weeks of spend history, so early in the series the $15 ACTUAL
# warning and `nuke.sh --check` are the controls that matter.
#
# Set enable_budget_hardstop = false for alerts only.

resource "aws_budgets_budget" "series_total" {
  name              = "${local.name}-series-total"
  budget_type       = "COST"
  limit_amount      = tostring(var.budget_hardstop_usd)
  limit_unit        = "USD"
  time_unit         = "ANNUALLY"
  time_period_start = var.series_start

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100 * var.budget_warn_usd / var.budget_hardstop_usd
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }
}

resource "aws_budgets_budget" "account_monthly" {
  name         = "${local.name}-account-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_hardstop_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = var.budget_forecast_warn_usd
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }
}

# --- The deny policy the action attaches ---

resource "aws_iam_policy" "budget_denyall" {
  name        = "${local.name}-budget-hardstop-deny"
  description = "Attached by AWS Budgets when the monthly hard-stop threshold is crossed."
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "DenyExpensiveComputeAtBudgetCap"
      Effect = "Deny"
      Action = [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateTransformJob",
        "sagemaker:CreateEndpoint",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:CreateMonitoringSchedule",
        "sagemaker:StartPipelineExecution",
        "glue:StartJobRun",
        "glue:StartCrawler",
      ]
      Resource = "*"
    }]
  })
}

# --- Role that Budgets assumes to perform the action ---

resource "aws_iam_role" "budget_action" {
  count = var.enable_budget_hardstop ? 1 : 0
  name  = "${local.name}-budget-action"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "budgets.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = {
        StringEquals = { "aws:SourceAccount" = local.account_id }
      }
    }]
  })
}

resource "aws_iam_role_policy" "budget_action" {
  count = var.enable_budget_hardstop ? 1 : 0
  name  = "attach-deny-policy"
  role  = aws_iam_role.budget_action[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["iam:AttachRolePolicy", "iam:DetachRolePolicy"]
      Resource = [
        aws_iam_role.sagemaker.arn,
        aws_iam_role.glue.arn,
        aws_iam_role.github_actions.arn,
      ]
      Condition = {
        ArnEquals = { "iam:PolicyARN" = aws_iam_policy.budget_denyall.arn }
      }
    }]
  })
}

resource "aws_budgets_budget_action" "hardstop" {
  count              = var.enable_budget_hardstop ? 1 : 0
  budget_name        = aws_budgets_budget.series_total.name
  action_type        = "APPLY_IAM_POLICY"
  approval_model     = "AUTOMATIC"
  notification_type  = "ACTUAL"
  execution_role_arn = aws_iam_role.budget_action[0].arn

  action_threshold {
    action_threshold_type  = "ABSOLUTE_VALUE"
    action_threshold_value = var.budget_hardstop_usd
  }

  definition {
    iam_action_definition {
      policy_arn = aws_iam_policy.budget_denyall.arn
      roles = [
        aws_iam_role.sagemaker.name,
        aws_iam_role.glue.name,
        aws_iam_role.github_actions.name,
      ]
    }
  }

  subscriber {
    address           = var.alert_email
    subscription_type = "EMAIL"
  }

  subscriber {
    address           = aws_sns_topic.alerts.arn
    subscription_type = "SNS"
  }
}
