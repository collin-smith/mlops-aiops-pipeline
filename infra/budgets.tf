# Cost guardrail (D-011, D-031, reworked by D-038). Two budgets:
#
#   1. series_total — THIS PROJECT: spend tagged project=<var.project>, cumulative from
#      var.series_start for a year, so it never resets mid-series. Carries the $15 warning
#      and the Budgets ACTION that attaches a deny policy to the SageMaker + Glue +
#      GitHub-Actions roles at $25. Tag-filtered since D-038, because the account is
#      shared with another project (StudySite) whose spend shouldn't trip this stop, and
#      this stop can't halt that project anyway.
#   2. account_monthly — THE WHOLE ACCOUNT, monthly: the safety net. Emails at $15 and $30
#      ACTUAL and at the $10 FORECAST (D-030), with no action. It catches what the tag
#      filter can't see: costs that can't carry a tag (CloudWatch custom metrics, Cost
#      Explorer queries, data transfer), anything someone forgot to tag, and every other
#      project in the account. It's the only account-wide budget: StudySite keeps just a
#      tag-filtered one (see the shared-account cost guide).
#
# Both count usage BEFORE credits (include_credit = false). The account is on the AWS Free
# plan, which records usage as a charge plus an equal negative credit, so a budget on the
# default setting reads $0 until the credits run out and could never fire.
#
# Forecasts need a few weeks of spend history, so early in the series the ACTUAL alerts
# and `nuke.sh --check` are the controls that matter.
#
# Set enable_budget_hardstop = false for alerts only.

resource "aws_budgets_budget" "series_total" {
  name              = "${local.name}-series-total"
  budget_type       = "COST"
  limit_amount      = tostring(var.budget_hardstop_usd)
  limit_unit        = "USD"
  time_unit         = "ANNUALLY"
  time_period_start = var.series_start

  # Only this project's tagged spend. format() rather than "...$${...}", which HCL would
  # read as an escaped interpolation.
  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:project$%s", var.project)]
  }

  cost_types {
    include_credit = false
  }

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
  limit_amount = tostring(var.account_alert_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_types {
    include_credit = false
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = var.account_warn_usd
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = var.account_alert_usd
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.alerts.arn]
  }

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
  # (Out of date: it's the series-total stop since D-031. Left as is because changing an IAM
  # policy's description forces Terraform to destroy and recreate the policy.)
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
