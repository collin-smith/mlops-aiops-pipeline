resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/${var.project}/pipeline"
  retention_in_days = 30
}

# --- Alarm + Budgets notification fan-out ---

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# Budgets and Cost Anomaly Detection publish to SNS; they need permission to do so.
resource "aws_sns_topic_policy" "alerts" {
  arn = aws_sns_topic.alerts.arn
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowBudgets"
        Effect    = "Allow"
        Principal = { Service = "budgets.amazonaws.com" }
        Action    = "SNS:Publish"
        Resource  = aws_sns_topic.alerts.arn
      },
      {
        Sid       = "AllowCostAnomalyDetection"
        Effect    = "Allow"
        Principal = { Service = "costalerts.amazonaws.com" }
        Action    = "SNS:Publish"
        Resource  = aws_sns_topic.alerts.arn
      },
    ]
  })
}

# Pipeline step-failure alarm. src/common/metrics.py emits StepFailure (0/1) per stage
# from Stage 3 on; this alarm is the MLOps-side smoke detector. Stage 7 adds the
# AIOps-side anomaly detection on top of the same namespace.
resource "aws_cloudwatch_metric_alarm" "step_failure" {
  alarm_name          = "${local.name}-pipeline-step-failure"
  namespace           = local.metric_namespace
  metric_name         = "StepFailure"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_description   = "A SageMaker pipeline step reported failure."
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}
