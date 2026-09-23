# Cost Anomaly Detection (D-030): free. Flags an unusual daily spend jump, such as a
# forgotten schedule or a job that never ended, and pages the alerts topic immediately
# rather than waiting for the $15 budget warning. Like budget forecasts, it needs some
# spend history before it has a baseline to compare against.
#
# AWS allows one AWS-services monitor per account. New accounts often already have AWS's
# default one; if `apply` fails with a limit error, set existing_anomaly_monitor_arn to
# its ARN (Billing -> Cost Anomaly Detection -> Cost monitors).

resource "aws_ce_anomaly_monitor" "services" {
  count             = var.existing_anomaly_monitor_arn == "" ? 1 : 0
  name              = "${local.name}-services"
  monitor_type      = "DIMENSIONAL"
  monitor_dimension = "SERVICE"
}

locals {
  anomaly_monitor_arn = (
    var.existing_anomaly_monitor_arn != ""
    ? var.existing_anomaly_monitor_arn
    : aws_ce_anomaly_monitor.services[0].arn
  )
}

resource "aws_ce_anomaly_subscription" "alerts" {
  name             = "${local.name}-anomalies"
  frequency        = "IMMEDIATE" # IMMEDIATE requires an SNS subscriber
  monitor_arn_list = [local.anomaly_monitor_arn]

  subscriber {
    type    = "SNS"
    address = aws_sns_topic.alerts.arn
  }

  # At this project's scale, a $1 unexpected impact is worth a look.
  threshold_expression {
    dimension {
      key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
      match_options = ["GREATER_THAN_OR_EQUAL"]
      values        = ["1"]
    }
  }

  depends_on = [aws_sns_topic_policy.alerts]
}
