data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
  name       = var.project

  bucket_name        = "${var.project}-datalake-${var.bucket_suffix}"
  athena_bucket_name = "${var.project}-athena-results-${var.bucket_suffix}"
  glue_database      = replace(var.project, "-", "_")

  metric_namespace = "MLOpsAIOps/Pipeline"
}
