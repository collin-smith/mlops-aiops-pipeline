# --- Shadow scoring (Stage 6, D-044) ---
#
# The Stage 5 challenger failed the fairness gate, so it can't go in the registry's main
# group. It still scores real requests, labelled not-for-use, so Stage 7 can measure how it
# would have done. It is registered here, in a group of its own, so the registry still lists
# every model that has scored a real request. Nothing in this group is ever approved: no
# project role may change a version's status here. A group costs nothing while it sits there.

resource "aws_sagemaker_model_package_group" "shadow" {
  # Must match src/deploy/shadow.py shadow_group()
  model_package_group_name        = "${local.name}-breach-risk-shadow"
  model_package_group_description = "Shadow models: score real requests labelled shadow-not-for-use, for Stage 7 to grade. Registered by hand as Rejected by the gate; never approved."
}

locals {
  shadow_versions_arn = "arn:${local.partition}:sagemaker:${var.region}:${local.account_id}:model-package/${aws_sagemaker_model_package_group.shadow.model_package_group_name}/*"
}

# The approver's allow only covers the main group's versions already. The deny states the
# rule, and keeps it if that allow is ever widened. (The training role is denied
# UpdateModelPackage everywhere, and the CI role was never granted it: registry.tf, iam.tf.)
resource "aws_iam_role_policy" "approver_no_shadow" {
  name = "never-approve-shadow"
  role = aws_iam_role.approver.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "ShadowIsNeverApproved"
      Effect   = "Deny"
      Action   = ["sagemaker:UpdateModelPackage"]
      Resource = [local.shadow_versions_arn]
    }]
  })
}

# scored/asof=<YYYY-MM-DD>/scores-<model tag>.parquet, written by the scoring job.
# Partition projection: a new snapshot date is queryable without a crawler or MSCK.
resource "aws_glue_catalog_table" "shadow_scores" {
  name          = "shadow_scores"
  database_name = aws_glue_catalog_database.main.name
  description   = "Shadow scores. NOT FOR USE: every row says usage = shadow-not-for-use (D-044)."
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    "classification"                = "parquet"
    "projection.enabled"            = "true"
    "projection.asof.type"          = "date"
    "projection.asof.format"        = "yyyy-MM-dd"
    "projection.asof.range"         = "2026-09-01,NOW"
    "projection.asof.interval"      = "1"
    "projection.asof.interval.unit" = "DAYS"
    "storage.location.template"     = "s3://${aws_s3_bucket.datalake.bucket}/scored/asof=$${asof}/"
  }

  partition_keys {
    name = "asof"
    type = "string"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.datalake.bucket}/scored/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }

    # Must match src/deploy/score.py OUTPUT_COLUMNS, in order
    columns {
      name = "usage"
      type = "string"
    }
    columns {
      name = "service_request_id"
      type = "string"
    }
    columns {
      name = "service_name"
      type = "string"
    }
    columns {
      name = "comm_code"
      type = "string"
    }
    columns {
      name = "comm_name"
      type = "string"
    }
    columns {
      name = "sector"
      type = "string"
    }
    columns {
      name = "srg"
      type = "string"
    }
    columns {
      name = "requested_date"
      type = "timestamp"
    }
    columns {
      name = "deadline"
      type = "timestamp"
    }
    columns {
      name = "threshold_days"
      type = "double"
    }
    columns {
      name = "category_seen"
      type = "boolean"
    }
    columns {
      name = "score"
      type = "float"
    }
    columns {
      name = "rank_in_sector"
      type = "bigint"
    }
    columns {
      name = "flagged"
      type = "boolean"
    }
    columns {
      name = "model_ref"
      type = "string"
    }
    columns {
      name = "snapshot_asof"
      type = "string"
    }
    columns {
      name = "scored_at"
      type = "string"
    }
  }
}

output "shadow_model_package_group" {
  value = aws_sagemaker_model_package_group.shadow.model_package_group_name
}
