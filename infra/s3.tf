# One data-lake bucket; the "zones" from the plan are prefixes within it:
#   raw/311/asof=YYYY-MM-DD/      immutable Socrata extracts (frozen snapshots)
#   processed/311/year=/month=/   partitioned Parquet, crawled into the Catalog
#   model-artifacts/              SageMaker model.tar.gz outputs
#   scored/asof=YYYY-MM-DD/       shadow scores, not for use (Stage 6, D-044; table shadow_scores)
#   scored-reports/<job>/         each scoring run's summary
#   pipeline-runs/<exec>/<step>/  each pipeline run's outputs (registered models point here)
#   monitoring/...                Stage 7 inputs, baselines, checks, shadow grades
#   code/, glue-tmp/              job code uploads and Glue scratch
#
# Scratch expires after local.scratch_days (D-046): code/ (re-uploaded on every run), glue-tmp/, and
# the monitoring inputs and check outputs (recreated by scripts/monitor_job.py). Snapshots,
# pipeline runs, scores, baselines and grades are kept: they are the history and the lineage.
locals {
  scratch_days     = 90
  scratch_prefixes = [
    "code/",
    "glue-tmp/",
    "monitoring/datasets/",
    "monitoring/injected/",
    "monitoring/checks/",
  ]
}

resource "aws_s3_bucket" "datalake" {
  bucket = local.bucket_name
}

resource "aws_s3_bucket_versioning" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "datalake" {
  bucket                  = aws_s3_bucket.datalake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "datalake" {
  bucket = aws_s3_bucket.datalake.id

  rule {
    id     = "expire-old-noncurrent-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
    # versioned bucket: once a deleted object's old versions expire, drop its delete marker
    expiration {
      expired_object_delete_marker = true
    }
  }

  dynamic "rule" {
    for_each = local.scratch_prefixes
    content {
      id     = "expire-scratch-${trimsuffix(replace(rule.value, "/", "-"), "-")}"
      status = "Enabled"
      filter {
        prefix = rule.value
      }
      expiration {
        days = local.scratch_days
      }
    }
  }
}

# --- Athena query results (short-lived; expire fast) ---

resource "aws_s3_bucket" "athena_results" {
  bucket = local.athena_bucket_name
}

resource "aws_s3_bucket_public_access_block" "athena_results" {
  bucket                  = aws_s3_bucket.athena_results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "athena_results" {
  bucket = aws_s3_bucket.athena_results.id
  rule {
    id     = "expire-query-results"
    status = "Enabled"
    filter {}
    expiration {
      days = 7
    }
  }
}

resource "aws_athena_workgroup" "primary" {
  name = var.project

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true
    bytes_scanned_cutoff_per_query     = 2 * 1024 * 1024 * 1024 # 2 GB guardrail

    result_configuration {
      output_location = "s3://${aws_s3_bucket.athena_results.bucket}/"
      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
}
