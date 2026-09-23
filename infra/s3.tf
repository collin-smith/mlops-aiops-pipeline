# One data-lake bucket; the "zones" from the plan are prefixes within it:
#   raw/311/asof=YYYY-MM-DD/      immutable Socrata extracts (frozen snapshots)
#   processed/311/year=/month=/   partitioned Parquet, crawled into the Catalog
#   model-artifacts/              SageMaker model.tar.gz outputs
#   scored/                       Batch Transform output (Stage 6)

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
