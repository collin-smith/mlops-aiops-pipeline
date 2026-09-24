resource "aws_glue_catalog_database" "main" {
  name        = local.glue_database
  description = "Calgary 311 lake for the mlops-aiops-pipeline series"
}

# --- Crawler role ---

resource "aws_iam_role" "glue" {
  name = "${local.name}-glue"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "glue.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy" "glue_s3" {
  name = "s3-datalake"
  role = aws_iam_role.glue.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = ["${aws_s3_bucket.datalake.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [aws_s3_bucket.datalake.arn]
      },
    ]
  })
}

# --- Crawler over the processed (Parquet) zone ---

resource "aws_glue_crawler" "processed_311" {
  name          = "${local.name}-processed-311"
  role          = aws_iam_role.glue.arn
  database_name = aws_glue_catalog_database.main.name
  description   = "Catalogs processed/311 partitioned Parquet. Run on demand; no schedule."
  # Glue names the table after the leaf folder, so without a prefix the table is `311`.
  # A name starting with a digit must be double-quoted in every Athena query.
  table_prefix = "processed_"

  s3_target {
    path = "s3://${aws_s3_bucket.datalake.bucket}/processed/311/"
  }

  # No `schedule` block -> on-demand only (cost + teardown discipline).

  schema_change_policy {
    delete_behavior = "LOG"
    update_behavior = "UPDATE_IN_DATABASE"
  }

  configuration = jsonencode({
    Version  = 1.0
    Grouping = { TableLevelConfiguration = 3 } # bucket / processed / 311  -> one table
  })
}

# --- Optional Spark ETL job (the "pure-AWS" path; local to-parquet is the default) ---

resource "aws_glue_job" "json_to_parquet" {
  name              = "${local.name}-json-to-parquet"
  role_arn          = aws_iam_role.glue.arn
  glue_version      = "4.0"
  worker_type       = "G.1X"
  number_of_workers = 2
  max_retries       = 0
  timeout           = 20 # minutes

  command {
    name            = "glueetl"
    script_location = "s3://${aws_s3_bucket.datalake.bucket}/glue-scripts/glue_job.py"
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"        = "python"
    "--enable-metrics"      = "true"
    "--PROCESSED_PATH"      = "s3://${aws_s3_bucket.datalake.bucket}/processed/311/"
    "--TempDir"             = "s3://${aws_s3_bucket.datalake.bucket}/glue-tmp/"
    "--enable-job-insights" = "false"
  }
}
