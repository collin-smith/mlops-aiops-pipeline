# --- SageMaker execution role (Stages 2-7) ---

resource "aws_iam_role" "sagemaker" {
  name = "${local.name}-sagemaker"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Deliberately NOT AmazonSageMakerFullAccess. Scoped to job-based compute only —
# there is no CreateEndpoint / CreateEndpointConfig here, on purpose (D-007; ci.yml also
# greps for it). The one Stage 6 Serverless demo runs under the operator's credentials
# and uses this role only as the model's execution role (D-029).
resource "aws_iam_role_policy" "sagemaker" {
  name = "pipeline-jobs"
  role = aws_iam_role.sagemaker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "JobBasedComputeOnly"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:CreateProcessingJob",
          "sagemaker:CreateTransformJob",
          "sagemaker:CreateModel",
          "sagemaker:CreateModelPackage",
          "sagemaker:CreateModelPackageGroup",
          "sagemaker:UpdateModelPackage",
          "sagemaker:CreatePipeline",
          "sagemaker:UpdatePipeline",
          "sagemaker:StartPipelineExecution",
          "sagemaker:CreateMonitoringSchedule",
          "sagemaker:DeleteMonitoringSchedule",
          "sagemaker:Describe*",
          "sagemaker:List*",
          "sagemaker:AddTags",
          "sagemaker:Search",
        ]
        Resource = "*"
      },
      {
        Sid      = "DataLake"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = ["${aws_s3_bucket.datalake.arn}/*"]
      },
      {
        Sid      = "DataLakeList"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [aws_s3_bucket.datalake.arn]
      },
      {
        Sid      = "Logs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
        Resource = "arn:${local.partition}:logs:*:${local.account_id}:log-group:/aws/sagemaker/*"
      },
      {
        Sid      = "CostTelemetry"
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
        Condition = {
          StringEquals = { "cloudwatch:namespace" = local.metric_namespace }
        }
      },
      {
        Sid      = "EcrPullBuiltins"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
        Resource = "*"
      },
      {
        Sid      = "PassSelf"
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = [aws_iam_role.sagemaker.arn]
        Condition = {
          StringEquals = { "iam:PassedToService" = "sagemaker.amazonaws.com" }
        }
      },
      {
        Sid      = "AthenaRead"
        Effect   = "Allow"
        Action   = ["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults", "athena:GetWorkGroup", "glue:GetTable", "glue:GetPartitions", "glue:GetDatabase"]
        Resource = "*"
      },
    ]
  })
}

# --- GitHub Actions OIDC deploy role (Stage 5) ---
#
# If this account already has a GitHub OIDC provider, `apply` fails with
# EntityAlreadyExists. Import it instead:
#   terraform import aws_iam_openid_connect_provider.github \
#     arn:aws:iam::<account>:oidc-provider/token.actions.githubusercontent.com

data "tls_certificate" "github" {
  url = "https://token.actions.githubusercontent.com/.well-known/openid-configuration"
}

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = data.tls_certificate.github.certificates[*].sha1_fingerprint
}

resource "aws_iam_role" "github_actions" {
  name = "${local.name}-github-actions"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = { "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com" }
        StringLike   = { "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*" }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_actions" {
  name = "trigger-pipeline"
  role = aws_iam_role.github_actions.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sagemaker:StartPipelineExecution", "sagemaker:Describe*", "sagemaker:List*", "sagemaker:UpdateModelPackage", "sagemaker:Search"]
        Resource = "*"
      },
      {
        Effect    = "Allow"
        Action    = ["iam:PassRole"]
        Resource  = [aws_iam_role.sagemaker.arn]
        Condition = { StringEquals = { "iam:PassedToService" = "sagemaker.amazonaws.com" } }
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.datalake.arn, "${aws_s3_bucket.datalake.arn}/*"]
      },
    ]
  })
}
