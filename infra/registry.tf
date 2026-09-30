# --- Model Registry and the approver role (Stage 4, D-041) ---
#
# The pipeline's Gate step registers a model that passes as a new version in this group,
# with status PendingManualApproval. Only the approver role can change that status. The
# training role is explicitly denied it, and can't register a version as already Approved.
# A model package group costs nothing while it sits there.

resource "aws_sagemaker_model_package_group" "breach_risk" {
  # Must match src/pipeline/definition.py model_package_group()
  model_package_group_name        = "${local.name}-breach-risk"
  model_package_group_description = "311 SLA-breach triage model. Versions arrive PendingManualApproval from the pipeline's gate; only the approver role approves."
}

locals {
  model_package_versions_arn = "arn:${local.partition}:sagemaker:${var.region}:${local.account_id}:model-package/${aws_sagemaker_model_package_group.breach_risk.model_package_group_name}/*"
}

# Who may assume the approver role. The default is the account root, which lets any IAM
# principal in this account that is allowed sts:AssumeRole assume it. On a one-person
# project, that's the same person who runs training, wearing a different hat (D-041).
variable "approver_principal_arns" {
  description = "IAM principals allowed to assume the approver role. Empty = this account's root (any principal granted sts:AssumeRole on it)."
  type        = list(string)
  default     = []
}

resource "aws_iam_role" "approver" {
  name = "${local.name}-approver"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        AWS = length(var.approver_principal_arns) > 0 ? var.approver_principal_arns : ["arn:${local.partition}:iam::${local.account_id}:root"]
      }
      Action = "sts:AssumeRole"
    }]
  })
  max_session_duration = 3600
}

# Review and decide, nothing else: it can't train, run the pipeline or register a model.
resource "aws_iam_role_policy" "approver" {
  name = "review-and-approve"
  role = aws_iam_role.approver.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ApproveOrRejectVersions"
        Effect   = "Allow"
        Action   = ["sagemaker:UpdateModelPackage"]
        Resource = [local.model_package_versions_arn]
      },
      {
        Sid    = "Review"
        Effect = "Allow"
        Action = [
          "sagemaker:DescribeModelPackage",
          "sagemaker:DescribeModelPackageGroup",
          "sagemaker:ListModelPackages",
          "sagemaker:ListModelPackageGroups",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:ListPipelineExecutionSteps",
        ]
        Resource = "*"
      },
      {
        Sid      = "ReadRunReports"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.datalake.arn}/pipeline-runs/*"]
      },
    ]
  })
}

# Attached to the training role: the separation of duties, stated as a deny so it holds
# even if someone later widens the role's allow list.
resource "aws_iam_role_policy" "sagemaker_no_self_approval" {
  name = "no-self-approval"
  role = aws_iam_role.sagemaker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "NoApproval"
        Effect   = "Deny"
        Action   = ["sagemaker:UpdateModelPackage"]
        Resource = "*"
      },
      {
        # registering a version as already Approved would skip the approver
        Sid      = "RegisterOnlyAsPending"
        Effect   = "Deny"
        Action   = ["sagemaker:CreateModelPackage"]
        Resource = "*"
        Condition = {
          StringNotEqualsIfExists = { "sagemaker:ModelApprovalStatus" = "PendingManualApproval" }
        }
      },
    ]
  })
}
