terraform {
  required_version = ">= 1.10" # S3 backend native locking (use_lockfile)

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # Remote state in S3 (D-032): losing a local state file would leave `terraform destroy`
  # blind to what it created. Partial config: the bucket name is account-specific, so it
  # lives in backend.hcl (git-ignored; copy backend.hcl.example). The bucket is created
  # once by hand (Stage 1 runbook step 2), outside this stack, so `destroy` can't delete
  # the state it depends on. Locking uses S3's native lockfile, so no DynamoDB table.
  #   terraform init -backend-config=backend.hcl
  # CI runs `terraform init -backend=false`, so it never touches this.
  backend "s3" {}
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project   = var.project
      managedby = "terraform"
      series    = "mlops-aiops-pipeline"
    }
  }
}
