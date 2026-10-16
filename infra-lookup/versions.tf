terraform {
  required_version = ">= 1.10" # S3 backend native locking (use_lockfile)

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # Its own state key in the same state bucket as infra/ (D-032), so the end-of-series
  # `terraform destroy` in infra/ never touches it (D-048). backend.hcl is git-ignored;
  # copy backend.hcl.example.
  #   terraform init -backend-config=backend.hcl
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
