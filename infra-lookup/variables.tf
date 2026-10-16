variable "region" {
  type    = string
  default = "ca-central-1"
}

variable "project" {
  description = "Cost-allocation tag value. Not mlops-aiops: this stack outlives the series (D-048)."
  type        = string
  default     = "yyc311-lookup"
}

variable "bucket_name" {
  description = "S3 bucket for the page. Global across AWS; the public never sees it (OAC)."
  type        = string
  default     = "yyc311-neighbourhood-lookup"
}
