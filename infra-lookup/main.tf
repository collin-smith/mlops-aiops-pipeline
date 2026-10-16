# The community lookup page (D-048): one static HTML file, private S3 bucket, CloudFront in
# front with Origin Access Control. No public bucket, no website endpoint, no custom domain.

resource "aws_s3_bucket" "page" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "page" {
  bucket                  = aws_s3_bucket.page.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "page" {
  bucket = aws_s3_bucket.page.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "page" {
  bucket = aws_s3_bucket.page.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_cloudfront_origin_access_control" "page" {
  name                              = var.bucket_name
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# AWS managed policies, looked up by name rather than hard-coded IDs.
data "aws_cloudfront_cache_policy" "optimized" {
  name = "Managed-CachingOptimized"
}

data "aws_cloudfront_response_headers_policy" "security" {
  name = "Managed-SecurityHeadersPolicy"
}

resource "aws_cloudfront_distribution" "page" {
  enabled             = true
  comment             = "Calgary 311 community lookup (independent; not a City of Calgary site)"
  default_root_object = "index.html"
  price_class         = "PriceClass_100" # North America + Europe: the readers are in Calgary
  http_version        = "http2and3"

  origin {
    domain_name              = aws_s3_bucket.page.bucket_regional_domain_name
    origin_id                = "page"
    origin_access_control_id = aws_cloudfront_origin_access_control.page.id
  }

  default_cache_behavior {
    target_origin_id           = "page"
    viewer_protocol_policy     = "redirect-to-https"
    allowed_methods            = ["GET", "HEAD"]
    cached_methods             = ["GET", "HEAD"]
    compress                   = true
    cache_policy_id            = data.aws_cloudfront_cache_policy.optimized.id
    response_headers_policy_id = data.aws_cloudfront_response_headers_policy.security.id
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

# Only this distribution may read the bucket.
data "aws_iam_policy_document" "page" {
  statement {
    sid       = "CloudFrontReadOnly"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.page.arn}/*"]
    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.page.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "page" {
  bucket     = aws_s3_bucket.page.id
  policy     = data.aws_iam_policy_document.page.json
  depends_on = [aws_s3_bucket_public_access_block.page]
}
