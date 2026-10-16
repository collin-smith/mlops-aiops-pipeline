# infra-lookup — the community lookup page (D-048)

One static page, `index.html`, served from a private S3 bucket through CloudFront (Origin
Access Control). A separate stack from `infra/` with its own state key, so the end-of-series
teardown leaves it running. Cost: within CloudFront's always-free tier (1 TB and 10 million
requests a month); the page is about 150 KB.

```bash
cd infra-lookup
cp backend.hcl.example backend.hcl        # set the state bucket (same as infra/)
terraform init -backend-config=backend.hcl
terraform plan                            # expect 7 to add
terraform apply
aws s3 cp <path>/index.html "s3://$(terraform output -raw bucket)/index.html" \
  --content-type "text/html; charset=utf-8" --cache-control "max-age=300"
terraform output -raw url
```

After re-uploading the page, either wait five minutes (`max-age=300`) or invalidate:
`aws cloudfront create-invalidation --distribution-id "$(terraform output -raw distribution_id)" --paths "/index.html"`.

To take it down: empty the bucket, then `terraform destroy` here.
