# Model card — 311 SLA-breach triage

*Template. Populated in Stage 4 and regenerated (or hand-updated) at every promotion.
Linked from the SageMaker Model Package. Values in `<…>` are filled by
`src/pipeline/evaluate.py` / the promotion step.*

| | |
|---|---|
| **Model name** | `311-breach-risk` |
| **Version / Model Package ARN** | `<arn>` |
| **Date promoted** | `<date>` |
| **Approved by** | `<iam-principal>` (separate from the training role — see `infra/iam.tf`) |
| **Training data** | Calgary 311 `iahh-g8bj`, snapshot `asof=<date>`, requests from `<start>`–`<end>` |
| **Algorithm** | XGBoost (SageMaker built-in), `<hyperparams>` |

## Intended use

Score **open** 311 service requests at or shortly after intake to estimate the
probability that the request will take **longer to resolve than its category's historical
norm** (the "breach" label). Output is a ranked list for a 311 operations team to review
for proactive routing or expectation-setting.

**Not** for: SLA enforcement, staff performance evaluation, automated decisions without a
human in the loop, or any use at address-level (the model operates at community grain).

## Label definition (and its caveat)

`breach = 1` if `days_to_close` exceeds the **75th percentile** of `days_to_close` for
that `service_name`, computed on the training split only. **This is a proxy** — a
category's own historical distribution — **not a City-published SLA.** A request can be
"a breach" here and still be well within any official target.

## Features (intake-time only)

`service_name`, `agency_responsible`, `comm_name`, `source`, request month / day-of-week /
holiday-week, and rolling 30-day backlog counts for the category and community. **No**
field derived from `updated_date`, `closed_date`, or `status_description` (leakage guard:
`src/features/build_labels.py::assert_no_leakage`, unit-tested).

## Performance

| Metric | Value |
|---|---|
| ROC-AUC (temporal holdout) | `<x>` |
| PR-AUC (temporal holdout) | `<x>` (base rate `<x>`) |
| Top-decile lift | `<x>`× |

### Where it under-performs

`<table: the 3–5 service categories and communities with the lowest per-group AUC / worst
calibration>` — reported deliberately. A triage model that is systematically wrong for one
category or one part of the city is a governance issue, not just a number.

## Known limitations

- Proxy label (above).
- Intake-time features only — no crew capacity, weather, or work-order detail, which are
  likely the real drivers of long resolutions. Expect modest AUC.
- Taxonomy drift: `service_name` values are renamed/merged over time; an alias map is
  applied but new categories fall back to a global threshold.
- Trained on a snapshot; production behaviour depends on the retrain cadence (Stage 5).
- Community coverage: communities with `< N` historical requests get noisy thresholds.

## Monitoring

Data-quality + model-quality drift via SageMaker Model Monitor (Stage 7). A documented
synthetic-drift test (`src/monitor/inject_drift.py`) verifies the monitor fires. Pipeline
operational health (run duration, cost, failure rate) is watched separately (Stage 7B).

## Retraining

Scheduled monthly (GitHub Actions, Stage 5). A challenger is promoted only if it beats the
**currently-approved** model on a frozen holdout by more than the guardband (`0.005` PR-AUC).
On promotion the previous version is marked `Deprecated`.
