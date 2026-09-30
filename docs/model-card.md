# Model card — 311 SLA-breach triage

*Updated at every promotion and linked from each Model Package's metadata. No version is
promoted yet: the only candidate so far, the Stage 2 model, was **rejected by the promotion
gate** on 2026-09-30 (pipeline run `wm2m10zouect`, D-041). Its numbers are below, because a
rejected model's card is part of the record too.*

| | |
|---|---|
| **Model name** | `311-breach-risk` |
| **Status** | Rejected by the gate: sector recall ratio 0.55, floor 0.8. Not registered. |
| **Version / Model Package ARN** | none (the group `mlops-aiops-breach-risk` is empty) |
| **Date promoted** | not promoted |
| **Approved by** | the `mlops-aiops-approver` role, which is separate from the training role (`infra/registry.tf`); nobody yet |
| **Training data** | Calgary 311 `iahh-g8bj`, snapshot `asof=2026-09-23`; the last year is held out as the test year |
| **Algorithm** | XGBoost 3.2.0 (the SageMaker built-in image's version), trained as a Processing job (D-039); hyperparameters in `src/pipeline/train.py` `PARAMS` |

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
| ROC-AUC (temporal holdout) | 0.662 |
| PR-AUC (temporal holdout) | 0.327 (base rate 0.181) |
| Top-decile lift | 2.04× (36.9% of flagged requests run late, against 18.1% overall) |
| Top-decile recall | 0.204 |

Test year: 476,613 requests. The saved artifact reproduces these exactly when reloaded.

### Where it under-performs

Reported deliberately: a triage model that is systematically weaker in one part of the
city is a governance issue, not just a number. At the operating point (flag the riskiest
10% with one city-wide cutoff), recall is the share of late requests the flag catches:

| Sector | Requests | Late | Flagged | Recall | Lift |
|---|---:|---:|---:|---:|---:|
| SOUTHEAST | 42,054 | 18.1% | 13.5% | **0.290** | 2.15 |
| NORTH | 46,150 | 16.9% | 11.2% | 0.252 | 2.25 |
| NORTHEAST | 60,502 | 17.7% | 10.3% | 0.247 | 2.40 |
| NORTHWEST | 51,103 | 16.7% | 10.9% | 0.231 | 2.13 |
| EAST | 22,458 | 18.1% | 9.7% | 0.223 | 2.30 |
| SOUTH | 71,443 | 16.5% | 8.7% | 0.191 | 2.19 |
| CENTRE | 116,868 | 19.9% | 10.3% | 0.176 | 1.71 |
| WEST | 34,747 | 16.2% | 8.0% | **0.159** | 1.98 |
| *no community* | 31,288 | 22.5% | 5.8% | 0.094 | 1.61 |

The flag catches 29% of SOUTHEAST's late requests but 16% of WEST's: a ratio of 0.55,
against the gate's floor of 0.8. By growth class it's 0.70 (DEVELOPING 0.209 against
COMPLETE 0.298). Lift stays well above 1 everywhere, so a flag means about the same thing
wherever it lands; the problem is how often it lands. The single cutoff flags 13.5% of
SOUTHEAST's requests and 8.0% of WEST's, so the areas where high-risk categories cluster
get more of the triage. Requests with no community are the weakest group of all (recall
0.094); they are reported, not gated.

Known category misfits (`docs/exploratory-findings.md`): traffic signs are over-flagged
because of the August 2025 backlog purge; graffiti and transit passes are under-flagged.

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
