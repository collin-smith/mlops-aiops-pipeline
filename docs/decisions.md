# Decisions

ADR-lite. Newest first. Seeded from the planning docs
(`output/00-master-plan.md` §2) and confirmed by Collin on 2026-09-09.

---

## D-036 — `processed/311` holds one snapshot; history lives in `raw/`

**Status:** Accepted (2026-09-24). `to-parquet` clears `processed/311` before rebuilding it.
`snapshot` mirrors it to S3, deleting objects the upload didn't write, while `raw/311/asof=…/`
is only ever added to. `replay --asof <date>` downloads that date's raw files and rebuilds
`processed/` from them locally. It no longer downloads `processed/`, which would be the
latest build rather than that date's.
**Why:** `pyarrow.write_to_dataset` gives every file a fresh uuid name, and the upload never
deleted anything. So the first re-pull (Stage 5's "retrain on new data") would have left
two copies of every month, and Athena and training would have counted everything twice.
The validation gate's duplicate-ID check would have failed it, but the fix belongs at the
source. Found on 2026-09-23 while checking whether "data lake" was an honest description;
the one snapshot so far was unaffected (the counts matched exactly).
**How to apply:** the crawler and Athena always see the latest snapshot. To reproduce an
older run, `replay` it, then query locally or snapshot it back. The datalake bucket is
versioned, so a deleted processed object can be recovered for 30 days (noncurrent-version
expiry). Tested in `tests/test_socrata_pull.py` with a fake S3.

## D-035 — The promotion gate includes a fairness check across City sectors

**Status:** Accepted (2026-09-23). `src/promote/fairness.py` scores the evaluation
predictions at the triage operating point: the top decile, with one city-wide cutoff. It
compares groups of communities, the City's 8 planning `sector`s and the `srg` class
(ESTABLISHED / DEVELOPING / …). Both come from Community District Boundaries
(`surr-xmvs`), which `socrata_pull.py communities` snapshots next to each 311 pull. The
Stage 4 `ConditionStep` passes only if the lowest group recall is at least 0.8 of the
highest and every group's lift is at least 1.5. Groups under 1,000 rows or 100 breaches,
and requests with no community, are reported but not gated. Fewer than two measurable
groups fails.
**Why:** Stage 4's headline asks "Whose request waits?", and the exploratory pass found a
real NW/SE gap. A triage model that works in one part of the city and not another is a
governance problem, not only a weaker metric. The check costs about $0, since it's pandas
inside the evaluate step. Recall is gated (equal opportunity) because finding slow
requests is the model's job. Base rates aren't gated: they really do differ (17–29%), and
when they do, not every fairness metric can be equal at once. Sectors are used because
the open data has no address-quadrant field.
**How to apply:** the evaluate step writes `fairness.json` beside `evaluation.json`, and the
ConditionStep reads `recall_ratio.value` and `min_group_lift.value`. The thresholds are
starting values; confirm them against the Stage 2 model's real per-sector numbers. The
per-group table goes in the model card. The Stage 4 subtitle now names "a fairness check"
(D-033: every subtitle claim must match the article). The check covers the model's
performance, not the civic finding. Stage 8 keeps its "not validated by the City" framing.

## D-034 — A data-validation gate runs before any training spend

**Status:** Accepted (2026-09-23). `src/pipeline/validate.py` checks the raw pull before
preprocessing. Structural checks **fail** the step, and with it the pipeline: schema,
row count (a minimum, and ±25% against the previous run's report), unique
`service_request_id`, null rates, and dates (parseable, not closed before requested, not
in the future, no more than 14 days stale against `input_asof_date`). Artifact detectors
only **warn**: `bulk_close` (≥25% of a community/category's closures on one day) and
`group_outliers` (a community ≥10× and ≥30 days above its category's median). Every run
writes `validation.json`; `--strict` promotes warnings to failures.
**Why:** a bad pull used to train without any warning. The exploratory pass already
found one bad pull, the Montgomery 196-day pothole mean, and a truncated or stale
Socrata pull would fail the same way. The artifact detectors warn rather than fail
because the 5-year window may itself contain Montgomery: failing on it would block the
first real run over a known issue, and a bulk-close guess isn't proof. The two detectors
together separate an artifact from a real but slow community (Bowness, ~7×).
Writing the gate also exposed a date-parsing bug. `pd.to_datetime(errors="coerce")`
infers the format from the first value and turns any value in a different ISO format
into NaT, silently dropping it from the label. Every date parse now passes
`format="ISO8601"`.
**How to apply:** `preprocess` calls the gate first (or it runs as its own Processing
step, whichever Stage 3's `definition.py` makes cheaper). Calibrate the thresholds on the
first real pull (Stage 1), and record what the detectors report there. Nothing new is
sent to CloudWatch, since custom metrics are the budget's largest line; the report goes
to S3 with the run's outputs. The Stage 3 article covers the gate under "Testing the
pipeline code" (D-016).
**Calibration on the first real pull (2026-09-23, 2,911,486 rows):** every structural check
passed (0 duplicate IDs). `group_outliers` flagged Montgomery / Pothole Maintenance
(194.5 days against a typical 8.8). The data confirms it was a backfill: 38 tickets from
2021–23 were closed on 2024-10-20, and Montgomery's median is 6 days. `bulk_close` missed
Montgomery, because those 38 tickets were only 18% of the community's closures. It did
catch pieces of a city-wide purge, one community at a time. So a third, city-wide warning
was added: `stale_bulk_close`. It fires when at least 30 tickets of one category close
on the same day, each older than both 10× that category's median and 120 days. The
120-day floor leaves out "311 Contact Us", which closes batches at 91–98 days old: an
automatic close by policy, not a cleanup. On this pull, the check flags 14,043 tickets
(0.48%) in 182 events across 60 categories. The largest is 3,505 traffic-sign tickets
closed on 2025-08-26, at a median age of 1,085 days. Backlog clearing turns out to be
routine, not rare. `stale_bulk_close_mask()` is exported so the Stage 2 label can use
the same definition. How to label those rows is still an open Stage 2 decision.

## D-033 — Every article carries an "MLOps + AIOps on AWS · Part N of 8" subtitle (amends D-022)

**Status:** Accepted (2026-09-23). Each article's front matter gets a `subtitle:` line,
used as the Medium subtitle and shown under the headline in the HTML preview:
`MLOps + AIOps on AWS · Part N of 8 — <what this stage does>`. Headlines are unchanged.
**Why:** D-022's story-first headlines name the AWS service but not the discipline. Only
Stage 7's headline said "AIOps", and none said "MLOps". Those are the terms readers search
for and what the series is about. The subtitle adds them to every article without making
the long headlines longer, and the "Part N of 8" makes the series order clear.
**How to apply:** new or redrafted articles keep the `subtitle:` line. Every claim in a
subtitle must match what the article actually covers. Stage 2's "2.8× lift" comes from
the exploratory run and gets re-checked once the Stage 2 model is trained. The subtitle
goes in Medium's subtitle field, and the LinkedIn post uses the same series tag.

## D-032 — Terraform state lives in S3, in a bucket created outside the stack

**Status:** Accepted (2026-09-22). `infra/versions.tf` uses an S3 backend (partial config;
the account-specific bucket name is in git-ignored `infra/backend.hcl`, copied from
`backend.hcl.example`), with S3's native lockfile (`use_lockfile`, Terraform ≥ 1.10), so
there's no DynamoDB table. The bucket, `mlops-aiops-tfstate-<account-id>`, is versioned,
blocks public access, and is created by hand once (Stage 1 runbook step 2).
**Why:** a lost or corrupted local state file leaves `terraform destroy` unable to see what
it created, and in a cost-first project orphaned resources are the failure to fear.
Versioning lets a bad state write be rolled back. Keeping the bucket outside the stack
means `destroy` can't delete the state it's reading. Costs cents.
**How to apply:** `terraform init -backend-config=backend.hcl`. CI runs
`terraform init -backend=false` and never touches state. The bucket is the last thing
deleted at the end of the series, after `destroy` succeeds (Stage 8 teardown).

## D-031 — The $25 hard stop counts all account spend since the series start (amends D-011)

**Status:** Accepted (2026-09-22). The hard stop moves from a tag-filtered monthly budget
to an **account-wide budget that accumulates from `series_start`** (`…-series-total`,
annual period, $15 warning, $25 action). A second account-wide budget,
`…-account-monthly`, carries the $10 forecast warning (D-030). Per-project spend is read
from Cost Explorer's tag filter, not from a budget.
**Why:** the tag-filtered monthly budget had three blind spots. (1) The `project` tag counts
nothing until it is activated as a cost-allocation tag, which isn't retroactive and can
take up to 24 hours. (2) Some costs can't carry a tag at all, including CloudWatch custom
metrics, the largest line in the estimate, and SDK-launched SageMaker jobs only carry it if
the code passes `Tags`. (3) A monthly budget resets, so a build spanning two calendar months
had an effective cap of $50. The account is dedicated to this series, apart from a
near-$0 static site, so account-wide spend is a safe, slightly conservative proxy. The
deny policy still attaches only to this project's roles, so the site is never affected.
**How to apply:** activate the `project` cost-allocation tag after the first apply (Stage 1
runbook step 2). Every SageMaker job passes `Tags` (`src.common.config.aws_tags()`), per the
Stage 2 coding rule. `nuke.sh` prints both numbers: account spend since the series start
(what the stop counts) and tagged month-to-date spend.

## D-030 — Pause safety: nothing may keep billing between sessions unnoticed

**Status:** Accepted (2026-09-22). Items 1–4 built 2026-09-22 (item 3 lives on
`…-account-monthly`, because a forecast on the annual series budget would extrapolate a
whole year); item 5 is a Stage 2 coding rule.
The Terraform base left up between stages costs cents a month. The risk is a session
that stops partway through with something still billing, which the budget catches late,
because AWS Budgets only updates a few times a day. Five controls:
1. **`scripts/nuke.sh --check`** (build before the first `terraform apply`, Stage 1): a
   check-only mode that lists anything billing without deleting it, and exits non-zero if
   anything is found. It also fills `nuke.sh`'s gaps: in-flight training, processing and
   transform jobs, pipeline executions, classic notebook instances, EventBridge rules and
   Lambda schedules tagged to the project, and whether `retrain.yml`'s schedule is enabled.
2. **AWS Cost Anomaly Detection** (Terraform, Stage 1): a monitor with an email
   subscription on the project's spend. Free; flags a daily spike without waiting for the
   $15 warning.
3. **A forecast alert on the budget** (Terraform, Stage 1): fires when forecast spend
   passes $10, before actual spend reaches $15.
4. **A "Before you pause" block in every stage's teardown**, listing exactly what that
   stage can leave billing, ending with a clean `nuke.sh --check` (📸).
5. **A maximum runtime on every SageMaker job** (coding rule from Stage 2 on:
   `max_run` / `MaxRuntimeInSeconds`, and `max_wait` for spot), so a stuck job ends
   itself.
**Why:** cost control is the series' first constraint, and the biggest single risk (a
forgotten hourly Model Monitor schedule, ~$165/month) builds up between sessions, not
during them. All five controls cost $0.

## D-029 — One short-lived Serverless Inference demo in Stage 6 (amends D-007)

**Status:** Accepted (2026-09-22). Stage 6 stands up a **SageMaker Serverless Inference**
endpoint for the approved model, invokes it a handful of times, records cold-start and warm
latency, and deletes it in the same run — `scripts/serverless_demo.py`, run by hand once.
Batch Transform stays the deploy path; nothing in the pipeline serves in real time.
**Why:** D-007's reason is the 24/7 bill of an instance-backed endpoint. Serverless has no
idle charge (billed per invocation), so the demo costs cents, and it replaces "I chose not
to" with a measured comparison: here's real-time serving on this model, here's its latency
and cost per call, and here's why batch still fits monthly-cadence data.
**Guardrails kept:**
- the pipeline's SageMaker role still has no `CreateEndpoint*` permission; the demo runs
  under the operator's own credentials
- `ci.yml` still fails on endpoint code in `src/`, `notebooks/`, and now `scripts/`, with
  exactly one named exclusion (`serverless_demo.py`)
- a second CI check fails on `ProvisionedConcurrency` anywhere, because that brings the
  idle charge back
- the script tags everything `project=mlops-aiops` (the Budget filter), refuses to run
  if the Budgets deny policy is attached to the project role, and deletes the endpoint,
  endpoint config, and model in a `finally`; `nuke.sh` sweeps any endpoint left behind
**Note:** the Budgets deny policy attaches to the project roles only, not to the operator,
so the hard-stop check in the script is what keeps the demo inside the $25 cap.

## D-028 — Civic findings are tracked over time, not published as a one-off snapshot

**Status:** Accepted (2026-09-22). Stage 7 gains a small Layer C, a "civic insight
scorecard": one Athena query recomputes each headline civic finding monthly (community
gap, top-agency breach rates, overall breach rate, chronic-location count, seasonal
index). It is backfilled from the full history, appended every pipeline run, and stored
as a small CSV in S3 — **not** as CloudWatch custom metrics. Layer B's EWMA/CUSUM detector
reads that CSV directly and alerts via the same SNS topic. Stage 8 adds a "keeping the findings honest
over time" section: what movement would weaken each finding, same-month-prior-year
comparison, and a before/after (difference-in-differences) method if the City acts on one.
**Why:** Stage 8's findings come from one sample at one point in time and aren't
City-validated; without a re-check mechanism they stay frozen. Reusing Layer B's detector
makes this nearly free — no new always-on resources, Athena scans of a few MB per run
inside the existing 2 GB workgroup cap. Custom metrics were rejected on cost: at
$0.30/metric/month they are already the largest line in the budget, and ~9 civic series
would add roughly $3–5 over the series. The S3 CSV costs effectively nothing.
**How to apply:** keep the metric list short (≤6 series) — per agency × community cells
would guarantee false alarms. Alerts are prompts to look, not findings; say so in both
articles. No new diagram node: the anomaly tile's subtitle reads "pipeline + civic".

## D-027 — One architecture preview image lives in `code/docs/`, as a narrow exception to D-026

**Status:** Accepted (2026-09-21). `code/docs/architecture.svg` — a copy of the
auto-generated "Full architecture" preview (`output/stages/_preview/full.svg`) — is
committed to the public repo and embedded in `README.md`'s Architecture section.
**Why this doesn't undo D-026:** the distinction there was code vs. *content* (article
prose, the full 9-page diagram set, anything meant for Medium). A single illustrative image
in the README documenting the repo's own architecture is neither — it's the same category
as a screenshot or logo many repos commit, and it contains nothing that isn't already
generic AWS service names and category colours. It is not the source of truth (the
`.drawio` files in `output/stages/` are) and not part of the article production pipeline.
**How to apply:** this file is a manual copy, not auto-synced — after changing `NODES`/
`EDGES` in `architecture_gen.py`, re-run it and `cp output/stages/_preview/full.svg
code/docs/architecture.svg`. It's the plain-box auto-generated rendering, not the
hand-tidied official-AWS-icon version (still pending the manual draw.io export — see the
project backlog to export & commit that PNG). Once that export exists, replace this file
with it rather than keeping both as a second, lower-fidelity copy.

## D-026 — Article/diagram content moved out of the public repo entirely, into `output/stages/`

**Status:** Accepted (2026-09-21). Everything that isn't pipeline code — article drafts,
outlines, HTML previews, `.drawio`/`.drawio.svg` diagrams, the `articles/` publish mirror,
per-stage `runbook.md` files — moved from `code/stages/` to `output/stages/` (a sibling of
`code/`, never committed/pushed). `architecture_gen.py` and `table_gen.py` (the generator
tooling, not the content they generate) stay in the repo at `code/scripts/`, writing their
output to `../output/stages/` rather than alongside themselves.
**Why:** matches the exact split already established and working in the `rag-tsb-safety`
sibling series — that repo's public README states plainly "written up outside this repo,
see `../../outputs/rag-tsb-safety/articles/`," and keeps only the rendering *script*
(`scripts/render_diagram_previews.py`) in `code/`. This project had it backwards:
everything, including unpublished outlines with working notes, was sitting inside the
folder slated to become a public GitHub repo. Moving the content out is the structural
version of D-025 — a working note that never ships inside the repo can't leak into
published prose later.
**How to apply:** `code/README.md`'s Architecture and Series-index sections reference
`../output/stages/...` as plain paths (not GitHub links — they resolve outside the repo, so
a link would 404 for anyone else who clones it). Regenerate diagrams with
`python code/scripts/architecture_gen.py`; regenerate table images with
`python code/scripts/table_gen.py` (or import `render_table` from it) — both write to
`output/stages/` automatically, never into `code/`.

## D-025 — The public repo holds engineering content only; planning notes stay private

**Status:** Accepted (2026-09-21). `code/` (the public repo) contains the pipeline code,
infrastructure, tests, and the engineering docs in `docs/`. Private planning notes, working
outlines, and anything about why the series was written live outside it in `output/`, and
are never committed.
**Why:** the series should read as what it is — a technical build on public civic data.
Working notes written for the author carry context that doesn't belong in a public repo,
and they have leaked into draft prose before.
**How to apply:** before drafting any stage from its outline, re-read the outline's
Objective line and framing notes and make sure nothing written for the author survives
into the published text. Decision records here state the engineering reason for a choice,
nothing else.

## D-024 — Publish-ready article mirror; tables render as images, never HTML/markdown tables

**Status:** Accepted (2026-09-21; location updated same day, see D-026).
`output/stages/articles/` is a flat, numbered mirror of every stage's article/outline + its
architecture diagram (`0N-<slug>.md/.html/.drawio/.drawio.svg`, plus
`00-full-architecture.*` and `00-network-hardened.*`) — the hand-off copy for Medium, kept
alongside `output/stages/stage-N-*/` (the working/build location, not auto-synced; re-copy
before publishing). Mirrors the layout already used for the `rag-tsb-safety` series.
**Why tables are images:** confirmed from that same series
(`../../../ragproject/outputs/rag-tsb-safety/articles/`) that Medium strips/mangles raw HTML
`<table>` markup on paste — a markdown table isn't safe either. Every table in every article
in this series (Stage 6's cost breakdown, Stage 8's stakeholder-lens and recommendations
tables from `docs/exploratory-findings.md`, etc.) renders as a small styled SVG → PNG via
`code/scripts/table_gen.py`, embedded as an image with descriptive alt text — never as
literal table markup in the `.md`/`.html`.
**How to apply:** call `table_gen.render_table(...)` per table, output to
`output/stages/articles/assets/<slug>-table-<name>.png`, embed that PNG. `cairosvg` is now
a `dev` extra (`uv sync --extra dev`) for the PNG step; the SVG alone is written even
without it.

## D-023 — Region: `ca-central-1` (Montreal), not `us-east-1`

**Status:** Accepted (2026-09-21). All infrastructure deploys to `ca-central-1`
(`infra/terraform.tfvars.example:3`); no resource is pinned to another region.
**Why:** the dataset is a Canadian municipality's, and the Stage 8 narrative buyer persona
is a Canadian municipal 311 operations manager — even though this is public open data with
no legal residency requirement, a real municipal buyer is realistically risk-averse about
data leaving Canada, and choosing `ca-central-1` answers "why would a Canadian city's
pipeline run in `us-east-1`?" before anyone has to ask. It's also the correct
Well-Architected call under the Security and Sustainability pillars for a project whose
subject *is* Canadian civic data.
**Tradeoffs, stated openly:** every service the pipeline uses (SageMaker Training/
Pipelines/Batch Transform/Model Registry/Model Monitor, Glue, Athena, S3, Budgets) is
available in `ca-central-1` at the instance types this project needs, so there's no
capability gap. Pricing runs modestly higher than `us-east-1` (roughly 5–10% on compute and
storage, depending on service) — worth naming as a conscious tradeoff in the article
("chose data-residency alignment over the cheapest region, and it still fit comfortably
under budget") rather than something a reader discovers unexplained. Latency is a
non-factor since nothing is served real-time (D-007).
**Addendum (2026-09-23), why not `ca-west-1` (Calgary):** the original decision compared
against `us-east-1` only. Checked afterwards against AWS's regional service table, every
service the pipeline uses (SageMaker AI, Glue, Athena, S3, CloudWatch and Logs, SNS)
exists in both Canadian regions. Both also satisfy the residency argument above.
`ca-central-1` stays, for three reasons. It has the wider catalogue (184 services to
`ca-west-1`'s 133), which leaves room for backlog items. The SageMaker quota cases and the
applied stack were already there. And being physically in Calgary gains a batch pipeline
nothing. That table lists services, not features: SageMaker feature- and instance-level
availability in `ca-west-1` (Pipelines, Model Monitor, spot `ml.m5.large`) was not checked,
so this isn't a claim that Calgary couldn't run the pipeline.

## D-022 — Story-forward article headlines

**Status:** Accepted (2026-09-09). Each article gets a `headline:` front-matter field — the
Medium H1 — that **leads with the civic story** and puts the AWS mechanic after a dash
(e.g. *"Whose request waits? — a human approval gate with the SageMaker Model Registry"*).
The short internal `title:` ("Model Registry & Governance Gate") stays as the repo-index
and diagram-page label.
**Why:** makes the series click for a general reader on Medium/LinkedIn instead of reading
as one more generic SageMaker tutorial — it ties the tech to an outcome. The mechanic
stays visible for readers who are skimming for the AWS content. Each article body also re-grounds in the stakes in its intro (Stage 1's draft
already does).

## D-021 — Drop the "intermission" (camera cost-cap) piece; the series is 8 articles

**Status:** Accepted (2026-09-09). The bonus mid-series piece (mining the Bedrock
security-camera project's logs) is removed from this project. The series is **8 stage
articles**, all on the Calgary 311 pipeline.
**Why:** it was never part of the pipeline — different subject, own data, own AWS
resources, no shared code. It was a cadence-breaker / buffer day; dropping it removes a
distraction and keeps the repo and the narrative on one thing. Cost-discipline continuity
with the camera + EKS projects is still made as a point in Stage 8, without a dedicated
article. Schedule tightened accordingly (D-013) — no T+10 buffer slot.

## D-020 — Default build has no customer VPC; the VPC-hardened variant is documented, not deployed

**Status:** Accepted (2026-09-09). Every service in the build (S3, Glue, Athena, SageMaker
jobs, CloudWatch, SNS, EventBridge, IAM, Budgets) is regional / managed. `infra/` provisions
**no `aws_vpc`, no subnets, no NAT**. SageMaker jobs run in AWS's managed VPC — `VpcConfig`
is not set.
**Why:** the enterprise pattern (SageMaker jobs in private subnets reaching S3/ECR/SageMaker/
CloudWatch/STS only through VPC endpoints, no IGW/NAT) needs ~6 interface endpoints at
~$7.30/month each ≈ **$44/month** — several times the whole project's $8 cap, for a teaching
series where the subject is the ML lifecycle, not networking (the EKS series already covered
networking).
**How to apply:** the VPC-hardened variant is a first-class *documented* artifact —
`output/stages/architecture-network.drawio` (the diagram) and
`infra/vpc.tf.example` (the Terraform, committed but not wired in). Stage 6 / Stage 8 name
batch-only + VPC-less as deliberate, budget-driven boundaries (see [[decisions]] D-018).
**(2026-09-21, reaffirmed):** "documented" means actively surfaced, not buried — the
network-hardened diagram is referenced from the top-level `README.md` architecture section
and paired explicitly with the Full-pipeline page in Stage 8's article, not left as a file
only `output/stages/README.md` mentions. All AWS-service nodes in both diagrams use
official AWS-2024 resource icons with the correct category colour (e.g. S3 =
`mxgraph.aws4.s3`, `fillColor=#7AA116` storage-green, white glyph) per D-019 — confirmed
already implemented in `architecture_gen.py`, no drift found.

## D-019 — Architecture diagrams: draw.io, generated, cumulative per stage

**Status:** Accepted (2026-09-09; icons iterated same day). Format is **draw.io**
(`.drawio`), not Excalidraw. AWS services render as **AWS-2024 resource icons**
(`shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.<name>` with `strokeColor=#ffffff` —
coloured square, white glyph). The `resIcon` **short** names matter: `sns` not
`simple_notification_service`, `ecr` not `elastic_container_registry`,
`identity_and_access_management` not `…_iam`, `budgets` / `cost_explorer` not
`cost_management` — the long/product names render a non-white or missing glyph. All in
`_RESICON` in the generator; fix a name there if a tile renders wrong. Non-AWS nodes
(Socrata API, local scripts, GitHub Actions) are plain coloured boxes. The `_preview/*.svg`
renders use small white category pictographs (they don't depend on draw.io shapes).
`code/scripts/architecture_gen.py` generates everything from a single `NODES`/`EDGES`
model, writing to `output/stages/` (D-026): `architecture.drawio` (9 pages — Full + Stage
1–8) plus a single-page `architecture.drawio` in each stage folder. Each stage's page is
**cumulative** — every
resource/flow that exists by the end of that stage. From Stage 2 on, components added that
stage get a thin pink ring on the icon + a pink label (no "NEW" badge, no offset box);
Stage 1 has no accents (everything is new there). Hand-tidy edge routing in the draw.io UI,
export a PNG next to each source.
**Why:** the series' value is showing an architecture *grow and get operated*; a diff-able
generated model keeps 9 diagrams consistent and cheap to update as the build evolves.
Articles publish to Medium (`collin-smith.medium.com`) — PNG exports embed there.

## D-018 — "Batch deploy, no always-on endpoint" is a stated boundary, not a hidden gap

**Status:** Accepted (2026-09-09; updated by D-029). Batch transform is the honest fit for
monthly-decision-cadence data, and an always-on endpoint bills 24/7 for no benefit here.
Stage 6 backs this with a measured Serverless demo (D-029) rather than an assertion, and
Stage 8 says it out loud, so it never reads as something that was avoided because it was
hard.
**Why:** a boundary you volunteer is a strength; one that surprises you is a weakness.

## D-017 — Data & privacy posture is documented, not assumed

**Status:** Accepted (2026-09-09). `docs/data-and-privacy.md` states it: Calgary 311 is
published open data, de-identified at source; no personal names; `address` is dropped in
the processed zone (community/ward granularity only); no attempt to re-identify. One
paragraph in Stage 1 and a line in the Stage 4 governance article.
**Why:** governance and safety are more than model metrics — they start with what data
you keep and why. Costs
nothing; rounds out the story.

## D-016 — Pipeline-code testing is a first-class beat, not buried in "Stage 3 mechanics"

**Status:** Accepted (2026-09-09). Stage 3 gets an explicit section — "Testing the pipeline
code" — covering: unit tests on the feature/label functions (leakage guard, threshold
math, backlog windowing — already written), a pipeline-step smoke test that runs
`preprocess`/`evaluate` entrypoints on a tiny fixture with no AWS calls, and `ruff` +
`pytest` + `terraform validate` gated in `ci.yml`.
**Why:** "how do you test the infra and pipeline code, not just the model" is a question
most MLOps write-ups skip; stating it as its own section makes the answer visible.
**How to apply:** in the Stage 3 article, give it a named subhead and ~200 words.

## D-015 — A model card ships in Stage 4

**Status:** Accepted (2026-09-09). `docs/model-card.md` — intended use, training-data
window and `asof` date, features, metrics (overall + the categories/wards where it
under-performs), known limitations, the proxy-label caveat, retrain cadence, approver.
Regenerated (or hand-updated) each promotion; linked from the Model Package.
**Why:** the governance gate is currently mechanical (a PR-AUC floor + a separate IAM
approver). A model card is the difference between "I gated a metric" and "I document what
I ship" — the latter is what AI governance looks like in practice. ~1 hour of work, no
new services.

## D-014 — Terraform manages the entire stack, all 8 stages — not just Stage 1's base

**Status:** Accepted (2026-09-09). Every AWS resource in the series is in `infra/`. Later
stages add resources behind their own `.tf` files (`sagemaker.tf`, `pipeline.tf`,
`monitor.tf`, …) applied incrementally; the SageMaker Pipeline *definition* is Python SDK
(idiomatic), but the role, registry, schedules, and monitor infra are Terraform. Stated
explicitly in the Stage 3/5/7 articles.
**Why:** end-to-end IaC is one of the strongest demonstrations of production discipline in
this series; the repo must not read like Terraform stops after Stage 1.

## D-013 — Publishing start date

**Status:** OPEN — the last blocker. Cadence is fixed (3–4 days); only `T+0` is undecided.
Schedule below assumes `T+0` and is filled into each article's `publish_date` front-matter
once set.

| Piece | Day | Piece | Day |
|---|---|---|---|
| Stage 1 | T+0 | Stage 5 | T+15 |
| Stage 2 | T+3 | Stage 6 | T+19 |
| Stage 3 | T+7 | Stage 7 | T+23 |
| Stage 4 | T+11 | Stage 8 | T+26 |

8 articles, ~3.5 weeks. Stage 5 (highest effort) and Stage 7 (two monitoring layers) each get
a 4-day gap; the rest are 3.

---

## D-012 — Parquet conversion runs locally; Glue is crawler-only

**Status:** Accepted (2026-09-09).
Pull 311 with `socrata_pull.py`, convert to partitioned Parquet locally, upload a frozen
snapshot to `s3://…/raw/311/asof=…/` + `s3://…/processed/311/`. Glue runs only the
crawler that catalogs the Parquet. The Spark ETL job (`src/ingest/glue_job.py`) is kept
as a documented alternative path for the article.
**Why:** ~150 MB Parquet is single-node; local conversion saves a few dollars and removes
the "forgot to shut the Glue job down" risk. The "I used Glue" story is intact via the
crawler + the committed (optional) job.

## D-011 — Budget hard-stop, not alerts only

**Status:** Accepted (2026-09-09); budget scope amended by D-031.
`enable_budget_hardstop = true`. An AWS Budgets **action** attaches a deny policy to the
SageMaker / Glue / GitHub-Actions roles when month-to-date ACTUAL spend tagged
`project=mlops-aiops` crosses $25. Warn notification at $15.
**Why:** the stated constraint is "must not risk a hundreds-of-dollars bill." An alert you
might miss is not a control. Detach steps are in `infra/README.md`.

## D-010 — Full series (~$8 expected), not the minimum-spend variant

**Status:** Accepted (2026-09-09).
All 8 stages incl. scheduled retrain, Model Monitor, and RCF/control-chart anomaly
detection. `$25` hard cap.
**Why:** the MLOps *and* AIOps halves of the series both need Stages 4–7; the <$3 variant delivers a
thinner MLOps story. Expected spend (~$8) is well within tolerance.

## D-009 — Dataset: City of Calgary 311 Service Requests (`iahh-g8bj`)

**Status:** Accepted (2026-09-09; recommended in planning 2026-09-07).
~7.5M rows, daily updates, Socrata API. Alberta + federal portals assessed and rejected
as primary (aggregate/monthly — wrong shape for Stages 5–7). Building Permits `c2es-76ed`
is the fallback.
**Why:** only candidate that is row-level, genuinely daily, seasonal, batch-scorable, and
carries a discoverable civic insight. National relevance is addressed by joining
StatCan/Alberta/Census data as *exogenous features*, not by switching to an aggregate series.

## D-008 — Modeled target: per-request binary classification (SLA-breach risk)

**Status:** Accepted (2026-09-09).
`breach = 1` if `days_to_close` > the request's `service_name` 75th-percentile threshold
(computed on the training split only). Monthly-volume forecasting is the Stage 2
*exploratory* insight, not the deployed model.
**Why:** per-request scoring is the honest fit for Batch Transform (Stage 6: score today's
open requests) and gives Stage 8 a concrete product. It is a **proxy** threshold, not a
City-published SLA — every article says so.
**Exploratory check (2026-09-09, `docs/exploratory-findings.md`):** label behaves as
designed (~22% breach rate); signal is **modest** — a tree model reaches ~0.69 ROC-AUC /
0.54 PR-AUC on a temporal split, 2.8× lift in the top decile. Triage-grade, not
high-accuracy. A weak final result is still publishable.

## D-007 — Deploy target: Batch Transform only, no always-on endpoint

**Status:** Accepted; amended by D-029 (one short-lived Serverless demo in Stage 6).
`ci.yml` fails the build if endpoint-creation code appears outside that demo; the pipeline's
SageMaker IAM role has no `CreateEndpoint*` permission.
**Why:** cost-honest for monthly-decision-cadence data; endpoints are the #1 surprise bill.

## D-006 — IaC: Terraform for infra, SageMaker Python SDK for the pipeline DAG

**Status:** Accepted. Consistency with the EKS series; SDK is the idiomatic way to express
a SageMaker Pipeline.

## D-005 — Cadence: deliberate 3–4 day publishing rhythm, ~3.5 weeks

**Status:** Accepted. A conscious choice not to drift into the near-daily pace of the K8s
series. Publish dates are pre-filled per D-013. (Was ~4 weeks; tightened when the
intermission was dropped — D-021.)

## D-004 — Camera project stays out of the pipeline

**Status:** Accepted (extended by D-021). Collin's home security-camera project is
separate — its own data and AWS resources. It gets no article in this series; the
cost-discipline continuity point is made in Stage 8 without one. The 311 pipeline is 100%
public Calgary 311 data.

## D-003 — Stage 8 product framing is narrative only

**Status:** Accepted. Buyer persona = a municipal 311 operations manager. No company
formed, no data-licensing/IP review. Stated explicitly in the article.
