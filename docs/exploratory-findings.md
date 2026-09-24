# Exploratory findings — Calgary 311, pre-build data check

**Date:** 2026-09-09
**Run against:** the live Socrata API (`data.calgary.ca/resource/iahh-g8bj`) — **no AWS, no
cost**. Aggregate SoQL queries + a ~230k-row local sample.
**Purpose:** before building, confirm that (a) the civic story is real, (b) the
classification target is actually learnable, (c) Calgary 311 is the right dataset.

**Verdict:** all three hold. The civic story is strong and interpretable. Breach risk is
**modestly** predictable at intake — good enough for a triage use case (2.8× lift in the
top decile on this sample), not good enough to claim high accuracy. *Update 2026-09-24:
on the full 2.9M-row pull, the Stage 2 baseline reaches **2.0×** (D-037).* That is a legitimate result to
publish either way. Proceed with Calgary 311 + binary classification.

> Repro: `python scripts/explore_311.py` (queries) — numbers below are from
> 2026-09-09 and will drift as the City updates the dataset daily.

---

## Top 3 recommendations

The sharpest three — each a problem the data actually shows, with a concrete fix. Full
list (8 operational + 7 pipeline) under [Recommendations](#recommendations).

| # | Problem the data shows | Proposed solution |
|---|---|---|
| **1** | **Resolution is reactive** — nothing at intake flags a request as likely to drag, so it isn't escalated until a resident calls back. | **Predictive triage at intake:** score each day's open requests, surface the top ~10% most likely to breach their category norm. In the exploratory model that shortlist is ~77% true-positive vs a 28% base rate — **2.8× lift**. Sub-dollar daily batch job; no real-time infra. |
| **2** | **Capacity is planned to an annual average, but demand isn't annual** — total volume swings 2× Dec→Jun and the category mix swings far harder (snow/ice ~20,000 in Jan → 3 in Jul; weeds the reverse). | **Forecast-driven seasonal capacity planning:** use the very stable per-category annual curves to pre-position crews/contractors 4–6 weeks ahead of each wave, with the 30-day backlog count as an in-season trigger. Needs only seasonal decomposition, not the model. |
| **3** | **The data can't be trusted for management reporting** — `service_name` categories have drifted across department prefixes and back-end systems, and ~26,000 requests in 5 years are duplicate reports of the same issue. | **(a)** Canonical category taxonomy with a named owner + an alias map for historical rows. **(b)** Intake-time "similar open request nearby?" check (location + category + time window) — flags duplicates and doubles as a priority signal when the same problem is reported repeatedly. |

*Maps to Stage 8's narrative-only framing: #1 is the sellable capability; #2 and #3 are
what a real engagement would also surface.*

---

## Civic insights (the Stage 8 story, in draft)

Early reads on ~5 years of data. Directional, not final — but this is the shape of the
"what did the data say" section.

### 1. The top of the 311 workload isn't what you'd guess

It's **garbage and recycling carts** (143,459 requests in 5 years) and **property-tax
questions** (~143,000 across two category names). Then bylaw complaints:
snow-and-ice on a neighbour's sidewalk (72,216), graffiti (64,035), long grass and weeds
(50,700). **Potholes are 15th** (~41,400). A triage model scoped to "roads" would be
looking at a small slice of the actual work.

### 2. Most requests close fast; a long tail does not

Median resolution is **3 days**. The mean is **19 days** — pulled there by a heavy tail
(90th percentile ~22 days, longest closed request 1,624 days). Whether something lands in
that tail is almost entirely a function of *category*:

| Fast (days) | Slow (avg days) |
|---|---|
| Noise concerns — 18 | Traffic signs & road-marking — **341** |
| Weed infractions — 27 | Signs missing / damaged — 223 |
| Most waste issues — days | Sidewalk / curb & gutter repair — 70 |
| | Backlane maintenance — 52 |

The slow categories are capital-style work that queues for a season or a budget cycle.
This 300× spread is why the model's "slower than expected" bar is set **per category**,
not as one global number.

### 3. Seasonality is extreme and completely legible

- **Snow-and-ice sidewalk complaints:** 19,827 in January, **3 in July**. It is a
  winter-only service line.
- **Weeds:** the exact mirror — 14 in January, 16,955 in July.
- **Potholes:** spring spike — 8,243 in April vs 544 in December (freeze–thaw).
- **Overall volume:** June (341,869) runs **2× December** (170,992).

The *mix* of work swings hard across the year. A model trained in one season and run in
another is a real risk — which is what the Stage 7 drift monitor exists to catch.

### 4. There is a neighbourhood gradient — and it deserves a careful look

Pothole resolution time by community: **older northwest communities** run slower —
Bowness 37 days, Beddington Heights 29, Huntington Hills 28, Dalhousie 20 — than **newer
southeast / northeast ones** — Mahogany 2.9, Taradale 3.0, Chaparral 3.4, Saddle Ridge
3.4.

Caveats, stated up front: this may reflect road age and type, contractor-zone
boundaries, or how requests are batched, as much as service allocation; one outlier
(Montgomery, 196 days) is almost certainly a data artifact. On the *category-relative*
breach measure the gap narrows to **17%–29%** across 143 communities. But "does 311 work
evenly across the city" is a legitimate question this data can start to answer, and
Stage 8 will report **where the model itself is weakest by area** — a triage tool that's
systematically wrong for one part of the city is a governance problem, not just a metric.

### 5. What predicts a slow resolution

- **The responsible agency matters most.** "OS – Mobility" breaches its category norm
  ~60% of the time; "TRAN – Roads" ~19%. Some teams are structurally behind.
- **Intake channel barely matters** — phone / app / web all breach ~21–22%.
- **No single field predicts much on its own** (all near a coin flip). The signal only
  appears in *combination*: "a weed complaint in July when the weed backlog is already
  high" is predictable in a way that none of those three facts is alone. (This is why a
  tree model reaches ~0.69 ROC-AUC while a linear one gets ~0.55 — see Finding 5.)

### 6. Operational / data-hygiene observations

- **~26,000 requests** in 5 years are explicitly flagged as **duplicates** — residents
  report the same pothole.
- **~70,000 requests are open right now** — the live queue a deployed model would score
  daily.
- Category names have **drifted**: property-tax inquiries filed under two department
  prefixes, "tree concern" split across two back-end systems. Any consumer of this data
  has to maintain an alias map.

---

## Stakeholder lenses

The same findings read differently depending on who's holding them. Stage 8's
narrative-only "who would use this" section is built around the operations-manager lens;
the others are context.

### A city councillor / their office

| What they'd pull from this | Caveat to hold onto |
|---|---|
| **Ward service equity** — older NW communities (Bowness 37 days, Beddington 29, Huntington Hills 28) wait far longer for pothole repair than newer SE/NE ones (Mahogany 2.9, Chaparral 3.4). A direct question to Administration. | Likely reflects road age, infrastructure backlog, and contractor-zone boundaries as much as resource allocation. The category-relative breach measure narrows the gap to 17–29%. |
| **Budget justification with a leading indicator** — backlog + season predicts delays, so seasonal crew funding can be argued *before* the complaint wave (July weeds ≈ 17,000; winter snow/ice ≈ 20,000 in January). | Proxy label, not a City SLA — frame as "requests slower than their own category norm," not "missed targets." |
| **Departmental accountability** — "OS – Mobility" misses its category norm ~60% of the time vs ~19% for Roads. A specific question for the relevant committee. | Category mix differs by agency; some agencies own inherently slow work. |
| **Communications timing** — the seasonal shape is legible enough to set constituent expectations ahead of the annual flashpoints. | — |

### A 311 / service-request operations manager  *(Stage 8's buyer persona)*

| What they'd pull from this | |
|---|---|
| **Predictive triage at intake** — 77% of the requests in the model's top-decile flag breach, vs a 28% base rate (2.8× lift on this sample; 2.0× on the full data, D-037). Enough to route or escalate a small daily shortlist proactively. | The concrete product. |
| **Capacity planning** — category-mix swings 2× across the year and by category up to ~6,600× (snow/ice Jan→Jul). Staffing to an annual average guarantees a winter and a summer crunch. | |
| **Backlog as an early-warning signal** — the 30-day category backlog is one of the few features that carries predictive weight; worth a dashboard even without the model. | |
| **Data-quality debt** — ~26k duplicates, category-name drift across back-end systems, one community with an obvious bulk-close artifact. Cleanup has ROI before any ML. | |

### A Calgary resident

| What they'd pull from this | |
|---|---|
| **Is my request normal or stuck?** — median resolution is 3 days, but it depends entirely on category: a noise complaint (18-day average) and a traffic-sign request (341) are not the same wait. | |
| **Does 311 work evenly across the city?** — the data can start to answer this; the honest answer so far is "there's a gradient, with caveats." | |
| **Transparency** — it's all built on data the City already publishes; anyone can reproduce it (`scripts/explore_311.py`). | |

### City Administration / a data & analytics team

| What they'd pull from this | |
|---|---|
| **A cheap, reproducible service-equity baseline** — before commissioning a study, this is a same-day read on where resolution times diverge by area and category. | |
| **Which categories are unmanaged** — 341-day averages on some sign work suggest a queue nobody is actively watching. | |
| **The case for a canonical taxonomy** — the `service_name` drift is a data-governance problem with a clear owner. | |

---

## Dataset at a glance

| | |
|---|---|
| Rows (all time) | **7,474,403** |
| Rows, last 5 years (`requested_date >= 2021`) | **2,892,237** |
| Last updated | 2026-09-09 17:00 UTC (daily cadence confirmed) |
| Columns | `service_request_id, requested_date, updated_date, closed_date, status_description, source, service_name, agency_responsible, address, comm_code, comm_name, location_type, longitude, latitude, point` + 4 `:@computed_region_*` spatial-join fields |

**Status breakdown, last 5 years:**

| status | n |
|---|---|
| Closed | 2,794,662 |
| Open | 71,826 |
| Duplicate (Closed) | 23,389 |
| Duplicate (Open) | 2,324 |
| TO BE DELETED | 36 |

**Open requests (null `closed_date`), last 5 years: 69,537** — these are exactly what
Batch Transform scores in Stage 6.

---

# Supporting detail

*The findings below are the evidence behind the civic insights above — full tables, the
label check, and the predictive-model numbers.*

## Finding 1 — resolution time varies ~300× by category

Average days-to-close, closed requests, last 5 years (`n > 5,000`), extremes:

| Category | Avg days |
|---|---|
| Roads – Traffic Signs / Roadmarking | 341 |
| Roads – Signs Missing/Damaged | 223 |
| Roads – Signs Parking | 129 |
| Water – Service & Main Valve Issues | 91 |
| Sidewalk / Curb & Gutter Repair | 70 |
| Backlane Maintenance | 52 |
| Long Grass / Weeds Infraction | 27 |
| Bylaw – Noise Concerns | 18 |
| Roads – Pothole Maintenance | ~10 |
| Bylaw – Snow & Ice on Sidewalk | days |

**Implication:** the per-`service_name` threshold in the label design is essential —
without it the model just learns "signs slow, noise fast." Confirmed.

---

## Finding 2 — seasonality is sharp and interpretable

Monthly request counts, last 5 years:

| Category | Pattern (monthly counts) |
|---|---|
| **Snow & Ice on Sidewalk** | Jan 19,827 → **Jul 3** → Dec 17,526. Essentially zero May–Sep. |
| **Long Grass / Weeds** | Jan 14 → **Jul 16,955** → Dec 20. The mirror image. |
| **Pothole Maintenance** | Spring peak (Apr 8,243), classic freeze–thaw. |
| **Cart Management** | Flat (~9k–15k every month) — an administrative category, not weather-driven. |
| All categories | Jun 341,869 vs Dec 170,992 — a 2× annual swing. |

**Implication:** month / season is a real feature, and category-mix shifts hard across the
year — a genuine drift vector for Stage 7 to watch.

---

## Finding 3 — there is a neighbourhood gap

Pothole resolution time by community, last 5 years (`n > 200`):

- **Slowest:** Montgomery 196 days *(almost certainly a data artifact — a bulk backfill
  close; a good "failures you'll hit" example)*, then Bowness 37, Beddington Heights 29,
  Huntington Hills 28, Dalhousie 20 — older NW communities.
- **Fastest:** Stoney 3 (1.8), Mahogany (2.9), Taradale (3.0), Chaparral (3.4), Saddle
  Ridge (3.4) — newer SE/NE communities.

On the **breach** label (which is category-relative, so it strips out "potholes are just
slow here"), the gap compresses: breach rate ranges **17.3%–29.2%** across 143 communities
(`n > 500`), median 22.1%. Present but modest — the honest framing for Stage 8.

Agencies vary more: **"OS – Mobility" 60% breach vs "TRAN – Roads" 19%.**

---

## Finding 4 — the breach label behaves as designed

On a 200k-row sample (closed, last few years):

| | |
|---|---|
| `days_to_close` | median **3.0**, mean **18.8**, p90 22, max 1,624 (heavy right skew) |
| Breach rate (per-category p75 threshold) | **~22%** — matches the ~25% design target |

The label is well-defined and produces the mild imbalance the plan expects
(`scale_pos_weight` / PR-AUC territory).

---

## Finding 5 — breach risk is *modestly* predictable at intake

Temporal split: **train** = 2024 sample (104k rows), **test** = 2025 Jan–Jul (127k rows).
Threshold computed on train only. Features: `service_name`, `agency_responsible`,
`comm_name`, `source`, month, day-of-week, a rough 30-day category backlog count.
(Base breach rate in test: 27.6%.)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Predict base rate | 0.500 | 0.276 |
| Category mean only | ~0.50 | ~0.29 |
| Logistic regression | 0.553 | 0.301 |
| **Gradient-boosted trees** (HistGBM) | **0.686** | **0.540** |
| Random 3-fold CV (whole sample), GBM | 0.554 ± 0.05 | — |

**Operational read — the number that matters for a triage product:**

| If the team reviews the model's… | …share that actually breach | Lift vs base |
|---|---|---|
| top 10% flagged | **77%** | **2.8×** (full data: 35% / 2.0×, D-037) |
| top 25% flagged | 49% | 1.8× |

**Interpretation:**

- A **linear** model finds almost nothing (0.55). Every single feature on its own is
  ≈ 0.50 AUC (best: `service_name` 0.54).
- A **tree** model with interactions gets to ~0.69 ROC / 0.54 PR. So the signal is in
  **interactions** — category × season × backlog — which is exactly what XGBoost should
  capture in Stage 2.
- This is a **"modest predictor"** result. Realistic Stage 2 expectation: ROC-AUC
  **0.68–0.76**, PR-AUC **0.45–0.60** on the full data with the complete feature set.
- Even if the final model lands weak, that is a publishable finding: *"intake-time data
  alone can't strongly predict resolution delay; here's what additional signal — crew
  capacity, weather, work-order detail — would be needed,"* plus the top-decile lift as
  the honest value story.

**Caveats on these numbers:** ~230k-row sample (not the full 2.9M); partial feature set
(no holiday flag, no community density, crude backlog, no top-N category capping);
train→test breach-rate drift (21.7% → 27.6%) that Stage 2's proper as-of discipline will
handle better.

---

## Finding 6 — data-quality traps, confirmed

Each of these feeds a "failures you'll actually hit" section:

1. **Taxonomy fragmentation.** "Property Tax Account Inquiry" appears under both
   `Finance –` and `AT –` prefixes; "Tree Concern" splits into `– GIS` and `– WAM`.
   → alias map + restrict to a recent window (Stage 2).
2. **The Montgomery 196-day pothole outlier** — a per-community average that large is a
   bulk backfill close, not reality. → outlier handling; a Stage 7 anomaly example.
   *Confirmed on the 2026-09-23 full pull:* 38 tickets from 2021–23 were closed on
   2024-10-20, and Montgomery's median is 6 days. The same pull shows backlog clearing is
   routine across the city (for example, 3,505 traffic-sign tickets closed on 2025-08-26).
   See D-034.
3. **4 `:@computed_region_*` fields** Socrata attaches (census tract / ward joins) —
   dropped unless a feature needs one, and then only at area level (see
   `docs/data-and-privacy.md`).
4. **~26k duplicate-status rows** in 5 years — decide to drop or keep as a signal.
5. **Socrata paging is fragile without an app token.** Large cursor + date-range queries
   returned read timeouts during this pass. → confirms Stage 1's teaching points: get the
   free app token, exponential backoff, and freeze a snapshot to S3 on day one.
6. **`requested_date` is date-granular** (midnight) — no hour-of-day features.

---

## Implications for the build

| Decision | Confirmed / adjusted |
|---|---|
| Dataset = Calgary 311 | **Confirmed** — row-level, daily, seasonal, batch-scorable, real civic story |
| Target = per-request binary classification | **Confirmed** — label behaves; signal is modest but real and triage-useful |
| Per-category threshold | **Essential** — keep as designed |
| Stage 2 model | XGBoost (tree, for interactions), `scale_pos_weight`, report **PR-AUC** not just ROC |
| Stage 2 features | add holiday-week, community density; the linear-vs-tree gap says don't bother tuning a linear baseline hard |
| Stage 2 preprocessing | alias map for `service_name`; cap to top ~30 categories + "Other" *(alias map dropped on the full data: D-037)* |
| Stage 7 drift vectors | seasonal category mix, community mix, taxonomy drift — all real |
| Stage 8 framing | lead with top-decile lift (2.8× here, 2.0× on the full data, D-037); be honest that overall accuracy is modest |

---

## Recommendations

Split the same way as the stakeholder lenses: what the findings suggest the **City** could
do, and what they mean for **this project**. The first group is narrative fuel for Stage 8
(it is *not* advice the City asked for); the second is build direction.

### Operational / policy — what the data points to for the City

| # | Recommendation | Basis in the findings |
|---|---|---|
| 1 | **Predictive triage at intake** — flag the daily top-decile of open requests for proactive routing or a courtesy update. | Top-decile flag catches ~77% of eventual breaches vs a 28% base rate (2.8× lift). Sub-dollar batch job. |
| 2 | **Staff to the seasonal curve, not the annual average** — pre-position crews/contractors against the forecast. | Category mix swings 2× across the year; snow/ice, weeds, potholes have fixed, predictable annual shapes. Needs only seasonal decomposition, not ML. |
| 3 | **Backlog-triggered escalation** — a plain threshold alert ("category backlog > X → add capacity"). | The 30-day category backlog is one of the few strong leading indicators of a breach. |
| 4 | **Adopt a canonical `service_name` taxonomy with a named owner.** | Property-tax inquiries filed under two department prefixes; "tree concern" split across two back-end systems. Breaks longitudinal and cross-department reporting. |
| 5 | **Duplicate detection at intake** — check for a similar open request nearby (location + category + time window). | ~26,000 duplicates in 5 years. Also a useful priority signal ("4th report of this pothole"). |
| 6 | **Review the categories nobody is actively watching.** | 341-day average on traffic-sign / road-marking work — either an accepted capital cadence (say so to residents) or an unmanaged queue. |
| 7 | **Give residents category-specific expected-time estimates at intake.** | Median resolution is 3 days but the tail runs to 1,600; residents in the tail get no signal, so they call again and re-report. |
| 8 | **Make service equity a recurring internal report** — ward/community resolution time, normalized for road-km, request volume, infrastructure age. | Older NW communities resolve potholes ~10× slower than newer SE/NE ones; the category-relative gap is 17–29%. Whether a real gap survives normalization is a council resourcing question. |

### Model / pipeline — build direction for this project

| # | Recommendation | Basis in the findings |
|---|---|---|
| 9 | **Tree model, not linear** — XGBoost as planned; don't over-invest in a linear baseline. | Linear ≈ 0.55 ROC, tree ≈ 0.69; every single feature ≈ 0.50 alone. The signal is in interactions. |
| 10 | **Shrink thin-history category thresholds toward a global / agency-level prior** (hierarchical) rather than a hard per-category quantile. | Communities and categories with few historical requests get noisy 75th-percentile thresholds. |
| 11 | **Feature priorities:** keep `agency_responsible` and the category backlog; keep `month` (weight is in its interactions); downweight `source`. | Agency breach rates span 19%–60%; backlog carries weight; `source` is ~21–22% across all channels. |
| 12 | **Treat the train→test breach-rate shift (21.7% → 27.6%) as label drift** — enforce as-of snapshot discipline, add probability recalibration, use it as a Stage 7 monitor test case. | Observed directly in the temporal split. |
| 13 | **Consider modeling time-to-close directly** (regression or survival) as a Stage 2 aside. | The binary throws away information; ops and residents care about "how long," not just "will it be slow." |
| 14 | **Add weather as an exogenous feature** (freeze-thaw days, temperature). | Weather *is* the seasonal signal behind the pothole and snow/ice curves. |
| 15 | **Add a data-validation gate to the pipeline (Stage 3) before training.** | The Montgomery 196-day artifact would silently poison per-community features. Reinforces D-016. |
