# Data & privacy

The pipeline runs entirely on **City of Calgary 311 Service Requests** (`iahh-g8bj`),
published as open data under the [City of Calgary Open Data Terms of Use][terms].

[terms]: https://data.calgary.ca/stories/s/u45n-7awa

## What's in the data

- Each row is one service request: category, responsible agency, intake channel,
  request/close timestamps, status, and a **location** — historically an `address`
  string plus community name/code and a lat/long point.
- **No personal identifiers.** No requester name, phone, email, or account number. The
  City de-identifies at source; 311 records are released specifically for public
  transparency about service delivery.

## What this project keeps

| Field | Raw zone | Processed zone (what the model sees) |
|---|---|---|
| `address` | retained as delivered | **dropped** |
| `longitude` / `latitude` / `point` | retained | retained (used only for community-level density features) |
| `comm_name` / `comm_code` | retained | retained — **community/ward is the finest geographic grain used** |
| requester identity | *not in the dataset* | — |

The processed zone (and everything downstream — features, training data, scored output)
is aggregated to **community granularity or coarser**. No model input, and no published
figure, is at address-level.

## What this project does not do

- No linkage to any other dataset that could re-identify a requester.
- No attempt to infer who filed a request.
- The `:@computed_region_*` fields Socrata attaches (census tract / ward spatial joins)
  are dropped unless a specific feature needs one, and then only at area level.

## Fairness note

Stage 8 reports where the model **under-performs** — the categories and communities with
the weakest predictions — because a triage model that's systematically wrong for one part
of the city is a governance problem, not just an accuracy one. This is called out in the
model card (`docs/model-card.md`) and the Stage 4 / Stage 8 articles.
