"""Athena views over the shadow scores (Stage 6, D-044).

``shadow_scores`` itself is a table (``infra/shadow.tf``). ``shadow_outcomes`` joins each
score to what has since happened to the request, in whatever snapshot ``processed_311``
holds now (D-036: always the latest). It is the hook Stage 7 grades the shadow model on.

An outcome is decided once the request closed (on time or late) or once the latest snapshot
was taken after its deadline with the request still open (late). Until a later snapshot
than the scored one exists, nearly every row is ``undecided``, and that's expected.
Stage 7 grades only rows whose deadline passed before the outcome snapshot
(``src/deploy/grade.py``); ``threshold_days`` and ``category_seen`` are here for that report.

DDL statements cost nothing in Athena; ``CREATE OR REPLACE`` makes this safe to re-run.
"""

from __future__ import annotations

SHADOW_OUTCOMES = """
CREATE OR REPLACE VIEW shadow_outcomes AS
WITH snap AS (
  SELECT date_add('day', 1, date_trunc('day', max(requested_date))) AS taken_at
  FROM processed_311
)
SELECT
  s.usage,
  s.asof AS scored_asof,
  s.model_ref,
  s.service_request_id,
  s.service_name,
  s.comm_name,
  s.sector,
  s.srg,
  s.score,
  s.rank_in_sector,
  s.flagged,
  s.deadline,
  s.threshold_days,
  s.category_seen,
  p.closed_date,
  snap.taken_at AS outcome_snapshot,
  CASE
    WHEN p.closed_date IS NOT NULL AND p.closed_date <= s.deadline THEN 'on_time'
    WHEN p.closed_date IS NOT NULL THEN 'late'
    WHEN snap.taken_at > s.deadline THEN 'late'
    ELSE 'undecided'
  END AS outcome
FROM shadow_scores s
JOIN processed_311 p ON p.service_request_id = s.service_request_id
CROSS JOIN snap
""".strip()

VIEWS = {"shadow_outcomes": SHADOW_OUTCOMES}

# Stage 7's grading rows (D-045): every score, for ``src.deploy.grade`` to dedupe and filter.
GRADE_ROWS = """
SELECT usage, scored_asof, service_request_id, service_name, sector, srg, flagged,
       deadline, threshold_days, category_seen, outcome, outcome_snapshot
FROM shadow_outcomes
""".strip()

# Stage 7's ★ query: the grade in one table, for the console screenshot. Each request on its
# first score, only where the deadline passed before the outcome snapshot (src.deploy.grade).
SHADOW_GRADE = """
WITH first AS (
  SELECT *, row_number() OVER (PARTITION BY service_request_id ORDER BY scored_asof) AS rn
  FROM shadow_outcomes
)
SELECT usage,
       flagged,
       count(*) AS requests,
       sum(CASE WHEN outcome = 'late' THEN 1 ELSE 0 END) AS late,
       round(avg(CASE WHEN outcome = 'late' THEN 1.0 ELSE 0.0 END), 3) AS late_rate
FROM first
WHERE rn = 1 AND deadline < outcome_snapshot
GROUP BY usage, flagged
ORDER BY flagged DESC
""".strip()

# The article's ★ query: the usage column stays in the crop.
TOP_FLAGGED = """
SELECT usage, service_name, comm_name, sector, score, rank_in_sector
FROM shadow_scores
WHERE asof = '{asof}' AND flagged
ORDER BY score DESC
LIMIT 20
""".strip()


def create_views(run_query) -> list[str]:
    """Create or replace every view; ``run_query`` is ``src.common.athena.run_query``."""
    for sql in VIEWS.values():
        run_query(sql)
    return list(VIEWS)
