#!/usr/bin/env bash
# Run at the end of every build session. Deletes or stops anything that bills while idle.
# Does NOT run `terraform destroy` — that is a deliberate, separate decision.
#
# Usage:
#   AWS_PROFILE=mlops-aiops ./scripts/nuke.sh           # clean up, then report spend
#   AWS_PROFILE=mlops-aiops ./scripts/nuke.sh --check   # report only, change nothing (D-030)
#
# --check exits 1 if anything that bills while idle is found, so "clean" is a yes/no answer.
# Run it last before every pause.
set -euo pipefail

REGION="${REGION:-ca-central-1}"
PROJECT="${MLOPS_PROJECT:-mlops-aiops}"
CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1
FOUND=0

say() { printf '\n=== %s ===\n' "$1"; }
# found <description> <command...>: report the item; outside --check, also run the command.
found() {
  local what="$1"
  shift
  FOUND=$((FOUND + 1))
  if ((CHECK)); then
    echo "!! $what"
  else
    "$@" >/dev/null
    echo "fixed: $what"
  fi
}
clean() { echo "none — good"; }
sm() { aws sagemaker "$@" --region "$REGION" --output text; }

((CHECK)) && echo "CHECK MODE: reporting only, nothing will be changed."

# Fail early and clearly, rather than halfway through with a raw API error. The usual cause
# on WSL is clock drift after Windows sleeps: most AWS APIs reject a request signed more
# than 5 minutes off.
if ! err=$(aws sts get-caller-identity --region "$REGION" --output text 2>&1 >/dev/null); then
  echo
  if [[ "$err" == *"Signature expired"* || "$err" == *"SignatureDoesNotMatch"* ]]; then
    echo "!! The local clock is off, so AWS rejects the request. Fix it, then re-run:"
    echo "     sudo ntpdate time.windows.com"
  else
    echo "!! Can't reach AWS with the current credentials (is AWS_PROFILE set?):"
    echo "   $err"
  fi
  echo "NOT CHECKED: nothing was inspected or changed."
  exit 2
fi

say "SageMaker endpoints (there should be NONE; the Stage 6 serverless demo deletes its own)"
items=$(sm list-endpoints --query 'Endpoints[].EndpointName')
for e in $items; do found "endpoint $e" sm delete-endpoint --endpoint-name "$e"; done
[[ -z "${items// /}" ]] && clean

say "SageMaker monitoring schedules (Stage 7 — the biggest idle risk, ~\$165/mo if hourly)"
items=$(sm list-monitoring-schedules --query 'MonitoringScheduleSummaries[].MonitoringScheduleName')
for s in $items; do
  found "monitoring schedule $s" sm delete-monitoring-schedule --monitoring-schedule-name "$s"
done
[[ -z "${items// /}" ]] && clean

say "In-flight SageMaker jobs (bill until they finish)"
n=0
for kind in training processing transform; do
  cap="$(tr '[:lower:]' '[:upper:]' <<<"${kind:0:1}")${kind:1}"
  for j in $(sm "list-${kind}-jobs" --status-equals InProgress \
    --query "${cap}JobSummaries[].${cap}JobName"); do
    n=$((n + 1))
    found "$kind job $j InProgress" sm "stop-${kind}-job" "--${kind}-job-name" "$j"
  done
done
((n == 0)) && clean

say "SageMaker pipeline executions still running"
n=0
for p in $(sm list-pipelines --query 'PipelineSummaries[].PipelineName'); do
  for x in $(sm list-pipeline-executions --pipeline-name "$p" \
    --query "PipelineExecutionSummaries[?PipelineExecutionStatus=='Executing'].PipelineExecutionArn"); do
    n=$((n + 1))
    found "pipeline $p execution Executing" sm stop-pipeline-execution --pipeline-execution-arn "$x"
  done
done
((n == 0)) && clean

say "Classic notebook instances (bill per hour while InService)"
items=$(sm list-notebook-instances --status-equals InService --query 'NotebookInstances[].NotebookInstanceName')
for i in $items; do found "notebook instance $i InService" sm stop-notebook-instance --notebook-instance-name "$i"; done
[[ -z "${items// /}" ]] && clean

say "SageMaker Studio apps (bill per hour while running)"
n=0
for line in $(aws sagemaker list-apps --region "$REGION" \
  --query "Apps[?Status=='InService'].[DomainId,UserProfileName,AppType,AppName]" \
  --output text | tr '\t' ',' || true); do
  IFS=',' read -r domain user atype aname <<<"$line"
  [[ -z "$aname" || "$aname" == "None" ]] && continue
  n=$((n + 1))
  found "Studio app $aname ($atype)" aws sagemaker delete-app --domain-id "$domain" \
    --user-profile-name "$user" --app-type "$atype" --app-name "$aname" --region "$REGION"
done
((n == 0)) && clean

say "Scheduled starts: Glue triggers, EventBridge rules + schedules named ${PROJECT}*"
n=0
for t in $(aws glue list-triggers --region "$REGION" --query 'TriggerNames' --output text); do
  [[ "$t" == ${PROJECT}* ]] || continue
  n=$((n + 1))
  found "Glue trigger $t" aws glue delete-trigger --name "$t" --region "$REGION"
done
for r in $(aws events list-rules --name-prefix "$PROJECT" --region "$REGION" \
  --query "Rules[?State=='ENABLED'].Name" --output text); do
  n=$((n + 1))
  found "EventBridge rule $r ENABLED" aws events disable-rule --name "$r" --region "$REGION"
done
for s in $(aws scheduler list-schedules --name-prefix "$PROJECT" --state ENABLED \
  --region "$REGION" --query 'Schedules[].Name' --output text 2>/dev/null); do
  n=$((n + 1))
  # Report only: disabling needs the schedule's full definition; do it in the console.
  FOUND=$((FOUND + 1))
  echo "!! EventBridge Scheduler schedule $s ENABLED (disable it in the console)"
done
((n == 0)) && clean

say "Glue crawler"
state=$(aws glue get-crawler --name "${PROJECT}-processed-311" --region "$REGION" \
  --query 'Crawler.State' --output text 2>/dev/null || echo "absent")
if [[ "$state" == "RUNNING" ]]; then
  found "crawler RUNNING" aws glue stop-crawler --name "${PROJECT}-processed-311" --region "$REGION"
else
  echo "crawler ${state,,} — good"
fi

say "GitHub Actions retrain schedule (outside AWS; runs even while you're paused)"
wf="$(dirname "$0")/../.github/workflows/retrain.yml"
if [[ -f "$wf" ]] && grep -qE '^[[:space:]]+schedule:' "$wf"; then
  FOUND=$((FOUND + 1))
  echo "!! retrain.yml has an active schedule: comment it out unless the monthly run is intended"
else
  echo "no active schedule — good"
fi

# Cost Explorer sums floats, so $0 can come back as -1e-09. Print dollars and cents.
usd() { awk '{ v = $1 + 0; if (v < 0.005 && v > -0.005) v = 0; printf "$%.2f\n", v }'; }

say "Spend: account-wide since the series start (what the \$25 hard stop counts)"
series_start="${MLOPS_SERIES_START:-2026-09-01}"
aws ce get-cost-and-usage --region us-east-1 \
  --time-period "Start=${series_start},End=$(date -u -d tomorrow +%Y-%m-%d)" \
  --granularity MONTHLY --metrics UnblendedCost \
  --query 'sum(ResultsByTime[].to_number(Total.UnblendedCost.Amount))' --output text 2>/dev/null \
  | usd || echo "(cost explorer not queryable yet — check the console)"

say "Month-to-date spend tagged project=${PROJECT} (blind until the tag is activated)"
start=$(date -u +%Y-%m-01)
aws ce get-cost-and-usage --region us-east-1 \
  --time-period "Start=${start},End=$(date -u -d tomorrow +%Y-%m-%d)" \
  --granularity MONTHLY --metrics UnblendedCost \
  --filter "{\"Tags\":{\"Key\":\"project\",\"Values\":[\"${PROJECT}\"]}}" \
  --query 'ResultsByTime[0].Total.UnblendedCost.Amount' --output text 2>/dev/null \
  | usd || echo "(cost explorer not queryable yet — check the console)"
echo "(Cost Explorer lags up to ~24 h, and each query costs \$0.01: run --check before a pause, not in a loop.)"

echo
if ((CHECK)); then
  if ((FOUND)); then
    echo "NOT CLEAN: $FOUND item(s) still billing or scheduled. Run ./scripts/nuke.sh to fix."
    exit 1
  fi
  echo "CLEAN: nothing billing while idle. Safe to pause."
else
  echo "done ($FOUND item(s) fixed). Confirm with: ./scripts/nuke.sh --check"
  echo "Custom CloudWatch metrics can't be deleted but stop billing once you stop emitting."
  echo "Full teardown when the series ends:  terraform -chdir=infra destroy"
fi
