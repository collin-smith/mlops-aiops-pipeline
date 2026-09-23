#!/usr/bin/env python3
"""Generate the series' architecture diagrams from one model.

This script lives in the public repo (code/scripts/) — it's reusable tooling. What it
generates is article/diagram *content*, which is not: outputs land in ../output/stages/
(a sibling of code/, never committed/pushed), matching the same code-vs-output split used
in the ragproject sibling series.

Outputs (all under ../output/stages/, relative to this file's mlops/ root):
  architecture.drawio            multi-page: Full + Stage 1..8, cumulative, AWS icons
  architecture-network.drawio    the VPC / 2-AZ hardened variant (NOT the default build)
  stage-N-*/architecture.drawio  single-page per stage
  _preview/*.svg                 clean SVG render of every page (review; PNG via cairosvg)

Each AWS service is a draw.io AWS-2024 resource icon (coloured square, white glyph):
`shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.<name>`. Non-AWS nodes (the Socrata
API, the local scripts, GitHub Actions) are plain coloured boxes. The resIcon short names
are in `_RESICON` below — if one renders wrong, fix it there and re-run.

    python code/scripts/architecture_gen.py

stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501  -- the NODES / EDGES tables are intentionally dense
import html
import pathlib
import xml.etree.ElementTree as ET

# The tables below are hand-aligned for readability; keep ruff format off them.
# fmt: off
CAT = {
    "storage":   "#7AA116",
    "analytics": "#8C4FFF",
    "ml":        "#01A88D",
    "compute":   "#ED7100",
    "mgmt":      "#E7157B",
    "security":  "#DD344C",
    "network":   "#8C4FFF",
    "external":  "#5A6B86",
}

# node key -> draw.io AWS-2024 resource-icon short name (used as resIcon=mxgraph.aws4.<name>).
# None => not an AWS service, render as a plain coloured box.
_RESICON = {
    "socrata":     None,
    "pull":        None,
    "gha":         None,
    "drift":       None,
    "s3_raw":      "s3",
    "s3_proc":     "s3",
    "s3_model":    "s3",
    "s3_score":    "s3",
    "glue":        "glue",
    "athena":      "athena",
    "sm_pipe":     "sagemaker",
    "sm_proc":     "sagemaker",
    "sm_train":    "sagemaker",
    "cond":        "sagemaker",
    "registry":    "sagemaker",
    "serverless":  "sagemaker",
    "champ":       "sagemaker",
    "batch":       "sagemaker",
    "monitor":     "sagemaker",
    "ecr":         "ecr",
    "anomaly":     "lambda",
    "approver":    "identity_and_access_management",
    "oidc":        "identity_and_access_management",
    "iam":         "identity_and_access_management",
    "cwmetric":    "cloudwatch",
    "cwlog":       "cloudwatch",
    "ebridge":     "eventbridge",
    "sns":         "sns",
    "budgets":     "budgets",
    "cost":        "cost_explorer",
    # network view
    "s3_svc":      "s3",
    "ecr_svc":     "ecr",
    "sm_svc":      "sagemaker",
    "logs_svc":    "cloudwatch",
    "sts_svc":     "sts",
    "s3gw":        "endpoints",
    "eni_a":       "sagemaker",
    "eni_b":       "sagemaker",
    "vpce_smapi":  "endpoints",
    "vpce_smrt":   "endpoints",
    "vpce_ecrdkr": "endpoints",
    "vpce_ecrapi": "endpoints",
    "vpce_sts":    "endpoints",
    "vpce_logs":   "endpoints",
}
_BOX_MARK = {
    "socrata": "{ }",
    "pull":    "&gt;_",
    "gha":     "CI",
    "drift":   "&gt;_",
}

# key -> (label, sub, cat, _unused_, x, y, stage-introduced)
NODES: dict[str, tuple] = {
    "socrata":    ("Calgary 311", "Socrata API", "external", "mxgraph.aws4.internet_alt1", 20, 130, 1),
    "pull":       ("socrata_pull.py", "local · pull→Parquet", "external", "mxgraph.aws4.command_line_interface", 20, 210, 1),
    "gha":        ("GitHub Actions", "scheduled retrain", "external", "mxgraph.aws4.git", 20, 470, 5),
    "drift":      ("inject_drift.py", "local · synthetic drift", "external", "mxgraph.aws4.command_line_interface", 20, 560, 7),
    "s3_raw":     ("S3  raw/", "asof=YYYY-MM-DD", "storage", "mxgraph.aws4.s3", 250, 120, 1),
    "s3_proc":    ("S3  processed/", "partitioned Parquet", "storage", "mxgraph.aws4.s3", 250, 200, 1),
    "s3_model":   ("S3  model-artifacts/", "model.tar.gz", "storage", "mxgraph.aws4.s3", 250, 290, 2),
    "s3_score":   ("S3  scored/", "+ Athena view", "storage", "mxgraph.aws4.s3", 250, 370, 6),
    "sm_pipe":    ("SageMaker Pipeline", "Process→Train→Eval→Register", "ml", "mxgraph.aws4.sagemaker", 730, 120, 3),
    "glue":       ("AWS Glue", "Catalog + Crawler", "analytics", "mxgraph.aws4.glue", 490, 200, 1),
    "athena":     ("Amazon Athena", "2 GB scan cap", "analytics", "mxgraph.aws4.athena", 730, 200, 1),
    "cond":       ("Condition step", "PR-AUC ≥ floor", "security", "mxgraph.aws4.sagemaker", 970, 200, 4),
    "ecr":        ("Amazon ECR", "XGBoost image", "compute", "mxgraph.aws4.elastic_container_registry", 490, 290, 2),
    "sm_proc":    ("SageMaker Processing", "split / evaluate", "ml", "mxgraph.aws4.sagemaker", 730, 290, 2),
    "sm_train":   ("SageMaker Training", "XGBoost · m5.large spot", "ml", "mxgraph.aws4.sagemaker", 970, 290, 2),
    "registry":   ("SageMaker Model Registry", "Model Package Group", "ml", "mxgraph.aws4.sagemaker", 1210, 120, 4),
    "approver":   ("Approver IAM", "separate principal", "security", "mxgraph.aws4.identity_and_access_management_iam", 1210, 200, 4),
    "champ":      ("champion / challenger", "vs approved model", "ml", "mxgraph.aws4.sagemaker", 1210, 290, 5),
    "serverless": ("SageMaker Serverless", "one-off demo · deleted", "ml", "mxgraph.aws4.sagemaker", 1450, 200, 6),
    "cwmetric":   ("CloudWatch metrics", "MLOpsAIOps/Pipeline", "mgmt", "mxgraph.aws4.cloudwatch", 1450, 290, 3),
    "batch":      ("SageMaker Batch Transform", "score open requests", "ml", "mxgraph.aws4.sagemaker", 730, 380, 6),
    "monitor":    ("SageMaker Model Monitor", "baseline + schedule", "ml", "mxgraph.aws4.sagemaker", 970, 380, 7),
    "anomaly":    ("Anomaly detection", "EWMA · pipeline + civic", "compute", "mxgraph.aws4.lambda", 1210, 380, 7),
    "oidc":       ("IAM OIDC role", "github-actions", "security", "mxgraph.aws4.identity_and_access_management_iam", 250, 470, 5),
    "ebridge":    ("Amazon EventBridge", "schedules", "mgmt", "mxgraph.aws4.eventbridge", 490, 470, 5),
    "iam":        ("IAM roles", "sagemaker · glue", "security", "mxgraph.aws4.identity_and_access_management_iam", 250, 560, 1),
    "cwlog":      ("CloudWatch Logs", "StepFailure alarm", "mgmt", "mxgraph.aws4.cloudwatch", 490, 560, 1),
    "sns":        ("Amazon SNS", "alerts", "mgmt", "mxgraph.aws4.simple_notification_service", 730, 560, 1),
    "budgets":    ("AWS Budgets", "$25 auto-deny action", "mgmt", "mxgraph.aws4.cost_management", 970, 560, 1),
    "cost":       ("AWS Cost Explorer", "docs/cost-log.md", "mgmt", "mxgraph.aws4.cost_management", 1210, 560, 1),
}

# label only the edges that carry non-obvious information; the rest are plain connectors
EDGES: list[tuple] = [
    ("socrata", "pull", "paged pull", 1, False),
    ("pull", "s3_raw", "", 1, False),
    ("pull", "s3_proc", "Parquet snapshot", 1, False),
    ("s3_raw", "s3_proc", "optional Glue ETL", 1, True),
    ("s3_proc", "glue", "crawl", 1, False),
    ("glue", "athena", "", 1, False),
    ("cwlog", "sns", "", 1, False),
    ("budgets", "sns", "", 1, True),
    ("athena", "sm_proc", "training CSV", 2, False),
    ("ecr", "sm_train", "", 2, False),
    ("sm_proc", "sm_train", "", 2, False),
    ("sm_train", "s3_model", "", 2, False),
    ("sm_pipe", "sm_proc", "", 3, False),
    ("sm_pipe", "sm_train", "", 3, False),
    ("sm_pipe", "cwmetric", "run metrics", 3, False),
    ("sm_pipe", "cond", "evaluation.json", 4, False),
    ("cond", "registry", "register if pass", 4, False),
    ("approver", "registry", "manual Approve", 4, False),
    ("ebridge", "gha", "", 5, False),
    ("gha", "oidc", "OIDC", 5, False),
    ("oidc", "sm_pipe", "", 5, False),
    ("sm_pipe", "champ", "", 5, False),
    ("champ", "registry", "promote / deprecate", 5, False),
    ("registry", "batch", "approved model", 6, False),
    ("registry", "serverless", "", 6, True),
    ("batch", "s3_score", "", 6, False),
    ("s3_score", "athena", "", 6, False),
    ("s3_proc", "monitor", "baseline", 7, False),
    ("batch", "monitor", "", 7, False),
    ("drift", "monitor", "synthetic drift", 7, False),
    ("monitor", "sns", "violation", 7, False),
    ("cwmetric", "anomaly", "", 7, False),
    ("anomaly", "sns", "anomaly alarm", 7, False),
]

STAGE_NAMES = {
    1: "The Question & the Data",
    2: "Baseline Model",
    3: "From Notebook to Pipeline",
    4: "Model Registry & Governance Gate",
    5: "CI/CD — Automated Retraining",
    6: "Deployment (Batch Transform)",
    7: "Watching the Watcher (AIOps)",
    8: "The Findings & the Pitch",
}
STAGE_DIRS = {
    1: "stage-1-question-and-data",
    2: "stage-2-baseline-model",
    3: "stage-3-notebook-to-pipeline",
    4: "stage-4-registry-governance",
    5: "stage-5-cicd-retraining",
    6: "stage-6-deployment",
    7: "stage-7-watching-the-watcher",
    8: "stage-8-findings-and-pitch",
}

TILE_W, TILE_H = 204, 52
NOTE_MAIN = (
    "No customer VPC — every service here is regional / managed; SageMaker jobs run without "
    "VpcConfig.  Network-hardened variant -> architecture-network.drawio  (decision D-020)."
)

# --- Network view ---
NET_NODES: dict[str, tuple] = {
    "s3_svc":      ("Amazon S3", "", "storage", "mxgraph.aws4.s3", 120, 120),
    "ecr_svc":     ("Amazon ECR", "", "compute", "mxgraph.aws4.elastic_container_registry", 360, 120),
    "sm_svc":      ("SageMaker API / Runtime", "", "ml", "mxgraph.aws4.sagemaker", 600, 120),
    "logs_svc":    ("CloudWatch Logs", "", "mgmt", "mxgraph.aws4.cloudwatch", 900, 120),
    "sts_svc":     ("AWS STS", "", "security", "mxgraph.aws4.identity_and_access_management_iam", 1140, 120),
    "s3gw":        ("S3 Gateway Endpoint", "route-table · free", "network", "mxgraph.aws4.endpoints", 130, 360),
    "eni_a":       ("SageMaker job ENI", "train / process / transform", "ml", "mxgraph.aws4.sagemaker", 200, 560),
    "vpce_smapi":  ("VPCE sagemaker.api", "interface", "network", "mxgraph.aws4.endpoints", 200, 700),
    "vpce_smrt":   ("VPCE sagemaker.runtime", "interface", "network", "mxgraph.aws4.endpoints", 470, 700),
    "vpce_ecrdkr": ("VPCE ecr.dkr", "interface", "network", "mxgraph.aws4.endpoints", 200, 840),
    "eni_b":       ("SageMaker job ENI", "train / process / transform", "ml", "mxgraph.aws4.sagemaker", 980, 560),
    "vpce_ecrapi": ("VPCE ecr.api", "interface", "network", "mxgraph.aws4.endpoints", 980, 700),
    "vpce_sts":    ("VPCE sts", "interface", "network", "mxgraph.aws4.endpoints", 1250, 700),
    "vpce_logs":   ("VPCE logs", "interface", "network", "mxgraph.aws4.endpoints", 980, 840),
}
NET_GROUPS = [
    ("cloud", 30, 52, 1660, 1018, "AWS Cloud", "#232F3E", 0),
    ("region", 56, 80, 1608, 976, "Region  ca-central-1", "#00A4A6", 1),
    ("vpc", 90, 300, 1540, 740, "VPC  10.0.0.0/16   ·   no Internet Gateway · no NAT Gateway", "#8C4FFF", 0),
    ("az_a", 120, 470, 720, 540, "Availability Zone  ca-central-1a", "#00A4A6", 1),
    ("az_b", 900, 470, 720, 540, "Availability Zone  ca-central-1b", "#00A4A6", 1),
    ("sn_a", 150, 520, 660, 470, "Private subnet  10.0.0.0/24", "#248814", 1),
    ("sn_b", 930, 520, 660, 470, "Private subnet  10.0.1.0/24", "#248814", 1),
]
NET_EDGES = [
    ("eni_a", "vpce_smapi", "", False),
    ("eni_a", "vpce_smrt", "", False),
    ("eni_a", "vpce_ecrdkr", "", False),
    ("eni_b", "vpce_ecrapi", "", False),
    ("eni_b", "vpce_sts", "", False),
    ("eni_b", "vpce_logs", "", False),
    ("vpce_smapi", "sm_svc", "443", False),
    ("vpce_smrt", "sm_svc", "443", False),
    ("vpce_ecrdkr", "ecr_svc", "443", False),
    ("vpce_ecrapi", "ecr_svc", "443", False),
    ("vpce_sts", "sts_svc", "443", False),
    ("vpce_logs", "logs_svc", "443", False),
    ("s3gw", "s3_svc", "prefix list", False),
    ("eni_a", "s3gw", "S3 via gateway", True),
    ("eni_b", "s3gw", "S3 via gateway", True),
]
NOTE_NET = (
    "Fully private: no IGW, no NAT. AWS API traffic stays on the AWS network via VPC endpoints. "
    "~6 interface endpoints x ~$7.30/mo ~= $44/mo -- why the default build omits this (D-020). "
    "SageMaker jobs launched with VpcConfig = {subnets, security groups}."
)

# fmt: on


# ============================ shared geometry ==============================
def _bbox(nodes, keys):
    xs = [nodes[k][4] for k in keys]
    ys = [nodes[k][5] for k in keys]
    return min(xs), min(ys), max(xs) + TILE_W, max(ys) + TILE_H


def _ortho(a, b, nodes):
    ax, ay = nodes[a][4], nodes[a][5]
    bx, by = nodes[b][4], nodes[b][5]
    acx, acy, bcx, bcy = ax + TILE_W / 2, ay + TILE_H / 2, bx + TILE_W / 2, by + TILE_H / 2
    if abs(bcx - acx) >= abs(bcy - acy):
        sx = ax + TILE_W if bcx > acx else ax
        ex = bx if bcx > acx else bx + TILE_W
        mx = (sx + ex) / 2
        pts = [(sx, acy), (mx, acy), (mx, bcy), (ex, bcy)]
    else:
        sy = ay + TILE_H if bcy > acy else ay
        ey = by if bcy > acy else by + TILE_H
        my = (sy + ey) / 2
        pts = [(acx, sy), (acx, my), (bcx, my), (bcx, ey)]
    return pts


# ============================ SVG ==========================================
def _glyph(cat: str, cx: float, cy: float, s: float) -> str:
    """A simple white pictograph centred on (cx, cy), roughly s wide. AWS-icon flavour."""
    h = s / 2
    W = 'stroke="#fff" stroke-width="2.1" fill="none" stroke-linecap="round"'
    F = 'fill="#fff"'
    if cat == "storage":  # stacked disks
        return "".join(
            f'<ellipse cx="{cx}" cy="{cy - h + 3 + k * (s / 3)}" rx="{h}" ry="{s / 7:.1f}" {W}/>'
            for k in range(3)
        )
    if cat == "analytics":  # bar chart
        bw = s / 4
        bars = [(cx - h, 0.45), (cx - h + bw + 2, 0.8), (cx - h + 2 * (bw + 2), 0.62)]
        return "".join(
            f'<rect x="{bx:.1f}" y="{cy + h - (s * hh):.1f}" width="{bw:.1f}" height="{s * hh:.1f}" {F}/>'
            for bx, hh in bars
        )
    if cat == "ml":  # connected nodes
        pts = [(cx - h + 2, cy - h + 3), (cx + h - 3, cy - 2), (cx - 2, cy + h - 3)]
        lines = f'<path d="M{pts[0][0]:.0f} {pts[0][1]:.0f} L{pts[1][0]:.0f} {pts[1][1]:.0f} L{pts[2][0]:.0f} {pts[2][1]:.0f}" {W}/>'
        dots = "".join(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="3" {F}/>' for px, py in pts)
        return lines + dots
    if cat == "compute":  # lambda
        return (
            f'<text x="{cx}" y="{cy + s / 3:.0f}" font-size="{s * 1.1:.0f}" text-anchor="middle" '
            f'font-family="Georgia,serif" font-weight="700" fill="#fff">&#955;</text>'
        )
    if cat == "mgmt":  # pulse line
        return (
            f'<path d="M{cx - h:.0f} {cy:.0f} h{s * 0.28:.0f} l{s * 0.12:.0f} -{h:.0f} '
            f'l{s * 0.16:.0f} {s:.0f} l{s * 0.12:.0f} -{h:.0f} h{s * 0.28:.0f}" {W}/>'
        )
    if cat == "security":  # shield
        return (
            f'<path d="M{cx:.0f} {cy - h:.0f} L{cx + h:.0f} {cy - h + 4:.0f} '
            f"V{cy + 2:.0f} Q{cx + h:.0f} {cy + h:.0f} {cx:.0f} {cy + h + 2:.0f} "
            f"Q{cx - h:.0f} {cy + h:.0f} {cx - h:.0f} {cy + 2:.0f} "
            f'V{cy - h + 4:.0f} Z" {F}/>'
        )
    if cat == "network":  # hexagon + dot
        return (
            f'<path d="M{cx - h:.0f} {cy:.0f} L{cx - h / 2:.0f} {cy - h:.0f} '
            f"L{cx + h / 2:.0f} {cy - h:.0f} L{cx + h:.0f} {cy:.0f} "
            f'L{cx + h / 2:.0f} {cy + h:.0f} L{cx - h / 2:.0f} {cy + h:.0f} Z" {W}/>'
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="2.6" {F}/>'
        )
    # external: terminal prompt
    return (
        f'<path d="M{cx - h + 2:.0f} {cy - h / 2:.0f} l{h:.0f} {h / 2:.0f} l-{h:.0f} {h / 2:.0f}" {W}/>'
        f'<path d="M{cx:.0f} {cy + h - 2:.0f} h{h - 2:.0f}" {W}/>'
    )


BADGE = "ring"  # "ring" = thin pink outline + pink label; "new" = ring + NEW tag; "none"


def _tile(x, y, label, sub, cat, accent):
    col = CAT[cat]
    S = 42  # icon square
    iy = y + (TILE_H - S) / 2
    g = []
    # solid category-coloured icon square + white glyph
    g.append(f'<rect x="{x}" y="{iy:.0f}" width="{S}" height="{S}" rx="6" fill="{col}"/>')
    g.append(_glyph(cat, x + S / 2, iy + S / 2, S * 0.52))
    # "added this stage" cue
    lab_col = "#16191F"
    if accent and BADGE != "none":
        g.append(
            f'<rect x="{x - 3}" y="{iy - 3:.0f}" width="{S + 6}" height="{S + 6}" rx="8" '
            f'fill="none" stroke="#D6336C" stroke-width="2"/>'
        )
        lab_col = "#B02A5B"
        if BADGE == "new":
            g.append(
                f'<rect x="{x + S - 12}" y="{iy - 12:.0f}" width="32" height="14" rx="3" fill="#D6336C"/>'
                f'<text x="{x + S + 4}" y="{iy - 2:.0f}" font-size="8" text-anchor="middle" '
                f'font-weight="700" fill="#fff">NEW</text>'
            )
    ty = y + (TILE_H / 2) - (2 if sub else -4)
    g.append(
        f'<text x="{x + S + 10}" y="{ty:.0f}" font-size="11" font-weight="700" '
        f'fill="{lab_col}">{html.escape(label)}</text>'
    )
    if sub:
        g.append(
            f'<text x="{x + S + 10}" y="{ty + 15:.0f}" font-size="9" fill="#5A6B86">'
            f"{html.escape(sub)}</text>"
        )
    return "<g>" + "".join(g) + "</g>"


def _grp(x, y, w, h, label, stroke, dashed):
    d = 'stroke-dasharray="8 5"' if dashed else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" stroke="{stroke}" '
        f'stroke-width="1.4" {d}/>'
        f'<rect x="{x + 10}" y="{y - 9}" width="{9 * len(label) + 16}" height="18" rx="3" fill="#F7F8FA"/>'
        f'<text x="{x + 18}" y="{y + 4}" font-size="12" font-weight="800" fill="{stroke}">'
        f"{html.escape(label)}</text>"
    )


def _edge(a, b, label, dashed, nodes):
    pts = _ortho(a, b, nodes)
    d = 'stroke-dasharray="6 4"' if dashed else ""
    path = "M" + " L".join(f"{px:.0f},{py:.0f}" for px, py in pts)
    out = f'<path d="{path}" fill="none" stroke="#AEB6C4" stroke-width="1.2" {d} marker-end="url(#arw)"/>'
    if label:
        # label on the connector segment (pts[1]-pts[2]) so labels spread out instead of
        # bunching near each source tile
        mx, my = (pts[1][0] + pts[2][0]) / 2, (pts[1][1] + pts[2][1]) / 2
        w = 5.6 * len(label) + 6
        out += (
            f'<rect x="{mx - w / 2:.0f}" y="{my - 7:.0f}" width="{w:.0f}" height="13" rx="2" '
            f'fill="#F7F8FA" opacity="0.95"/>'
            f'<text x="{mx:.0f}" y="{my + 3:.0f}" font-size="8" text-anchor="middle" '
            f'fill="#5A6B86">{html.escape(label)}</text>'
        )
    return out


def _wrap(text, n):
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > n:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    lines.append(line)
    return lines


def render_svg(
    nodes,
    groups,
    edges,
    title,
    note,
    *,
    keep=None,
    accent_stage=None,
    external=(),
    autofit_group=None,
):
    vis = [k for k in nodes if keep is None or k in keep]
    # optionally recompute one group box to fit content
    grp = list(groups)
    if autofit_group is not None:
        inner = [k for k in vis if k not in external]
        x0, y0, x1, y1 = _bbox(nodes, inner)
        pad = 34
        region = (x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad)
        cloud = (region[0] - 18, region[1] - 30, region[2] + 36, region[3] + 48)
        grp = [
            ("cloud", *cloud, "AWS Cloud", "#232F3E", 0),
            ("region", *region, "Region  ca-central-1", "#00A4A6", 1),
        ]
        bottom = cloud[1] + cloud[3]
    else:
        bottom = max(g[2] + g[4] for g in grp)  # y + h
    note_lines = _wrap(note, 150)
    H = bottom + 24 + 15 * len(note_lines) + 20
    W = max([g[1] + g[3] for g in grp] + [nodes[k][4] + TILE_W for k in vis]) + 30

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
        'font-family="Helvetica,Arial,sans-serif">',
        '<defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#AEB6C4"/></marker></defs>',
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="#F7F8FA"/>',
        f'<text x="34" y="34" font-size="16" font-weight="800" fill="#16191F">{html.escape(title)}</text>',
    ]
    for g in grp:
        p.append(_grp(g[1], g[2], g[3], g[4], g[5], g[6], g[7]))
    for e in edges:
        a, b = e[0], e[1]
        if a in vis and b in vis:
            p.append(_edge(a, b, e[2], e[-1], nodes))
    for k in vis:
        n = nodes[k]
        stg = n[6] if len(n) > 6 else None
        p.append(
            _tile(n[4], n[5], n[0], n[1], n[2], accent_stage is not None and stg == accent_stage)
        )
    for i, ln in enumerate(note_lines):
        p.append(
            f'<text x="34" y="{bottom + 26 + i * 15:.0f}" font-size="10" fill="#5A6B86" '
            f'font-style="italic">{html.escape(ln)}</text>'
        )
    p.append("</svg>")
    return "".join(p)


# ============================ draw.io ======================================
def _icon_style(name: str, cat: str) -> str:
    # canonical draw.io AWS resource-icon style — coloured square, white glyph
    return (
        "sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;"
        f"fillColor={CAT[cat]};strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;"
        "verticalAlign=top;align=center;html=1;fontSize=10;fontStyle=0;aspect=fixed;"
        f"shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.{name};"
    )


def _box_style(cat: str) -> str:
    return (
        f"rounded=1;arcSize=22;html=1;fillColor={CAT[cat]};strokeColor=none;"
        "fontColor=#ffffff;fontSize=12;fontStyle=1;verticalAlign=middle;align=center;"
    )


def _drawio_page(model, nodes, groups, edges, title, note, *, keep=None, accent_stage=None):
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    ET.SubElement(
        root,
        "mxCell",
        {
            "id": "t",
            "parent": "1",
            "vertex": "1",
            "value": title,
            "style": "text;html=1;align=left;fontSize=15;fontStyle=1;",
        },
    ).append(_geo(40, 16, 1400, 26))

    for i, g in enumerate(groups):
        _, x, y, w, h, label, stroke, dashed = g
        ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"g{i}",
                "parent": "1",
                "vertex": "1",
                "value": label,
                "style": (
                    f"rounded=1;arcSize=2;html=1;whiteSpace=wrap;fillColor=none;strokeColor={stroke};"
                    f"dashed={dashed};verticalAlign=top;align=left;spacingLeft=8;fontColor={stroke};"
                    "fontSize=12;fontStyle=1;"
                ),
            },
        ).append(_geo(x, y, w, h))

    for k, n in nodes.items():
        if keep is not None and k not in keep:
            continue
        stg = n[6] if len(n) > 6 else None
        x, y = n[4], n[5]
        new_here = bool(accent_stage) and stg == accent_stage
        # the icon (AWS resource icon) or a plain box for non-AWS nodes.
        # edges attach to THIS cell (id = k)
        ic = _RESICON.get(k)
        if ic is None:
            ET.SubElement(
                root,
                "mxCell",
                {
                    "id": k,
                    "parent": "1",
                    "vertex": "1",
                    "value": _BOX_MARK.get(k, ""),
                    "style": _box_style(n[2]),
                },
            ).append(_geo(x, y, 46, 46))
        else:
            ET.SubElement(
                root,
                "mxCell",
                {
                    "id": k,
                    "parent": "1",
                    "vertex": "1",
                    "value": "",
                    "style": _icon_style(ic, n[2]),
                },
            ).append(_geo(x, y, 48, 48))
        # "added this stage" cue
        if new_here and BADGE != "none":
            ET.SubElement(
                root,
                "mxCell",
                {
                    "id": f"{k}_r",
                    "parent": "1",
                    "vertex": "1",
                    "value": "",
                    "style": "rounded=1;arcSize=18;html=1;fillColor=none;strokeColor=#D6336C;strokeWidth=2;",
                },
            ).append(_geo(x - 3, y - 3, 54, 54))
            if BADGE == "new":
                ET.SubElement(
                    root,
                    "mxCell",
                    {
                        "id": f"{k}_n",
                        "parent": "1",
                        "vertex": "1",
                        "value": "NEW",
                        "style": "text;html=1;align=center;fontSize=8;fontStyle=1;fontColor=#ffffff;fillColor=#D6336C;strokeColor=none;rounded=1;",
                    },
                ).append(_geo(x + 30, y - 14, 30, 13))
        name_col = " color='#B02A5B'" if (new_here and BADGE != "none") else ""
        lbl = f"<b><font{name_col}>{n[0]}</font></b>" + (
            f"<br><font style='font-size:9px' color='#5A6B86'>{n[1]}</font>" if n[1] else ""
        )
        ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"{k}_l",
                "parent": "1",
                "vertex": "1",
                "value": lbl,
                "style": "text;html=1;align=left;verticalAlign=middle;fontSize=11;spacing=2;",
            },
        ).append(_geo(x + 56, y - 3, 200, 54))

    for j, e in enumerate(edges):
        a, b = e[0], e[1]
        if keep is not None and (a not in keep or b not in keep):
            continue
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;fontSize=9;endArrow=block;"
            "labelBackgroundColor=#FFFFFF;strokeColor=#8893A8;" + ("dashed=1;" if e[-1] else "")
        )
        ec = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"e{j}",
                "parent": "1",
                "edge": "1",
                "source": a,
                "target": b,
                "value": e[2],
                "style": style,
            },
        )
        ec.append(ET.Element("mxGeometry", {"relative": "1", "as": "geometry"}))

    ny = max(g[2] + g[4] for g in groups) + 22
    ET.SubElement(
        root,
        "mxCell",
        {
            "id": "note",
            "parent": "1",
            "vertex": "1",
            "value": note,
            "style": "text;html=1;align=left;fontSize=10;fontColor=#5A6B86;fontStyle=2;",
        },
    ).append(_geo(40, ny, 1500, 46))


def _geo(x, y, w, h):
    return ET.Element(
        "mxGeometry",
        {"x": str(int(x)), "y": str(int(y)), "width": str(w), "height": str(h), "as": "geometry"},
    )


def _mxfile(pages):
    f = ET.Element("mxfile", {"host": "app.diagrams.net"})
    for name, kind, stage in pages:
        d = ET.SubElement(f, "diagram", {"name": name, "id": kind + str(stage or "")})
        m = ET.SubElement(
            d,
            "mxGraphModel",
            {
                "dx": "1400",
                "dy": "900",
                "grid": "0",
                "page": "1",
                "pageWidth": "1760",
                "pageHeight": "1180",
                "math": "0",
            },
        )
        if kind == "net":
            _drawio_page(
                m,
                NET_NODES,
                NET_GROUPS,
                NET_EDGES,
                "Network — VPC-hardened variant  (NOT the default build)",
                NOTE_NET,
            )
        else:
            s = stage
            keep = None if kind == "full" else {k for k, v in NODES.items() if v[6] <= s}
            groups = [
                ("c", 196, 70, 1520, 628, "AWS Cloud", "#232F3E", 0),
                ("r", 222, 92, 1470, 576, "Region  ca-central-1", "#00A4A6", 1),
            ]
            title = (
                "Full architecture — MLOps + AIOps pipeline on AWS (Calgary 311)"
                if kind == "full"
                else f"Stage {s} — {STAGE_NAMES[s]}   ·   architecture by end of stage"
            )
            # accent the delta only from Stage 2 on (Stage 1 is entirely new — no signal)
            acc = s if (s and s > 1) else None
            note = NOTE_MAIN + ("   Pink ring + label = introduced in this stage." if acc else "")
            _drawio_page(m, NODES, groups, EDGES, title, note, keep=keep, accent_stage=acc)
    return ET.ElementTree(f)


# ============================ main ========================================
def _out_dir() -> pathlib.Path:
    """../output/stages/, relative to this script's mlops/ root — never inside code/."""
    mlops_root = pathlib.Path(__file__).resolve().parents[2]
    out = mlops_root / "output" / "stages"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    here = _out_dir()
    prev = here / "_preview"
    prev.mkdir(exist_ok=True)

    pages = [("Full architecture", "full", None)] + [
        (f"Stage {s} — {STAGE_NAMES[s]}", "stage", s) for s in range(1, 9)
    ]
    _mxfile(pages).write(here / "architecture.drawio", encoding="utf-8", xml_declaration=True)
    _mxfile([("Network — VPC hardened variant", "net", None)]).write(
        here / "architecture-network.drawio", encoding="utf-8", xml_declaration=True
    )
    for s, d in STAGE_DIRS.items():
        if (here / d).exists():
            _mxfile([(f"Stage {s} — {STAGE_NAMES[s]}", "stage", s)]).write(
                here / d / "architecture.drawio", encoding="utf-8", xml_declaration=True
            )
    print("wrote architecture.drawio + architecture-network.drawio + per-stage files")

    ext = ("socrata", "pull", "gha", "drift")
    (prev / "full.svg").write_text(
        render_svg(
            NODES,
            [],
            EDGES,
            "Full architecture — MLOps + AIOps pipeline (Calgary 311)",
            NOTE_MAIN,
            external=ext,
            autofit_group=True,
        )
    )
    for s in range(1, 9):
        keep = {k for k, v in NODES.items() if v[6] <= s}
        acc = s if s > 1 else None
        (prev / f"stage{s}.svg").write_text(
            render_svg(
                NODES,
                [],
                EDGES,
                f"Stage {s} — {STAGE_NAMES[s]}   ·   architecture by end of stage",
                NOTE_MAIN + ("   Pink ring + label = introduced in this stage." if acc else ""),
                keep=keep,
                accent_stage=acc,
                external=ext,
                autofit_group=True,
            )
        )
    (prev / "network.svg").write_text(
        render_svg(
            NET_NODES,
            NET_GROUPS,
            NET_EDGES,
            "Network — VPC-hardened variant  (NOT the default build)",
            NOTE_NET,
        )
    )
    print(f"wrote {len(list(prev.glob('*.svg')))} SVG previews -> {prev}")


def main_compare_newbadges():
    """Write a side comparison file with the old rose NEW badges (does not touch the
    generated diagrams). Open output/stages/_compare-new-badges.drawio in draw.io."""
    global BADGE
    here = _out_dir()
    BADGE = "new"
    pages = [("Full architecture", "full", None)] + [
        (f"Stage {s} — {STAGE_NAMES[s]}", "stage", s) for s in range(1, 9)
    ]
    _mxfile(pages).write(
        here / "_compare-new-badges.drawio", encoding="utf-8", xml_declaration=True
    )
    BADGE = "ring"
    print("wrote output/stages/_compare-new-badges.drawio  (NEW-badge style, for comparison only)")


if __name__ == "__main__":
    import sys

    if "--new-badges" in sys.argv:
        main_compare_newbadges()
    else:
        main()
