"""Stage 1 — pull City of Calgary 311 Service Requests from the Socrata Open Data API.

Subcommands
-----------
  count        print the live row count + dataset metadata (updatedAt, columns)
  pull         paged pull of the last N years into local newline-delimited JSON (gzipped)
  communities  pull the community lookup (comm_code -> sector, srg) for the fairness check
  to-parquet   convert the local NDJSON into partitioned Parquet (year=/month=)
  snapshot     upload the local raw + processed data to S3 under asof=<date>/
  replay       download a previously-frozen S3 snapshot back to local (no API call)

The Socrata trap this stage teaches: a naive ``GET .../iahh-g8bj.json`` returns only
1,000 rows and gives you no error. You page. Offset paging (``$offset``) also has a
ceiling on large datasets, so this uses **cursor paging on the system ``:id`` column**
(``$order=:id`` + ``$where=:id > <last>``), which is stable and has no ceiling.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from src.common.config import SOCRATA_SELECT_COLUMNS, get_config

log = logging.getLogger("socrata_pull")

PAGE_SIZE = 50_000
ROWS_PER_FILE = 500_000
MAX_RETRIES = 6
DATA_ROOT = Path("data")
SYS_ID = ":id"  # Socrata system row id — sortable, unique, no offset ceiling
COMMUNITY_COLUMNS = ("comm_code", "name", "sector", "srg", "class")


# --------------------------------------------------------------------------- HTTP


def _session() -> requests.Session:
    s = requests.Session()
    cfg = get_config()
    s.headers.update({"User-Agent": "mlops-aiops-pipeline/0.1 (+github.com/collin-smith)"})
    if cfg.socrata_app_token:
        s.headers["X-App-Token"] = cfg.socrata_app_token
    else:
        log.warning("no SOCRATA_APP_TOKEN set — you will be throttled harder on a big pull")
    return s


def _get(session: requests.Session, url: str, params: dict) -> list[dict]:
    """GET with exponential backoff + jitter on 429/5xx."""
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(url, params=params, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            sleep = min(60, 2**attempt) + random.uniform(0, 1.5)
            log.warning(
                "HTTP %s (attempt %d/%d) — backing off %.1fs",
                resp.status_code,
                attempt,
                MAX_RETRIES,
                sleep,
            )
            time.sleep(sleep)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"gave up after {MAX_RETRIES} retries: {url}")


# --------------------------------------------------------------------------- count


def cmd_count(args: argparse.Namespace) -> None:
    cfg = get_config()
    session = _session()

    where = _window_where(args.years)
    params = {"$select": "count(*) as n"}
    if where:
        params["$where"] = where
    n = int(_get(session, cfg.socrata_base_url, params)[0]["n"])

    meta = session.get(cfg.socrata_metadata_url, timeout=60).json()
    updated = datetime.fromtimestamp(meta.get("rowsUpdatedAt", 0), tz=UTC)
    cols = [c["fieldName"] for c in meta.get("columns", [])]

    print(f"resource      : {cfg.socrata_domain}/{cfg.socrata_resource}")
    print(f"window        : {where or 'all rows'}")
    print(f"row count     : {n:,}")
    print(f"rows updated  : {updated:%Y-%m-%d %H:%M UTC}")
    print(f"columns ({len(cols)}) : {', '.join(cols)}")


def _window_where(years: int | None) -> str:
    if not years:
        return ""
    start_year = datetime.now(UTC).year - years
    return f"requested_date >= '{start_year}-01-01T00:00:00'"


# --------------------------------------------------------------------------- pull


def cmd_pull(args: argparse.Namespace) -> None:
    cfg = get_config()
    session = _session()
    asof = args.asof or datetime.now(UTC).strftime("%Y-%m-%d")
    out_dir = DATA_ROOT / "raw" / "311" / f"asof={asof}"
    out_dir.mkdir(parents=True, exist_ok=True)

    window = _window_where(args.years)
    select = ",".join((SYS_ID, *SOCRATA_SELECT_COLUMNS))

    last_id: str | None = None
    total = 0
    file_idx = 0
    buf: list[dict] = []
    t0 = time.monotonic()

    while True:
        clauses = [window] if window else []
        if last_id is not None:
            clauses.append(f"{SYS_ID} > '{last_id}'")
        params = {
            "$select": select,
            "$order": SYS_ID,
            "$limit": PAGE_SIZE,
        }
        if clauses:
            params["$where"] = " AND ".join(clauses)

        page = _get(session, cfg.socrata_base_url, params)
        if not page:
            break

        last_id = page[-1][SYS_ID]
        buf.extend(page)
        total += len(page)
        log.info("pulled %d rows (%.0f rows/s)", total, total / max(time.monotonic() - t0, 1e-6))

        if len(buf) >= ROWS_PER_FILE:
            _flush(buf, out_dir, file_idx)
            file_idx += 1
            buf = []

        if len(page) < PAGE_SIZE:
            break

    if buf:
        _flush(buf, out_dir, file_idx)

    manifest = {
        "resource": cfg.socrata_resource,
        "asof": asof,
        "window_where": window,
        "row_count": total,
        "pulled_at": datetime.now(UTC).isoformat(),
        "page_size": PAGE_SIZE,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("done: %d rows across %d file(s) in %s", total, file_idx + (1 if buf else 0), out_dir)


def _flush(rows: list[dict], out_dir: Path, idx: int) -> None:
    path = out_dir / f"part-{idx:05d}.ndjson.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            row.pop(SYS_ID, None)
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    log.info("wrote %s (%d rows)", path, len(rows))


# --------------------------------------------------------------------- to-parquet


def cmd_to_parquet(args: argparse.Namespace) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    asof = args.asof or _latest_asof()
    raw_dir = DATA_ROOT / "raw" / "311" / f"asof={asof}"
    files = sorted(raw_dir.glob("part-*.ndjson.gz"))
    if not files:
        sys.exit(f"no NDJSON under {raw_dir} — run `pull` first")

    out_root = DATA_ROOT / "processed" / "311"
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for f in files:
        df = pd.read_json(f, lines=True, dtype=False)
        for col in ("requested_date", "updated_date", "closed_date"):
            if col in df:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True, format="ISO8601")
        for col in ("longitude", "latitude"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["requested_date"])
        df["year"] = df["requested_date"].dt.year.astype("int16")
        df["month"] = df["requested_date"].dt.month.astype("int8")

        pq.write_to_dataset(
            pa.Table.from_pandas(df, preserve_index=False),
            root_path=str(out_root),
            partition_cols=["year", "month"],
            existing_data_behavior="overwrite_or_ignore",
            compression="snappy",
        )
        total += len(df)
        log.info("%s -> parquet (%d rows, running total %d)", f.name, len(df), total)

    log.info("done: %d rows partitioned under %s", total, out_root)


def _latest_asof() -> str:
    dirs = sorted((DATA_ROOT / "raw" / "311").glob("asof=*"))
    if not dirs:
        sys.exit("no local raw snapshots found")
    return dirs[-1].name.split("=", 1)[1]


# ------------------------------------------------------------------- snapshot / replay


# --------------------------------------------------------------------------- communities


def cmd_communities(args: argparse.Namespace) -> None:
    """~313 rows, one page. Frozen next to the 311 snapshot so the Stage 4 fairness check
    groups a run's predictions the same way every time it's replayed."""
    cfg = get_config()
    asof = args.asof or datetime.now(UTC).strftime("%Y-%m-%d")
    rows = _get(
        _session(),
        cfg.socrata_communities_url,
        {"$select": ",".join(COMMUNITY_COLUMNS), "$limit": 5000},
    )
    out = DATA_ROOT / "raw" / "communities" / f"asof={asof}" / "communities.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    log.info("wrote %s (%d communities)", out, len(rows))


def _s3():
    import boto3

    return boto3.client("s3", region_name=get_config().region)


def cmd_snapshot(args: argparse.Namespace) -> None:
    cfg = get_config()
    asof = args.asof or _latest_asof()
    s3 = _s3()

    pairs = [
        (DATA_ROOT / "raw" / "311" / f"asof={asof}", f"raw/311/asof={asof}"),
        (DATA_ROOT / "raw" / "communities" / f"asof={asof}", f"raw/communities/asof={asof}"),
        (DATA_ROOT / "processed" / "311", "processed/311"),
    ]
    for local_dir, key_prefix in pairs:
        if not local_dir.exists():
            log.warning("skip %s (does not exist)", local_dir)
            continue
        for path in local_dir.rglob("*"):
            if path.is_file():
                key = f"{key_prefix}/{path.relative_to(local_dir).as_posix()}"
                s3.upload_file(str(path), cfg.bucket, key)
                log.info("s3://%s/%s", cfg.bucket, key)
    log.info("snapshot asof=%s uploaded to s3://%s", asof, cfg.bucket)


def cmd_replay(args: argparse.Namespace) -> None:
    cfg = get_config()
    asof = args.asof
    if not asof:
        sys.exit("--asof is required for replay")
    s3 = _s3()

    for key_prefix, local_root in (
        (f"raw/311/asof={asof}", DATA_ROOT / "raw" / "311" / f"asof={asof}"),
        (f"raw/communities/asof={asof}", DATA_ROOT / "raw" / "communities" / f"asof={asof}"),
        ("processed/311", DATA_ROOT / "processed" / "311"),
    ):
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=cfg.bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(key_prefix) :].lstrip("/")
                dest = local_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(cfg.bucket, obj["Key"], str(dest))
                log.info("<- s3://%s/%s", cfg.bucket, obj["Key"])
    log.info("replayed snapshot asof=%s from s3://%s", asof, cfg.bucket)


# --------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="socrata_pull", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("count", help="live row count + metadata")
    c.add_argument("--years", type=int, default=5)
    c.set_defaults(func=cmd_count)

    pl = sub.add_parser("pull", help="paged pull -> local NDJSON")
    pl.add_argument("--years", type=int, default=5, help="last N years of requests (0 = all)")
    pl.add_argument("--asof", help="snapshot date label (default: today, UTC)")
    pl.set_defaults(func=cmd_pull)

    cm = sub.add_parser("communities", help="community lookup -> local JSON")
    cm.add_argument("--asof", help="snapshot date label (default: today, UTC)")
    cm.set_defaults(func=cmd_communities)

    tp = sub.add_parser("to-parquet", help="local NDJSON -> partitioned Parquet")
    tp.add_argument("--asof", help="which raw snapshot to convert (default: latest local)")
    tp.set_defaults(func=cmd_to_parquet)

    sn = sub.add_parser("snapshot", help="upload local raw + processed to S3")
    sn.add_argument("--asof", help="snapshot date label (default: latest local)")
    sn.set_defaults(func=cmd_snapshot)

    rp = sub.add_parser("replay", help="download a frozen S3 snapshot back to local")
    rp.add_argument("--asof", required=True)
    rp.set_defaults(func=cmd_replay)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
