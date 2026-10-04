"""Disclosed synthetic drift: a copy of an intake CSV with one documented change (D-045).

This is the known-positive control for Layer A. It shows the analyzer fires on a shift we
made on purpose; the real intake runs show what it says about shifts we didn't make. It
never touches the real data: it reads one CSV and writes another, and the launcher puts
the result under ``monitoring/injected/``.

The change: requests whose ``service_name`` contains ``--pattern`` (default: potholes) are
repeated until they make up ``--share`` of the file (default 15%), and the other rows are
thinned so the file keeps its size. Nothing else moves.

The size is chosen to cross the analyzer's default categorical threshold (an L-infinity
distance of 0.1 between the baseline's and the check's category shares), and the article
says so. A realistic surge doesn't: potholes are about 1% of a September intake, and
tripling them moves the distance by 0.02. So the default threshold can't see a single
category tripling; that is a finding, not a footnote. (Snow and ice was the first pattern
tried; a September intake has none, which is the seasonality lesson in miniature.)

    python -m src.monitor.inject_drift --src intake.csv --out injected.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATTERN = "Pothole"
DEFAULT_SHARE = 0.15
SEED = 7


def inject(
    df: pd.DataFrame, pattern: str = DEFAULT_PATTERN, share: float = DEFAULT_SHARE, seed=SEED
) -> tuple[pd.DataFrame, dict]:
    """``df`` with the matching rows making up ``share`` of it, same row count."""
    hit = df["service_name"].astype("string").str.contains(pattern, case=False, regex=False)
    hit = hit.fillna(False).to_numpy()
    n, k = len(df), int(hit.sum())
    if not k:
        raise SystemExit(f"no rows match {pattern!r}; pick another --pattern")
    if not 0 < share < 1:
        raise ValueError(f"share must be between 0 and 1, got {share}")
    rng = np.random.default_rng(seed)
    n_hit = round(share * n)
    hits = rng.choice(np.flatnonzero(hit), size=n_hit, replace=True)
    rest = rng.choice(np.flatnonzero(~hit), size=n - n_hit, replace=False)
    out = df.iloc[np.sort(np.concatenate([hits, rest]))].reset_index(drop=True)
    note = {
        "synthetic": True,
        "pattern": pattern,
        "rows": n,
        "share_before": round(k / n, 4),
        "share_after": round(n_hit / n, 4),
    }
    return out, note


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="a real intake CSV")
    ap.add_argument("--out", type=Path, required=True, help="the injected copy to write")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="service_name substring")
    ap.add_argument("--share", type=float, default=DEFAULT_SHARE, help="target share (0-1)")
    args = ap.parse_args(argv)
    out, note = inject(pd.read_csv(args.src), args.pattern, args.share)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
