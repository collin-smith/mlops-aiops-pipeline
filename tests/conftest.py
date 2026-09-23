import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def requests_frame() -> pd.DataFrame:
    """Synthetic 311 rows: two categories with very different resolution profiles,
    plus a handful of still-open requests."""
    rng = np.random.default_rng(42)
    rows = []
    base = pd.Timestamp("2021-01-01", tz="UTC")

    for i in range(400):
        req = base + pd.Timedelta(days=int(rng.integers(0, 1600)))
        cat = "Pothole Repair" if i % 2 == 0 else "Tree Concern"
        # potholes close fast (median ~3d), trees slow (median ~40d)
        dtc = rng.gamma(2.0, 1.5) if cat == "Pothole Repair" else rng.gamma(4.0, 12.0)
        still_open = i % 20 == 0
        closed = None if still_open else (req + pd.Timedelta(days=float(dtc))).isoformat()
        rows.append(
            {
                "service_request_id": f"SR-{i:05d}",
                "requested_date": req.isoformat(),
                "closed_date": closed,
                "updated_date": (req + pd.Timedelta(days=1)).isoformat(),
                "status_description": "Open" if still_open else "Closed",
                "service_name": cat,
                "agency_responsible": "Roads" if cat == "Pothole Repair" else "Parks",
                "comm_name": rng.choice(["BELTLINE", "BOWNESS", "HILLHURST"]),
                "comm_code": "X",
                "source": rng.choice(["Phone", "App", "Web"]),
                "longitude": -114.07,
                "latitude": 51.05,
            }
        )
    return pd.DataFrame(rows)
