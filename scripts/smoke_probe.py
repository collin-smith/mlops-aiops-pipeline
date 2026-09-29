"""Runs INSIDE the smoke-test Processing job (scripts/smoke_processing.py), not locally.

Prints versions and hardware, then times an XGBoost fit on synthetic data roughly the
shape of one month of 311 requests.
"""

import os
import platform
import time

import numpy as np
import xgboost as xgb

print("python", platform.python_version(), "| xgboost", xgb.__version__)
mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
print("cpus", os.cpu_count(), "| memory_gb", round(mem_gb, 1))

rng = np.random.default_rng(0)
X = rng.random((200_000, 20))
y = (X[:, 0] + rng.random(200_000) > 1.2).astype(int)
t0 = time.time()
params = {"tree_method": "hist", "max_depth": 7, "objective": "binary:logistic"}
xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=100)
print("xgboost_fit_seconds", round(time.time() - t0, 1))
print("SMOKE OK")
