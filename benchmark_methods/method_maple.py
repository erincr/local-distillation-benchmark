"""MAPLE (Plumb, Molitor, Talwalkar 2018).

Thin wrapper around the reference implementation``_maple_ref.py``.  

    1. Random forest (n_estimators=200, max_features=0.5, min_samples_leaf=10)
       fit to y.
    2. Leaf co-membership weights, per-tree normalized by leaf size
       (``training_point_weights``).
    3. DStump feature selection: rank features by impurity at the tree
       root, sweep retain=1..p and pick the value minimising weighted-ridge
       validation RMSE on an internally held-out 20% of the training set.
    4. Per-test-point weighted ridge (alpha=0.001) on the retained features.

Reproducibility note: the ref code does not thread a seed into the
RandomForestRegressor, so we seed numpy globally here.
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
from _maple_ref import MAPLE
from common import stability_diagnostics


def run(X_tr, y_tr, X_te, y_te, seed, phi=None):
    t0 = time.time()
    p = X_tr.shape[1]

    # Internal 80/20 split for DStump's CV; the paper takes (train, val)
    # as two separate arguments.
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_tr, y_tr, test_size=0.20, random_state=seed)

    np.random.seed(seed)  # reference MAPLE doesn't thread a seed
    mdl = MAPLE(X_sub, y_sub, X_val, y_val, fe_type="rf")

    m = len(y_te)
    yhat = np.zeros(m)
    betas = np.zeros((m, p))
    intercepts = np.zeros(m)
    for i in range(m):
        exp = mdl.explain(X_te[i])
        yhat[i] = float(exp["pred"][0])
        intercepts[i] = float(exp["coefs"][0])
        betas[i] = exp["coefs"][1:]

    stab = stability_diagnostics(betas, X_te, phi=phi)
    return {"yhat": yhat, "betas": betas, "intercepts": intercepts,
            "time_ms_per_test_point": (time.time() - t0) * 1000 / m,
            "retain": int(mdl.retain), "n_estimators": mdl.n_estimators,
            **stab}
