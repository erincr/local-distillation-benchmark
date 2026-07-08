"""LOESS: per-test-point weighted OLS, Gaussian kernel in standardized
x-space with bandwidth h = 0.5 * median pairwise training distance.
No penalty (tiny 1e-8 ridge for numerical stability only).  LOESS is
dense by construction — no feature selection — so Jaccard is vacuous
(all ones before we overwrite to NaN) and active_set=p.
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
sys.path.insert(0, str(Path(__file__).parent))
from common import stability_diagnostics, NAN


def run(X_tr, y_tr, X_te, y_te, seed, phi=None):
    t0 = time.time()
    p = X_tr.shape[1]
    h = 0.5 * float(np.median(pdist(X_tr)))
    h2 = max(h * h, 1e-12)
    m = len(y_te)
    yhat = np.zeros(m)
    betas = np.zeros((m, p))
    intercepts = np.zeros(m)
    for i in range(m):
        d = np.sum((X_tr - X_te[i]) ** 2, axis=1)
        w = np.exp(-d / h2)
        wsum = w.sum()
        if wsum < 1e-12:
            yhat[i] = float(np.mean(y_tr))
            continue
        xbar = (w[:, None] * X_tr).sum(0) / wsum
        ybar = (w * y_tr).sum() / wsum
        Xc = X_tr - xbar
        yc = y_tr - ybar
        XtWX = (Xc * w[:, None]).T @ Xc + 1e-8 * np.eye(p)
        XtWy = Xc.T @ (w * yc)
        beta = np.linalg.solve(XtWX, XtWy)
        alpha = ybar - xbar @ beta
        betas[i] = beta
        intercepts[i] = alpha
        yhat[i] = float(X_te[i] @ beta + alpha)
    stab = stability_diagnostics(betas, X_te, phi=phi)
    # LOESS is dense; Jaccards are vacuously 1, so report NaN and override
    # active_set to p.
    stab["mean_active_set"] = float(p)
    stab["jaccard_xspace_mean"] = NAN
    stab["jaccard_phi_mean"] = NAN
    return {"yhat": yhat, "betas": betas, "intercepts": intercepts,
            "time_ms_per_test_point": (time.time() - t0) * 1000 / m,
            "bandwidth_h": h,
            **stab}
