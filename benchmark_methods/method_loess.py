"""LOESS (Cleveland): local linear regression with a nearest-neighbor span
and tricube weights.  Span is chosen by leave-one-out CV on the training
set over SPAN_GRID.  Distances are computed in standardized x-space
(internal, using training stds); the local fit is in the original
coordinates so betas are on the same scale as the other methods.
Rank-deficient local systems are solved by minimum-norm lstsq and
counted in `n_rank_deficient`.  LOESS is dense — no feature selection —
so Jaccard is reported as NaN and active_set = p.
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0, str(Path(__file__).parent))
from common import stability_diagnostics, NAN

SPAN_GRID = (0.2, 0.3, 0.5, 0.75)


def _fit_points(X_tr, y_tr, X_q, D, k, exclude_self=False):
    """Local linear fit at each query point.  D: (m, n) standardized
    distances from queries to training points.  exclude_self drops the
    zero-distance self match (for LOO on the training set)."""
    m, p = X_q.shape
    yhat = np.full(m, np.nan)
    betas = np.full((m, p), np.nan)
    intercepts = np.full(m, np.nan)
    n_eff = np.full(m, np.nan)
    n_rank_def = 0
    for i in range(m):
        d = D[i]
        if exclude_self:
            d = d.copy(); d[i] = np.inf
        idx = np.argpartition(d, k - 1)[:k]
        dk = d[idx].max()
        if dk <= 0:
            yhat[i] = y_tr[idx].mean(); intercepts[i] = yhat[i]; betas[i] = 0.0
            n_eff[i] = k
            continue
        w = (1 - (d[idx] / dk) ** 3) ** 3
        n_eff[i] = w.sum() ** 2 / (w ** 2).sum()
        sw = np.sqrt(w)
        A = np.column_stack([np.ones(k), X_tr[idx]]) * sw[:, None]
        coef, _, rank, _ = np.linalg.lstsq(A, y_tr[idx] * sw, rcond=None)
        n_rank_def += rank < p + 1
        intercepts[i] = coef[0]
        betas[i] = coef[1:]
        yhat[i] = coef[0] + X_q[i] @ coef[1:]
    return yhat, betas, intercepts, n_eff, n_rank_def


def run(X_tr, y_tr, X_te, y_te, seed, phi=None, span_grid=SPAN_GRID):
    t0 = time.time()
    n, p = X_tr.shape
    m = len(y_te)

    sd = X_tr.std(0)
    sd[sd == 0] = 1.0
    D_tr = cdist(X_tr / sd, X_tr / sd)
    D_te = cdist(X_te / sd, X_tr / sd)

    # neighborhood size: span*n, never fewer than p+2 (clipped to n-1 for LOO)
    def k_of(s):
        return int(np.clip(np.ceil(s * n), p + 2, n - 1))

    loo_mse = {}
    for s in span_grid:
        yl = _fit_points(X_tr, y_tr, X_tr, D_tr, k_of(s), exclude_self=True)[0]
        loo_mse[s] = float(np.mean((yl - y_tr) ** 2))
    span = min(loo_mse, key=loo_mse.get)
    k = k_of(span)

    yhat, betas, intercepts, n_eff, n_rank_def = _fit_points(X_tr, y_tr, X_te, D_te, k)

    stab = stability_diagnostics(betas, X_te, phi=phi)
    stab["mean_active_set"] = float(p)
    stab["jaccard_xspace_mean"] = NAN
    stab["jaccard_phi_mean"] = NAN
    return {"yhat": yhat, "betas": betas, "intercepts": intercepts,
            "time_ms_per_test_point": (time.time() - t0) * 1000 / m,
            "span": span, "k_neighbors": k,
            "loo_mse_by_span": loo_mse,
            "n_eff_min": float(np.nanmin(n_eff)),
            "n_eff_median": float(np.nanmedian(n_eff)),
            "n_rank_deficient": int(n_rank_def),
            **stab}
