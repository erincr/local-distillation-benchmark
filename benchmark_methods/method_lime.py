"""LIME — continuous variant, hand-rolled to batch all teacher calls.

The ``lime`` library calls the teacher once per test point, which for
our 200-test-point benchmarks at num_samples=1000 costs ~7 min per
(dataset, seed) cell on CPU.  All m * num_samples perturbation rows
are stacked into ONE TabPFN predict() call for speed.

Algorithm (LIME continuous / ``discretize_continuous=False`` variant):
    1. Fit TabPFN on (X_tr, y_tr) once.
    2. For each test point x_*, build a num_samples x p perturbation
       block.  Row 0 is x_* itself; rows 1.. are N(train_mean, train_std)
       iid per feature (same distribution regardless of x_*).
    3. Stack all m blocks, do ONE teacher predict() on the (m*num_samples, p)
       matrix.
    4. Per test point: weight the num_samples rows by the LIME kernel
       w = sqrt(exp(-d^2 / kw^2)),  kw = 0.75 * sqrt(p), d in
       standardized x-space.  Fit a weighted lasso (5-fold CV lambda)
       on those rows → coefficients and intercept.

Two deviations from the LIME paper:
    (1) Continuous perturbations (``discretize_continuous=False``)
        instead of quartile discretization. 
    (2) Weighted lasso (CV-chosen lambda) as the local surrogate,
        instead of the paper's K-LASSO feature selection + OLS refit
        (or the library's Ridge(alpha=1) default).
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import stability_diagnostics, adelie_lasso_cv


def run(X_tr, y_tr, X_te, y_te, seed, LIME_NUM_SAMPLES=1000, phi=None):
    from tabpfn import TabPFNRegressor
    t0 = time.time()
    n, p = X_tr.shape
    m = len(y_te)
    rng = np.random.default_rng(seed)

    from common import TABPFN_N_ESTIMATORS, TABPFN_DEVICE
    reg = TabPFNRegressor(device=TABPFN_DEVICE, ignore_pretraining_limits=True,
                            n_estimators=TABPFN_N_ESTIMATORS)
    reg.fit(X_tr, y_tr)

    feat_mean = X_tr.mean(0)
    feat_std = X_tr.std(0) + 1e-12  # ~1 in our outer pipeline (X_tr is pre-standardized)

    # (m, num_samples, p) perturbations.  Row 0 of each block is x_*.
    X_pert = np.empty((m, LIME_NUM_SAMPLES, p))
    X_pert[:, 0, :] = X_te
    X_pert[:, 1:, :] = rng.standard_normal((m, LIME_NUM_SAMPLES - 1, p)) * feat_std + feat_mean

    # Teacher call on the (m * num_samples, p) stack, chunked.
    X_flat = X_pert.reshape(-1, p)
    LIME_PREDICT_BATCH = 2000
    pieces = [reg.predict(X_flat[s:s + LIME_PREDICT_BATCH]).astype(float)
              for s in range(0, X_flat.shape[0], LIME_PREDICT_BATCH)]
    yss = np.concatenate(pieces).reshape(m, LIME_NUM_SAMPLES)

    # Per-test-point kernel-weighted lasso (5-fold CV lambda) on
    # standardized perturbations.
    Xs = (X_pert - feat_mean) / feat_std
    kw2 = (0.75 * np.sqrt(p)) ** 2
    betas = np.zeros((m, p)); intercepts = np.zeros(m); yhat = np.zeros(m)
    for i in range(m):
        d2 = ((Xs[i] - Xs[i, 0]) ** 2).sum(axis=1)
        w = np.sqrt(np.exp(-d2 / kw2))
        cv = adelie_lasso_cv(Xs[i], yss[i], n_folds=5, seed=seed, weights=w)
        betas[i] = cv["beta"]
        intercepts[i] = float(cv["intercept"])
        yhat[i] = float(Xs[i, 0] @ betas[i] + intercepts[i])

    stab = stability_diagnostics(betas, X_te, phi=phi)
    return {"yhat": yhat, "betas": betas, "intercepts": intercepts,
            "time_ms_per_test_point": (time.time() - t0) * 1000 / m,
            "num_samples": LIME_NUM_SAMPLES,
            **stab}
