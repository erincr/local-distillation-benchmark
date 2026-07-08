"""Global ridge baseline: adelie CV-selected lambda, ridge (alpha=0) on
standardized X.

The ridge analogue of method_global_lasso, and the honest baseline for the
ridge students (ld_ridge / ld_xgb_ridge / ld_tabfm_ridge) -- i.e. the
(uniform weights, no anchor) corner of the ridge side.  Uses the same
ridge-specific CV as method_ld_ridge so the shrinkage is on the right scale
(a lasso lambda would not be).  Ridge is dense, so betas are the tiled global
vector: beta-Lipschitz is 0 and Jaccard 1 by construction (reported for
completeness only).
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import adelie_lasso_predict, stability_diagnostics
from method_ld_ridge import adelie_ridge_cv


def run(X_tr, y_tr, X_te, y_te, seed, phi=None):
    t0 = time.time()
    cv = adelie_ridge_cv(X_tr, y_tr, n_folds=5, seed=seed)
    yhat = adelie_lasso_predict(X_te, cv["beta"], cv["intercept"])
    betas = np.tile(cv["beta"], (len(y_te), 1))
    stab = stability_diagnostics(betas, X_te, phi=phi)
    return {"yhat": yhat, "betas": betas,
            "time_ms_per_test_point": (time.time() - t0) * 1000 / len(y_te),
            "lambda_hat": float(cv["best_lambda"]),
            **stab}
