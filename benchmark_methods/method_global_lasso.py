"""Global lasso baseline: adelie CV-selected lambda on standardized X.

The β vector is global but we tile it across all test points so the
stability diagnostics are well-defined.  β-Lipschitz is therefore 0
by construction, and Jaccard is 1.  These are "vacuous wins"; they're
reported for completeness only.
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import (adelie_lasso_cv, adelie_lasso_predict,
                    stability_diagnostics)


def run(X_tr, y_tr, X_te, y_te, seed, phi=None):
    t0 = time.time()
    cv = adelie_lasso_cv(X_tr, y_tr, n_folds=5, seed=seed)
    yhat = adelie_lasso_predict(X_te, cv["beta"], cv["intercept"])
    betas = np.tile(cv["beta"], (len(y_te), 1))
    stab = stability_diagnostics(betas, X_te, phi=phi)
    return {"yhat": yhat, "betas": betas,
            "time_ms_per_test_point": (time.time() - t0) * 1000 / len(y_te),
            "lambda_hat": float(cv["best_lambda"]),
            **stab}
