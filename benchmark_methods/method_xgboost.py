"""XGBoost: standalone global non-linear baseline (and alternative teacher).

An XGBRegressor with fixed hyperparameters (see common.XGB_KWARGS).
No per-test-point coefficients, so stability diagnostics are NaN."""
from __future__ import annotations
import time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import xgboost_predict, NAN


def run(X_tr, y_tr, X_te, y_te, seed, phi=None):
    t0 = time.time()
    yhat = xgboost_predict(X_tr, y_tr, X_te, random_state=seed)
    return {"yhat": yhat, "betas": None,
            "time_ms_per_test_point": (time.time() - t0) * 1000 / len(y_te),
            "beta_lipschitz_xspace_median": NAN,
            "jaccard_xspace_mean":          NAN,
            "beta_lipschitz_phi_median":    NAN,
            "jaccard_phi_mean":             NAN,
            "mean_active_set":              NAN}
