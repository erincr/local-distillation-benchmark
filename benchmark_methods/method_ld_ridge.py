"""LD with ridge (alpha=0) instead of lasso (alpha=1).

Otherwise identical to method_ld. Selects lambda under a ridge-specific CV
so the shrinkage is on the right scale (lasso lambda wouldn't be).
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
import adelie as ad
sys.path.insert(0, str(Path(__file__).parent))
from common import (adelie_lasso_fit, adelie_lasso_predict,
                    local_distill, stability_diagnostics, KFold,
                    _glmnet_min_ratio)


def adelie_ridge_cv(X_train, y_train, n_folds=5, seed=None, weights=None):
    X = np.asfortranarray(X_train.astype(np.float64))
    y = y_train.astype(np.float64)
    n, p = X_train.shape
    glm_obj = (ad.glm.gaussian(y, weights=weights.astype(np.float64))
               if weights is not None else ad.glm.gaussian(y))
    cv_res = ad.cv.cv_grpnet(
        X, glm_obj, n_folds=n_folds, seed=seed, alpha=0.0,
        min_ratio=_glmnet_min_ratio(n, p), progress_bar=False,
    )
    best_lmda = float(cv_res.lmdas[cv_res.best_idx])
    single = adelie_lasso_fit(X_train, y_train, lmda=best_lmda,
                               weights=weights, alpha=0.0)
    return {"best_lambda": best_lmda, "beta": single["beta"],
            "intercept": single["intercept"]}


def run(X_tr, y_tr, X_te, y_te, seed, phi, teacher_oof, teacher_test,
        lambda_scale="fixed"):
    t0 = time.time()
    stu_oof = np.zeros(len(y_tr))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, te_idx in kf.split(X_tr):
        f = adelie_ridge_cv(X_tr[tr_idx], y_tr[tr_idx], n_folds=5, seed=seed)
        stu_oof[te_idx] = adelie_lasso_predict(X_tr[te_idx], f["beta"], f["intercept"])

    gcv = adelie_ridge_cv(X_tr, y_tr, n_folds=5, seed=seed)
    lam = float(gcv["best_lambda"])

    ld = local_distill(
        X_tr, y_tr, X_te,
        teacher_preds_test=teacher_test,
        teacher_oof_preds=teacher_oof,
        student_oof_preds=stu_oof,
        lambda_=lam, random_state=seed,
        lambda_scale=lambda_scale, alpha=0.0)
    stab = stability_diagnostics(ld["betas"], X_te, phi=phi)
    return {"yhat": ld["predictions"], "betas": ld["betas"],
            "intercepts": ld["intercepts"],
            "time_ms_per_test_point": (time.time() - t0) * 1000 / len(y_te),
            "mu": float(ld["mu"]), "used_global": bool(ld["used_global"]),
            "lambda_hat": lam,
            **stab}
