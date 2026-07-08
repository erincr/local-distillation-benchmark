"""ld: our local-distillation method at its benchmark configuration.

    lambda_scale = "fixed"   (single global lambda_hat across all test points)
    alpha        = 1.0       (pure lasso)

Needs teacher-prediction OOF on training and on test (the usual TabPFN
setup) plus 5-fold lasso student OOF for the mu estimate.
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import (adelie_lasso_cv, adelie_lasso_predict,
                    local_distill,
                    stability_diagnostics, KFold)


def run(X_tr, y_tr, X_te, y_te, seed, phi, teacher_oof, teacher_test,
        lambda_scale="fixed", alpha=1.0):
    """``phi`` is ``teacher_test`` — we keep the duplicate for symmetry
    with the other methods' signature."""
    t0 = time.time()
    # 5-fold lasso student OOF on training data.
    stu_oof = np.zeros(len(y_tr))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, te_idx in kf.split(X_tr):
        f = adelie_lasso_cv(X_tr[tr_idx], y_tr[tr_idx], n_folds=5, seed=seed)
        stu_oof[te_idx] = adelie_lasso_predict(X_tr[te_idx], f["beta"], f["intercept"])

    gcv = adelie_lasso_cv(X_tr, y_tr, n_folds=5, seed=seed)
    lam = float(gcv["best_lambda"])

    ld = local_distill(
        X_tr, y_tr, X_te,
        teacher_preds_test=teacher_test,
        teacher_oof_preds=teacher_oof,
        student_oof_preds=stu_oof,
        lambda_=lam, random_state=seed,
        lambda_scale=lambda_scale, alpha=alpha)
    stab = stability_diagnostics(ld["betas"], X_te, phi=phi)
    return {"yhat": ld["predictions"], "betas": ld["betas"],
            "intercepts": ld["intercepts"],
            "time_ms_per_test_point": (time.time() - t0) * 1000 / len(y_te),
            "mu": float(ld["mu"]), "used_global": bool(ld["used_global"]),
            "lambda_hat": lam,
            **stab}
