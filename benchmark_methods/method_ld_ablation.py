"""ld_ablation: the two novel cells of local distillation's 2x2 ablation --
the similarity weights S and the teacher pseudo-observation (anchor).  Both
cells share one global mu and the same mu <= 1 fallback, so the difference
between them isolates each component's contribution.

Returns one result per cell:
    ld_anchor   uniform weights, + anchor
    ld_weights  teacher weights S, no anchor

The other two corners of the 2x2 are recovered from methods already computed
elsewhere and are NOT recomputed: (uniform, no anchor) == global_lasso, and
(S, anchor) == ld (method_ld).

Like method_ld, this needs teacher-prediction OOF on training and on test
(the usual TabPFN setup) plus a 5-fold lasso student OOF for the mu estimate.
Both cells are computed in one shared pass; the reported per-test-point time
is that shared cost, identical across the two rows.
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import (adelie_lasso_cv, adelie_lasso_predict,
                    local_distill_ablation,
                    stability_diagnostics, KFold)

# Cell name in local_distill_ablation -> benchmark method name.
CELL_TO_METHOD = {"anchor": "ld_anchor", "weights": "ld_weights"}


def run(X_tr, y_tr, X_te, y_te, seed, phi, teacher_oof, teacher_test):
    """Run both novel ablation cells in one pass.

    Returns ``{method_name: result_dict}`` -- the driver writes one row per
    cell.  ``phi`` is ``teacher_test`` (kept for signature symmetry).
    """
    t0 = time.time()

    # 5-fold lasso student OOF on training data (for the mu estimate).
    stu_oof = np.zeros(len(y_tr))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, te_idx in kf.split(X_tr):
        f = adelie_lasso_cv(X_tr[tr_idx], y_tr[tr_idx], n_folds=5, seed=seed)
        stu_oof[te_idx] = adelie_lasso_predict(X_tr[te_idx], f["beta"], f["intercept"])

    gcv = adelie_lasso_cv(X_tr, y_tr, n_folds=5, seed=seed)
    lam = float(gcv["best_lambda"])

    abl = local_distill_ablation(
        X_tr, y_tr, X_te,
        teacher_preds_test=teacher_test,
        teacher_oof_preds=teacher_oof,
        student_oof_preds=stu_oof,
        lambda_=lam, random_state=seed)

    elapsed_ms_per = (time.time() - t0) * 1000 / len(y_te)

    out = {}
    for cell, mname in CELL_TO_METHOD.items():
        c = abl["cells"][cell]
        stab = stability_diagnostics(c["betas"], X_te, phi=phi)
        out[mname] = {
            "yhat": c["predictions"], "betas": c["betas"],
            "intercepts": c["intercepts"],
            "time_ms_per_test_point": elapsed_ms_per,
            "mu": float(abl["mu"]), "used_global": bool(abl["used_global"]),
            "lambda_hat": lam,
            **stab}
    return out
