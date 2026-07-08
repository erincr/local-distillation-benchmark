"""Shared helpers for the benchmark method scripts.
Contents:
    adelie_lasso_cv, adelie_lasso_fit, adelie_lasso_predict
    teacher_similarity
    tabpfn_predict, tabpfn_cv
    xgboost_predict, xgboost_cv
    tabfm_predict, tabfm_cv
    local_distill
    load_dataset, split_and_standardize
    stability_diagnostics
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import adelie as ad
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import tabfm as _tabfm
    HAS_TABFM = True
except ImportError:
    HAS_TABFM = False

XGB_KWARGS = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                  n_jobs=1, verbosity=0)

ROOT = Path(__file__).resolve().parent.parent
NAN = float("nan")

# TabPFN ensemble size.  Default is 32; we use 8 — benchmarked to be
# ~4x faster with negligible MSE change across our dataset panel.
TABPFN_N_ESTIMATORS = 8

# TabPFN model generation — pinned explicitly so the teacher is fixed
# regardless of which tabpfn package version is installed (the package's own
# default has moved v2 -> v2.5 -> v2.6 -> v3 across releases).  v3 = TabPFN-3.
# We resolve by ENUM MEMBER NAME (V3, V2_6, ...) rather than by string value,
# because the enum's *values* differ across tabpfn releases (e.g. "v3" is not
# always a valid value).  Override with env TABPFN_MODEL_VERSION in either
# form, e.g. "V3" / "v3" / "v2.6" / "V2_6".
if HAS_TABPFN:
    import os as _os_mv, tabpfn as _tabpfn_pkg
    _mv_req = _os_mv.environ.get("TABPFN_MODEL_VERSION", "V3")
    _mv_name = _mv_req.upper().replace(".", "_")  # "v2.6" -> "V2_6", "v3" -> "V3"
    try:
        TABPFN_MODEL_VERSION = ModelVersion[_mv_name]
    except KeyError as _e:
        raise ValueError(
            f"TABPFN_MODEL_VERSION={_mv_req!r} (member {_mv_name!r}) is not "
            f"available in installed tabpfn "
            f"{getattr(_tabpfn_pkg, '__version__', '?')}; valid members: "
            f"{[m.name for m in ModelVersion]}. Upgrade tabpfn (e.g. "
            f"`pip install tabpfn==8.0.3`) for TabPFN-3 (V3)."
        ) from _e

# TabPFN device.  Override with env var TABPFN_DEVICE={cuda,mps,cpu}; otherwise
# auto-detect CUDA > MPS > CPU.  MPS needs PYTORCH_ENABLE_MPS_FALLBACK=1 so
# unsupported ops fall back to CPU.
import os as _os
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
def _autodetect_device():
    env = _os.environ.get("TABPFN_DEVICE")
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
TABPFN_DEVICE = _autodetect_device()


# --------------------------------------------------------------------------- #
# Adelie lasso / elastic-net wrappers                                          #
# --------------------------------------------------------------------------- #
def _glmnet_min_ratio(n, p):
    """glmnet's lambda.min.ratio convention.  Adelie's default (0.1) can
    truncate the path before reaching the CV-optimal lambda on n > p
    problems."""
    return 0.01 if n < p else 0.0001


def adelie_lasso_cv(X_train, y_train, n_folds=5, seed=None, weights=None):
    """Fit lasso with CV lambda selection via adelie.
    ``weights`` are observation weights (same length as y_train); None = uniform.
    Returns dict with ``best_lambda``, ``beta``, ``intercept``, ``cv_result``.
    """
    X = np.asfortranarray(X_train.astype(np.float64))
    y = y_train.astype(np.float64)
    n, p = X_train.shape

    if weights is None:
        glm_obj = ad.glm.gaussian(y)
    else:
        glm_obj = ad.glm.gaussian(y, weights=weights.astype(np.float64))

    cv_res = ad.cv.cv_grpnet(
        X, glm_obj, n_folds=n_folds, seed=seed,
        min_ratio=_glmnet_min_ratio(n, p), progress_bar=False,
    )
    best_lmda = float(cv_res.lmdas[cv_res.best_idx])                                                                                                        
                                                                                                                                                              
    # Refit at best_lmda using the single-lambda solver that every                                                                                          
    # per-test-point fit downstream uses.  Path-fit vs single-lambda                                                                                        
    # cold-start differ by ~1e-2 in practice (glmnet-family behavior);                                                                                      
    # using the same solver everywhere keeps global_lasso and ld on
    # one β scale.                                                                                                                                          
    single = adelie_lasso_fit(X_train, y_train, lmda=best_lmda,
                                weights=weights, alpha=1.0)                                                                                                  
                                                                                                                                                              
    return {"best_lambda": best_lmda, "beta": single["beta"],
            "intercept": single["intercept"], "cv_result": cv_res}   


def adelie_lasso_predict(X, beta, intercept):
    return X @ beta + intercept


def adelie_lasso_fit(X_train, y_train, lmda, weights=None, alpha=1.0):
    """Fit elastic-net at a single lambda via adelie.  Penalty:
        lambda * [alpha * ||beta||_1 + (1-alpha)/2 * ||beta||_2^2]
    alpha=1.0 → pure lasso.  Always fits an unpenalized intercept.
    """
    X = np.asfortranarray(X_train.astype(np.float64))
    y = y_train.astype(np.float64)
    if weights is not None:
        glm_obj = ad.glm.gaussian(y, weights=weights.astype(np.float64))
    else:
        glm_obj = ad.glm.gaussian(y)
    state = ad.solver.grpnet(
        X, glm_obj, alpha=alpha,
        lmda_path=np.array([lmda]),
        intercept=True,
        progress_bar=False,
    )
    beta = state.betas[0].toarray().flatten()
    ic = float(state.intercepts[0])
    return {"beta": beta, "intercept": ic}


# --------------------------------------------------------------------------- #
# TabPFN wrappers                                                              #
# --------------------------------------------------------------------------- #
def _make_tabpfn(device):
    """TabPFNRegressor pinned to TABPFN_MODEL_VERSION (default v3), so the
    teacher is the same model regardless of installed package version."""
    if not HAS_TABPFN:
        raise RuntimeError("TabPFN not installed.  `pip install tabpfn`")
    return TabPFNRegressor.create_default_for_version(
        TABPFN_MODEL_VERSION, device=device,
        ignore_pretraining_limits=True, n_estimators=TABPFN_N_ESTIMATORS)


def tabpfn_predict(X_train, y_train, X_test, device=None):
    """Fit TabPFN on (X_train, y_train) and return predictions on X_test."""
    if device is None: device = TABPFN_DEVICE
    reg = _make_tabpfn(device)
    reg.fit(X_train, y_train)
    return reg.predict(X_test).astype(float)


def tabpfn_cv(X_train, y_train, n_folds=5, device=None, random_state=None):
    """K-fold out-of-fold TabPFN predictions on the training set."""
    if device is None: device = TABPFN_DEVICE
    n = X_train.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(n)
    reg = _make_tabpfn(device)
    for tr, te in kf.split(X_train):
        reg.fit(X_train[tr], y_train[tr])
        oof[te] = reg.predict(X_train[te]).astype(float)
    return {"predictions": oof, "residuals": y_train - oof}


# --------------------------------------------------------------------------- #
# XGBoost wrappers (second teacher for the benchmark)                          #
# --------------------------------------------------------------------------- #
def xgboost_predict(X_train, y_train, X_test, random_state=None):
    """Fit XGBoost on (X_train, y_train) and return predictions on X_test."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost not installed.  `pip install xgboost`")
    reg = XGBRegressor(random_state=random_state, **XGB_KWARGS)
    reg.fit(X_train, y_train)
    return reg.predict(X_test).astype(float)


def xgboost_cv(X_train, y_train, n_folds=5, random_state=None):
    """K-fold out-of-fold XGBoost predictions on the training set."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost not installed.  `pip install xgboost`")
    n = X_train.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(n)
    for tr, te in kf.split(X_train):
        reg = XGBRegressor(random_state=random_state, **XGB_KWARGS)
        reg.fit(X_train[tr], y_train[tr])
        oof[te] = reg.predict(X_train[te]).astype(float)
    return {"predictions": oof, "residuals": y_train - oof}


# --------------------------------------------------------------------------- #
# TabFM wrappers (third teacher) — google-research/tabfm, PyTorch backend      #
# --------------------------------------------------------------------------- #
# API verified against examples/regression_example.py (main, 2026-07):
#   model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression")
#   reg   = tabfm.TabFMRegressor(model=model); reg.fit(X, y); reg.predict(Xte)
# Pretrained weights load once (HF Hub) and are cached module-side; each fit
# only re-seeds the in-context training set (no gradient training).  Fed the
# same standardized matrix as TabPFN/XGBoost, so it is an apples-to-apples
# third teacher rather than TabFM's native raw mixed-type DataFrame path.
_TABFM_MODEL = None


def _get_tabfm_model():
    if not HAS_TABFM:
        raise RuntimeError(
            "tabfm not installed.  Install from source: "
            "`git clone https://github.com/google-research/tabfm && "
            "cd tabfm && pip install -e .[pytorch]`")
    global _TABFM_MODEL
    if _TABFM_MODEL is None:
        # load() defaults to device=None == CPU; on CPU TabFM is ~200x slower
        # (917s vs seconds for n=127).  Force the resolved GPU device (same one
        # TabPFN uses); override with TABFM_DEVICE.
        device = _os.environ.get("TABFM_DEVICE", TABPFN_DEVICE)
        _TABFM_MODEL = _tabfm.tabfm_v1_0_0_pytorch.load(
            model_type="regression", device=device)
    return _TABFM_MODEL


def tabfm_predict(X_train, y_train, X_test):
    """Fit TabFM in-context on (X_train, y_train) and predict X_test.
    Mirrors tabpfn_predict; live refit per call (no caching)."""
    reg = _tabfm.TabFMRegressor(model=_get_tabfm_model())
    reg.fit(X_train, y_train)
    return np.asarray(reg.predict(X_test), dtype=float)


def tabfm_cv(X_train, y_train, n_folds=5, random_state=None):
    """K-fold out-of-fold TabFM predictions on the training set (mirror of tabpfn_cv)."""
    n = X_train.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(n)
    model = _get_tabfm_model()
    for tr, te in kf.split(X_train):
        reg = _tabfm.TabFMRegressor(model=model)
        reg.fit(X_train[tr], y_train[tr])
        oof[te] = np.asarray(reg.predict(X_train[te]), dtype=float)
    return {"predictions": oof, "residuals": y_train - oof}


# --------------------------------------------------------------------------- #
# Teacher-similarity weights                                                   #
# --------------------------------------------------------------------------- #
def teacher_similarity(teacher_oof, teacher_test, epsilon=1e-12):
    """similarity matrix from scalar teacher predictions.  
    ``S[i, j]`` is softmax over Gaussian-distance
    (in teacher-prediction space, bandwidth var(teacher_oof))."""
    sigma2 = max(float(np.var(teacher_oof)), epsilon)
    d = (teacher_oof[None, :] - teacher_test[:, None]) ** 2 / sigma2
    log_S = -d - logsumexp(-d, axis=1, keepdims=True)
    return np.exp(log_S)

# --------------------------------------------------------------------------- #
# Local distillation
# --------------------------------------------------------------------------- #
def local_distill(X_train, y_train, X_test,
                   teacher_preds_test, teacher_oof_preds, student_oof_preds,
                   lambda_=None, epsilon=1e-8, random_state=None,
                   min_n_ratio = .05,
                   lambda_scale="fixed", alpha=1.0):
    """Per-test-point local distillation.

    1. μ = MSE_OOF(student) / MSE_OOF(teacher).  If μ ≤ 1, return global
       lasso for all test points (teacher gives no edge).
    2. Similarity S[i, :] = softmax over teacher predictions.
    3. Per-test-point λ: ``lambda_hat`` or scaled by sqrt(n/max(n_eff_i, n_min)).
    4. Fit weighted (elastic-net, α defaults to 1 = pure lasso) on the
       augmented (training + anchor) design and read off per-test-point
       coefficients.
    """
    if lambda_scale not in ("fixed", "n_eff"):
        raise ValueError(f"unknown lambda_scale: {lambda_scale}")

    n, m = X_train.shape[0], X_test.shape[0]
    p = X_train.shape[1]

    cv_result = adelie_lasso_cv(X_train, y_train, n_folds=5, seed=random_state)
    global_beta = cv_result["beta"]
    global_intercept = float(cv_result["intercept"])
    global_preds = adelie_lasso_predict(X_test, global_beta, global_intercept)
    if lambda_ is None:
        lambda_ = cv_result["best_lambda"]
    lambda_ = float(lambda_)

    L_student = float(np.mean((y_train - student_oof_preds) ** 2))
    L_teacher = float(np.mean((y_train - teacher_oof_preds) ** 2))
    mu = L_student / max(L_teacher, epsilon)

    if mu <= 1.0:
        return {
            "predictions": global_preds,
            "global_predictions": global_preds,
            "mu": mu, 
            "used_global": True,
            "sim_weights": None, 
            "n_eff": None,
            "betas": np.tile(global_beta, (m, 1)),
            "intercepts": np.full(m, global_intercept),
            "lambda_": lambda_,
            "lambdas_used": np.full(m, lambda_),
            "lambda_scale": lambda_scale,
            "alpha": float(alpha),
        }

    S = teacher_similarity(teacher_oof_preds, teacher_preds_test)
    n_eff = 1.0 / (S ** 2).sum(axis=1)
    distill_weight = mu / np.sqrt(n_eff)

    n_min = min_n_ratio * n  # floor on n_eff for the scaled-lambda case
    if lambda_scale == "n_eff":
        lambdas_i = lambda_ * np.sqrt(n / np.maximum(n_eff, n_min))
    else:
        lambdas_i = np.full(m, lambda_)

    local_preds = np.zeros(m)
    betas = np.zeros((m, p))
    intercepts = np.zeros(m)
    for i in range(m):
        X_aug = np.vstack([X_train, X_test[i:i + 1]])
        y_aug = np.concatenate([y_train, [teacher_preds_test[i]]])
        w_aug = np.concatenate([S[i], [distill_weight[i]]])
        fit_i = adelie_lasso_fit(X_aug, y_aug, lmda=lambdas_i[i],
                                  weights=w_aug, alpha=alpha)
        betas[i] = fit_i["beta"]
        intercepts[i] = float(fit_i["intercept"])
        local_preds[i] = float(X_test[i] @ fit_i["beta"] + fit_i["intercept"])

    return {
        "predictions": local_preds,
        "global_predictions": global_preds,
        "mu": mu, "used_global": False,
        "sim_weights": S, "n_eff": n_eff,
        "betas": betas, "intercepts": intercepts,
        "lambda_": lambda_, "lambdas_used": lambdas_i,
        "lambda_scale": lambda_scale,
        "alpha": float(alpha)
    }


# --------------------------------------------------------------------------- #
# Local distillation — 2x2 ablation of its two teacher-derived components
# --------------------------------------------------------------------------- #
def local_distill_ablation(X_train, y_train, X_test,
                           teacher_preds_test, teacher_oof_preds,
                           student_oof_preds,
                           lambda_=None, epsilon=1e-8, random_state=None):
    """Two novel cells of local distillation's 2x2 ablation:
        "anchor"  : uniform weights, + teacher anchor (x_i, phi(x_i))
        "weights" : teacher-similarity row weights S, no anchor

    The other two corners of the 2x2 are recovered from methods computed
    elsewhere and are NOT recomputed here: the (uniform, no-anchor) corner is
    the global lasso (`global_lasso`), and the (S, anchor) corner is full local
    distillation (`ld` / `local_distill`).

    Both cells share one global mu and the same mu <= 1 fallback to the global
    lasso, so any difference between them isolates each component's effect.
    Fixed global lambda, pure lasso (alpha=1) -- the same config as method_ld.
    Uniform-weight rows sum to 1 (matching the softmax scale of S), so the
    effective size is n_eff = n and the anchor weight is mu / sqrt(n).

    Returns dict: mu, used_global, lambda_, sim_weights, n_eff, and ``cells``
    (a dict mapping the two cell names to their per-point fits).
    """
    n, m = X_train.shape[0], X_test.shape[0]
    p = X_train.shape[1]

    cv_result = adelie_lasso_cv(X_train, y_train, n_folds=5, seed=random_state)
    global_beta = cv_result["beta"]
    global_intercept = float(cv_result["intercept"])
    if lambda_ is None:
        lambda_ = cv_result["best_lambda"]
    lambda_ = float(lambda_)

    L_student = float(np.mean((y_train - student_oof_preds) ** 2))
    L_teacher = float(np.mean((y_train - teacher_oof_preds) ** 2))
    mu = L_student / max(L_teacher, epsilon)

    if mu <= 1.0:
        # Teacher gives no edge -- both cells are the global lasso.
        g = {"predictions": adelie_lasso_predict(X_test, global_beta,
                                                 global_intercept),
             "betas": np.tile(global_beta, (m, 1)),
             "intercepts": np.full(m, global_intercept)}
        return {
            "mu": mu, "used_global": True, "lambda_": lambda_,
            "sim_weights": None, "n_eff": None,
            "cells": {"anchor": g, "weights": g},
        }

    S = teacher_similarity(teacher_oof_preds, teacher_preds_test)
    n_eff = 1.0 / (S ** 2).sum(axis=1)

    w_unif = np.full(n, 1.0 / n)        # uniform rows, sum to 1 like S
    dw_unif = mu / np.sqrt(n)           # anchor weight, uniform cell (n_eff = n)

    cells = {
        "anchor":  {"predictions": np.zeros(m), "betas": np.zeros((m, p)),
                    "intercepts": np.zeros(m)},
        "weights": {"predictions": np.zeros(m), "betas": np.zeros((m, p)),
                    "intercepts": np.zeros(m)},
    }

    for i in range(m):
        # weights: teacher-weighted rows, no anchor.
        fw = adelie_lasso_fit(X_train, y_train, lmda=lambda_,
                              weights=S[i], alpha=1.0)
        cells["weights"]["betas"][i] = fw["beta"]
        cells["weights"]["intercepts"][i] = float(fw["intercept"])
        cells["weights"]["predictions"][i] = float(
            X_test[i] @ fw["beta"] + fw["intercept"])

        # anchor: uniform rows + anchor.
        X_aug = np.vstack([X_train, X_test[i:i + 1]])
        y_aug = np.concatenate([y_train, [teacher_preds_test[i]]])
        wa = np.concatenate([w_unif, [dw_unif]])
        fa = adelie_lasso_fit(X_aug, y_aug, lmda=lambda_, weights=wa, alpha=1.0)
        cells["anchor"]["betas"][i] = fa["beta"]
        cells["anchor"]["intercepts"][i] = float(fa["intercept"])
        cells["anchor"]["predictions"][i] = float(
            X_test[i] @ fa["beta"] + fa["intercept"])

    return {
        "mu": mu, "used_global": False, "lambda_": lambda_,
        "sim_weights": S, "n_eff": n_eff,
        "cells": cells,
    }


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
OPENML_DIR = ROOT / "openml_data"
UCI_DIR    = ROOT / "uci_data"


def load_dataset(name):
    """Load a regression dataset by name.  Tries, in order:

        <ROOT>/<name>.npz                 (cached, preferred)
        OPENML_DIR/<name>.csv             (case-insensitive match)
        UCI_DIR/<name>.csv                (case-insensitive match)

    CSVs must have a column called ``target``; every other column is a
    feature.  Rows with NaN and exact duplicates are dropped."""
    npz_path = ROOT / f"{name}.npz"
    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=True)
        X = d["X"].astype(np.float64)
        y = d["y"].astype(np.float64)
        feat = list(d["feature_names"])
    else:
        csv_path = None
        for d in (OPENML_DIR, UCI_DIR):
            if not d.exists(): continue
            hits = [p for p in d.iterdir()
                    if p.suffix.lower() == ".csv"
                    and p.stem.lower() == name.lower()]
            if hits:
                csv_path = hits[0]; break
        if csv_path is None:
            raise FileNotFoundError(
                f"No cache for '{name}': tried {npz_path}, "
                f"{OPENML_DIR}/{name}.csv, {UCI_DIR}/{name}.csv")
        df = pd.read_csv(csv_path).dropna(axis=1, how="all")
        if "target" not in df.columns:
            raise ValueError(f"{csv_path} has no 'target' column")
        df = df.dropna()
        feat = [c for c in df.columns if c != "target"]
        X_df = df[feat]
        non_numeric = [c for c in feat
                       if not np.issubdtype(X_df[c].dtype, np.number)]
        if non_numeric:
            print(f"  [load_dataset] {name}: one-hot encoding "
                  f"{len(non_numeric)} non-numeric columns: {non_numeric}")
            X_df = pd.get_dummies(X_df, columns=non_numeric,
                                  drop_first=True, dtype=float)
        feat = list(X_df.columns)
        X = X_df.values.astype(np.float64)
        y = df["target"].values.astype(np.float64)
    combo = np.hstack([X, y[:, None]])
    _, uniq = np.unique(combo, axis=0, return_index=True)
    uniq.sort()
    return X[uniq], y[uniq], feat


def split_and_standardize(X, y, seed=42, test_size=0.20):
    """80/20 split + StandardScaler fit on training only.  Returns
    (X_tr_std, X_te_std, y_tr, y_te, fitted_scaler)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                               random_state=seed)
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te), y_tr, y_te, sc


# --------------------------------------------------------------------------- #
# Stability diagnostics (x-space AND phi-space 5-NN pairs)                    #
# --------------------------------------------------------------------------- #
def _nn_pairs(coords, k=5):
    """Unordered k-NN pairs on ``coords`` (2D matrix or 1D reshaped)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, ind = nn.kneighbors(coords)
    pairs = set()
    for i in range(len(coords)):
        for j in ind[i, 1:]:
            pairs.add((int(min(i, j)), int(max(i, j))))
    return sorted(pairs)


def _stab_pairs(betas, pairs, coords, tol=1e-6):
    """For each pair (i, j), compute β-Lipschitz ratio and Jaccard.
    ``coords`` is 1D (phi) or 2D (X); distance is |Δ| or ||Δ||₂ accordingly."""
    beta_lips, jaccards = [], []
    for i, j in pairs:
        if coords.ndim == 1:
            d = abs(float(coords[i] - coords[j]))
        else:
            d = float(np.linalg.norm(coords[i] - coords[j]))
        if d < 1e-12:
            continue
        beta_lips.append(float(np.linalg.norm(betas[i] - betas[j]) / d))
        Si = set(np.flatnonzero(np.abs(betas[i]) > tol))
        Sj = set(np.flatnonzero(np.abs(betas[j]) > tol))
        union = Si | Sj
        inter = Si & Sj
        jaccards.append(len(inter) / len(union) if union else 1.0)
    return (float(np.median(beta_lips)) if beta_lips else NAN,
            float(np.mean(jaccards)) if jaccards else NAN)


def _local_lipschitz_per_point(betas, coords, k=5, tol=1e-6):
    """Per-point local Lipschitz à la Alvarez-Melis & Jaakkola (2018), Eq. 2.

    For each test point i, take its k nearest neighbors in ``coords`` and
    return max_j ||β_i − β_j|| / ||x_i − x_j||.  Aggregation across points
    is left to the caller (we report the mean and the median).

    ``coords`` may be 2D (x-space) or 1D (φ-space).
    """
    if betas is None or len(coords) <= k:
        return NAN, NAN, NAN
    coords2d = coords.reshape(-1, 1) if coords.ndim == 1 else coords
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords2d)
    _, ind = nn.kneighbors(coords2d)
    per_point = []
    jaccs_per_point = []
    for i in range(len(coords2d)):
        neigh = ind[i, 1:]
        ratios, jaccs = [], []
        Si = set(np.flatnonzero(np.abs(betas[i]) > tol))
        for j in neigh:
            d = float(np.linalg.norm(coords2d[i] - coords2d[j]))
            if d < 1e-12:
                continue
            ratios.append(float(np.linalg.norm(betas[i] - betas[j]) / d))
            Sj = set(np.flatnonzero(np.abs(betas[j]) > tol))
            union = Si | Sj
            inter = Si & Sj
            jaccs.append(len(inter) / len(union) if union else 1.0)
        if ratios:
            per_point.append(max(ratios))
        if jaccs:
            jaccs_per_point.append(float(np.mean(jaccs)))
    if not per_point:
        return NAN, NAN, NAN
    return (float(np.mean(per_point)),
            float(np.median(per_point)),
            float(np.mean(jaccs_per_point)) if jaccs_per_point else NAN)


def stability_diagnostics_v2(betas, X_test, phi=None, k=5, tol=1e-6):
    """AM&J-style stability: per-point max-over-k-NN, then aggregate.

    Returns a dict with both x-space and φ-space variants:
        beta_lipschitz_xspace_mean    mean of per-point local Lipschitz (x-space)
        beta_lipschitz_xspace_median  median of per-point local Lipschitz (x-space)
        jaccard_xspace_mean_v2        mean of per-point mean active-set Jaccard
        beta_lipschitz_phi_mean       same in φ-space
        beta_lipschitz_phi_median     ...
        jaccard_phi_mean_v2           ...
        mean_active_set               # nonzero coefs per point
        smoothness_k                  the k value used (for sensitivity sweeps)

    Reference: Alvarez-Melis & Jaakkola (2018), "On the Robustness of
    Interpretability Methods", arXiv:1806.08049, Eq. 2 (discrete variant);
    we substitute a k-NN neighborhood for their ε-ball to ensure consistent
    estimator support across heterogeneous datasets.
    """
    if betas is None:
        return {"beta_lipschitz_xspace_mean":   NAN,
                "beta_lipschitz_xspace_median": NAN,
                "jaccard_xspace_mean_v2":       NAN,
                "beta_lipschitz_phi_mean":      NAN,
                "beta_lipschitz_phi_median":    NAN,
                "jaccard_phi_mean_v2":          NAN,
                "mean_active_set":              NAN,
                "smoothness_k":                 k}
    density = float(np.mean((np.abs(betas) > tol).sum(axis=1)))
    blx_mean, blx_med, jcx = _local_lipschitz_per_point(betas, X_test, k=k, tol=tol)
    if phi is not None:
        phi_1d = np.asarray(phi).ravel()
        blp_mean, blp_med, jcp = _local_lipschitz_per_point(
            betas, phi_1d, k=k, tol=tol)
    else:
        blp_mean = blp_med = jcp = NAN
    return {
        "beta_lipschitz_xspace_mean":   blx_mean,
        "beta_lipschitz_xspace_median": blx_med,
        "jaccard_xspace_mean_v2":       jcx,
        "beta_lipschitz_phi_mean":      blp_mean,
        "beta_lipschitz_phi_median":    blp_med,
        "jaccard_phi_mean_v2":          jcp,
        "mean_active_set":              density,
        "smoothness_k":                 k,
    }


def stability_diagnostics(betas, X_test, phi=None, tol=1e-6):
    """Summarise the smoothness of the per-test-point coefficient field.

    Returns a dict with:
        beta_lipschitz_xspace_median  β-Lipschitz median over 5-NN pairs in X-space
        jaccard_xspace_mean           active-set Jaccard mean over same pairs
        beta_lipschitz_phi_median     β-Lipschitz median over 5-NN pairs in φ-space
        jaccard_phi_mean              active-set Jaccard mean over same pairs
        mean_active_set               mean # of nonzero coefficients per test point

    Passing ``betas=None`` (e.g. for the teacher, which has no coefficients)
    returns NaN for everything.
    """
    if betas is None:
        return {"beta_lipschitz_xspace_median": NAN,
                "jaccard_xspace_mean":          NAN,
                "beta_lipschitz_phi_median":    NAN,
                "jaccard_phi_mean":             NAN,
                "mean_active_set":              NAN}
    density = float(np.mean((np.abs(betas) > tol).sum(axis=1)))
    bl_x, jc_x = _stab_pairs(betas, _nn_pairs(X_test, k=5), X_test, tol=tol)
    if phi is not None:
        phi_1d = np.asarray(phi).ravel()
        bl_phi, jc_phi = _stab_pairs(betas,
                                      _nn_pairs(phi_1d.reshape(-1, 1), k=5),
                                      phi_1d, tol=tol)
    else:
        bl_phi, jc_phi = NAN, NAN
    return {
        "beta_lipschitz_xspace_median": bl_x,
        "jaccard_xspace_mean":          jc_x,
        "beta_lipschitz_phi_median":    bl_phi,
        "jaccard_phi_mean":             jc_phi,
        "mean_active_set":              density,
    }
