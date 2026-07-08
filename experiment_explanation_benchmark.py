"""Benchmark local distillation vs LIME, LOESS, MAPLE on OpenML regression datasets.

Incremental-logging, resumable: each (dataset, method, seed) is written to
results.jsonl as soon as it finishes.  If the run is killed and restarted,
already-completed tasks are skipped.

Methods
-------
global_lasso : adelie_lasso_cv on standardized X; predicts y.
teacher      : TabPFN (GBM fallback if TabPFN throws); predicts y.
xgboost      : XGBRegressor with fixed hyperparameters (see common.XGB_KWARGS);
               predicts y.  Second global baseline / alternative teacher.
lime         : `lime` package's LimeTabularExplainer, explains the teacher.
               Surrogate linear coefficients are read off at each test
               point; prediction is the surrogate's value at x*.
loess        : weighted OLS per test point, Gaussian kernel in standardized
               x-space, bandwidth = 0.5 * median pairwise training distance.
maple        : random-forest leaf co-membership weights (Plumb et al.),
               then weighted LASSO at the global CV lambda per test point.
ld    : our method, alpha=1.0, lambda_scale="fixed", no prior.  TabPFN teacher.
ld_xgb       : same as ld, but XGBoost is the teacher (OOF + test predictions).

Metrics per (dataset, method, seed)
----------------------------------
test_mse, test_r2                 : on y_true
mean_active_set                   : mean |{k : |beta_{i,k}| > 1e-6}| across i
                                     (for loess: report p)
beta_lipschitz_median             : ||beta_i - beta_j|| / ||x_i - x_j||
                                     across 5-NN pairs in x-space
active_set_jaccard_mean           : across 5-NN pairs in x-space
time_per_test_point_ms            : total method time / n_test in ms

Datasets are loaded from the local .npz cache I already created; to add
more datasets, convert their OpenML parquet to .npz (see the other scripts).

Seed 42 for method-internal randomness in LIME/MAPLE's RF; the outer
`seed` argument varies the train/test split and CV folds.
"""
from __future__ import annotations

import json, os, sys, time, traceback, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED_INTERNAL = 42

# --- helpers come from benchmark_methods/common.py ------------------------
sys.path.insert(0, str(Path(__file__).parent / "benchmark_methods"))
from common import (adelie_lasso_cv, adelie_lasso_fit, adelie_lasso_predict,  # noqa: E402
                    tabpfn_predict, tabpfn_cv,
                    tabfm_predict, tabfm_cv,
                    xgboost_predict, xgboost_cv, local_distill,
                    load_dataset, split_and_standardize,
                    stability_diagnostics, NAN)
import method_global_lasso, method_global_ridge, method_lime, method_loess, method_maple, method_ld, method_xgboost  # noqa: E402
import method_ld_ablation  # noqa: E402
import method_ld_ridge  # noqa: E402  ridge student (alpha=0) for each teacher

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results" / "local_explanation_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSONL = RESULTS_DIR / "results.jsonl"
BETAS_DIR = RESULTS_DIR / "betas"
BETAS_DIR.mkdir(exist_ok=True)
AGG_DIR = RESULTS_DIR / "aggregates"
AGG_DIR.mkdir(exist_ok=True)
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- datasets (adjust this list to extend the benchmark) -------------------
# The 17-dataset panel matches the paper (7 UCI + 10 OpenML-CTR23).
# Datasets resolve via
# benchmark_methods.common.load_dataset which prefers <ROOT>/<name>.npz
# if present and otherwise loads the CSV from openml_data/ or uci_data/.
# Ordered small → large so feedback comes early; the three biggest
# (space_ga, white_wine, abalone) sit at the end of each pass.
DATASETS = [
    "Automobile",                          # n=159
    "Servo",                               # n=167
    "Liver_Disorders",                     # n=341
    "Auto_MPG",                            # n=392
    "Real_Estate_Valuation",               # n=414
    "student_performance_por",             # n=649
    "cars",                                # n=804
    "QSAR_fish_toxicity",                  # n=907
    "concrete_compressive_strength",       # n=1005
    "Infrared_Thermography_Temperature",   # n=1018
    "socmob",                              # n=1156
    "red_wine",                            # n=1359
    "airfoil_self_noise",                  # n=1503
    "auction_verification",                # n=2043
    "space_ga",                            # n=3107
    "white_wine",                          # n=3961
    "abalone",                             # n=4177
]

# Seeds to run.  Overridable via env var BENCHMARK_SEEDS="0-19" or
# "0,1,2,5".  Default is 20 seeds.
def _parse_seeds(spec):
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-"); out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return out
SEEDS = _parse_seeds(os.environ.get("BENCHMARK_SEEDS", "0-19"))

METHODS = ["global_lasso", "global_ridge", "teacher", "xgboost", "tabfm", "lime", "loess", "maple",
           "ld", "ld_xgb", "ld_tabfm",
           # Ridge students (alpha=0) sharing each teacher's OOF/test preds.
           "ld_ridge", "ld_xgb_ridge", "ld_tabfm_ridge",
           # Two novel ablation cells (computed in one shared pass).  The other
           # two corners of the 2x2 are global_lasso (== ld_global) and ld
           # (== ld_full), so we don't recompute them.
           "ld_anchor", "ld_weights"]

# Ablation cells produced together by method_ld_ablation.run (TabPFN teacher).
ABL_METHODS = ["ld_anchor", "ld_weights"]


# --- incremental logging ---------------------------------------------------
def done_keys():
    done = set()
    if RESULTS_JSONL.exists():
        with open(RESULTS_JSONL) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done.add((r["dataset"], r["method"], int(r["seed"])))
                except Exception:
                    pass
    return done


def append_result(r):
    line = json.dumps(r) + "\n"
    with open(RESULTS_JSONL, "a") as f:
        f.write(line); f.flush(); os.fsync(f.fileno())


LIME_NUM_SAMPLES = 1000  # LIME paper default is 5000; reduced to 1000 to
                          # keep total wall time < 1 day.


# --- per-dataset driver ----------------------------------------------------
def run_dataset_seed(name, seed, already_done):
    X, y, feat = load_dataset(name)
    X_tr, X_te, y_tr, y_te, _ = split_and_standardize(X, y, seed=seed)
    n, p = X_tr.shape
    m = len(y_te)
    print(f"\n[{name} seed={seed}]  X_tr={X_tr.shape}  X_te={X_te.shape}")

    # Check what's pending.  If only teacher is pending, we still fit once
    # upfront so that phi is available for any retries; it's the cheapest
    # operation compared to the rest.
    pending = [m_ for m_ in METHODS if (name, m_, seed) not in already_done]
    if not pending:
        print("  all methods done for this (dataset, seed)")
        return

    # Fit the TabPFN teacher upfront ONLY when a pending method needs its test
    # predictions: teacher / ld / ld_ridge / the ablation cells.  Teacher-free
    # methods (global_lasso, global_ridge, loess, maple, lime — which fits its
    # own TabPFN internally) and the xgb-/tabfm-based methods don't use it, so a
    # baseline-only resume skips the fit entirely.  phi (== these teacher preds)
    # only feeds the phi-space stability columns; it is None when not fit, which
    # just NaNs those columns — fine, since the phi-space analysis was dropped.
    NEEDS_TABPFN = ("teacher", "ld", "ld_ridge", *ABL_METHODS)
    teacher_yhat_test = None
    teacher_label = None
    if any(m_ in NEEDS_TABPFN for m_ in pending):
        t0 = time.time()
        teacher_label = "tabpfn"
        try:
            teacher_yhat_test = tabpfn_predict(X_tr, y_tr, X_te)
        except Exception as e:
            print(f"  TabPFN failed ({e!s:.80}); falling back to GBM")
            from sklearn.ensemble import GradientBoostingRegressor
            gbm = GradientBoostingRegressor(n_estimators=500, max_depth=5,
                                             random_state=seed).fit(X_tr, y_tr)
            teacher_yhat_test = gbm.predict(X_te); teacher_label = "fallback_gbm"
        print(f"  teacher fit upfront in {time.time()-t0:.0f}s ({teacher_label})")
    phi = teacher_yhat_test

    # teacher OOF, fit lazily if anything that needs it is pending.
    # The ld ablation cells share the TabPFN teacher OOF with ld.
    teacher_oof = None
    if any(m_ in ("teacher", "ld", "ld_ridge", *ABL_METHODS) for m_ in pending):
        t0 = time.time()
        try:
            teacher_oof = tabpfn_cv(X_tr, y_tr, n_folds=5,
                                     random_state=seed)["predictions"]
        except Exception:
            # 5-fold GBM OOF as fallback.
            teacher_oof = np.zeros(n)
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            for tr_idx, te_idx in kf.split(X_tr):
                from sklearn.ensemble import GradientBoostingRegressor
                g = GradientBoostingRegressor(n_estimators=500, max_depth=5,
                                               random_state=seed).fit(
                    X_tr[tr_idx], y_tr[tr_idx])
                teacher_oof[te_idx] = g.predict(X_tr[te_idx])
        print(f"  teacher OOF in {time.time()-t0:.0f}s")

    # XGBoost teacher (for standalone 'xgboost' method and 'ld_xgb') —
    # fit lazily if anything that needs it is pending.
    xgb_yhat_test = None
    xgb_oof = None
    if any(m_ in ("xgboost", "ld_xgb", "ld_xgb_ridge") for m_ in pending):
        t0 = time.time()
        xgb_yhat_test = xgboost_predict(X_tr, y_tr, X_te, random_state=seed)
        print(f"  xgboost teacher fit in {time.time()-t0:.0f}s")
        if any(m_ in ("ld_xgb", "ld_xgb_ridge") for m_ in pending):
            t0 = time.time()
            xgb_oof = xgboost_cv(X_tr, y_tr, n_folds=5,
                                 random_state=seed)["predictions"]
            print(f"  xgboost OOF in {time.time()-t0:.0f}s")

    # TabFM teacher (for standalone 'tabfm' and 'ld_tabfm') — live refit per
    # seed, no caching, mirroring the TabPFN/XGBoost path.  Fit lazily.
    tabfm_yhat_test = None
    tabfm_oof = None
    if any(m_ in ("tabfm", "ld_tabfm", "ld_tabfm_ridge") for m_ in pending):
        t0 = time.time()
        tabfm_yhat_test = tabfm_predict(X_tr, y_tr, X_te)
        print(f"  tabfm teacher fit in {time.time()-t0:.0f}s")
        if any(m_ in ("ld_tabfm", "ld_tabfm_ridge") for m_ in pending):
            t0 = time.time()
            tabfm_oof = tabfm_cv(X_tr, y_tr, n_folds=5,
                                 random_state=seed)["predictions"]
            print(f"  tabfm OOF in {time.time()-t0:.0f}s")

    # ld ablation: all four cells in one shared pass (TabPFN teacher).
    # Computed upfront if any cell is pending; the dispatch loop reads cells
    # from this cache.  Recomputed only when at least one cell is pending.
    abl_results = None
    if any(m_ in ABL_METHODS for m_ in pending):
        t0 = time.time()
        abl_results = method_ld_ablation.run(
            X_tr, y_tr, X_te, y_te, seed, phi=phi,
            teacher_oof=teacher_oof, teacher_test=teacher_yhat_test)
        print(f"  ld ablation (4 cells) in {time.time()-t0:.0f}s")

    for method in METHODS:
        key = (name, method, seed)
        if key in already_done:
            print(f"  skip {method}: already done")
            continue
        t_start = time.time()
        try:
            if method == "global_lasso":
                r = method_global_lasso.run(X_tr, y_tr, X_te, y_te, seed, phi=phi)
            elif method == "global_ridge":
                r = method_global_ridge.run(X_tr, y_tr, X_te, y_te, seed, phi=phi)
            elif method == "teacher":
                # Use the teacher prediction already fit upfront; don't refit.
                r = {"yhat": teacher_yhat_test, "betas": None,
                     "time_ms_per_test_point": 0.0,
                     "beta_lipschitz_xspace_median": NAN,
                     "jaccard_xspace_mean":          NAN,
                     "beta_lipschitz_phi_median":    NAN,
                     "jaccard_phi_mean":             NAN,
                     "mean_active_set":              NAN,
                     "teacher_backend": teacher_label}
            elif method == "lime":
                r = method_lime.run(X_tr, y_tr, X_te, y_te, seed,
                                     LIME_NUM_SAMPLES=LIME_NUM_SAMPLES, phi=phi)
            elif method == "loess":
                r = method_loess.run(X_tr, y_tr, X_te, y_te, seed, phi=phi)
            elif method == "maple":
                r = method_maple.run(X_tr, y_tr, X_te, y_te, seed, phi=phi)
            elif method == "ld":
                r = method_ld.run(X_tr, y_tr, X_te, y_te, seed, phi=phi,
                                   teacher_oof=teacher_oof,
                                   teacher_test=teacher_yhat_test)
            elif method == "ld_ridge":
                r = method_ld_ridge.run(X_tr, y_tr, X_te, y_te, seed, phi=phi,
                                        teacher_oof=teacher_oof,
                                        teacher_test=teacher_yhat_test)
            elif method == "xgboost":
                r = method_xgboost.run(X_tr, y_tr, X_te, y_te, seed, phi=phi)
            elif method == "ld_xgb":
                r = method_ld.run(X_tr, y_tr, X_te, y_te, seed, phi=phi,
                                   teacher_oof=xgb_oof,
                                   teacher_test=xgb_yhat_test)
            elif method == "ld_xgb_ridge":
                r = method_ld_ridge.run(X_tr, y_tr, X_te, y_te, seed, phi=phi,
                                        teacher_oof=xgb_oof,
                                        teacher_test=xgb_yhat_test)
            elif method == "tabfm":
                # Reuse the teacher prediction fit upfront; don't refit.
                r = {"yhat": tabfm_yhat_test, "betas": None,
                     "time_ms_per_test_point": 0.0,
                     "beta_lipschitz_xspace_median": NAN,
                     "jaccard_xspace_mean":          NAN,
                     "beta_lipschitz_phi_median":    NAN,
                     "jaccard_phi_mean":             NAN,
                     "mean_active_set":              NAN}
            elif method == "ld_tabfm":
                r = method_ld.run(X_tr, y_tr, X_te, y_te, seed, phi=phi,
                                   teacher_oof=tabfm_oof,
                                   teacher_test=tabfm_yhat_test)
            elif method == "ld_tabfm_ridge":
                r = method_ld_ridge.run(X_tr, y_tr, X_te, y_te, seed, phi=phi,
                                        teacher_oof=tabfm_oof,
                                        teacher_test=tabfm_yhat_test)
            elif method in ABL_METHODS:
                # All four cells were computed together upfront; read this one.
                r = abl_results[method]
            else:
                raise ValueError(method)

            yhat = r["yhat"]
            row = {
                "dataset": name, "method": method, "seed": int(seed),
                "status": "ok",
                "n_train": int(n), "n_test": int(m), "p": int(p),
                "test_mse": float(mean_squared_error(y_te, yhat)),
                "test_r2":  float(r2_score(y_te, yhat)),
                "mean_active_set": r.get("mean_active_set", NAN),
                "beta_lipschitz_xspace_median": r.get("beta_lipschitz_xspace_median", NAN),
                "jaccard_xspace_mean":          r.get("jaccard_xspace_mean", NAN),
                "beta_lipschitz_phi_median":    r.get("beta_lipschitz_phi_median", NAN),
                "jaccard_phi_mean":             r.get("jaccard_phi_mean", NAN),
                "time_per_test_point_ms": r.get("time_ms_per_test_point",
                                                  r.get("time_ms", NAN)),
                "method_wall_s": float(time.time() - t_start),
            }
            if method in ("ld", "ld_xgb", "ld_tabfm",
                          "ld_ridge", "ld_xgb_ridge", "ld_tabfm_ridge",
                          *ABL_METHODS) and "used_global" in r:
                row["ld_used_global"] = r["used_global"]
            # Persist muhat per distillation row so the aggregation can pick the
            # CV-selected teacher post-hoc (argmax muhat) with no refit.
            if "mu" in r:
                row["mu_hat"] = float(r["mu"])
            if method == "teacher" and "teacher_backend" in r:
                row["teacher_backend"] = r["teacher_backend"]
            # Persist full (m, p) beta matrix when available, for post-hoc
            # analyses (feature variance, deviation-from-global, pointwise
            # β-Lipschitz, etc.) without re-running the benchmark.  Teacher
            # returns betas=None and is skipped.
            if r.get("betas") is not None:
                np.savez_compressed(
                    BETAS_DIR / f"{name}__{method}__{seed}.npz",
                    betas=np.asarray(r["betas"]),
                    yhat=np.asarray(yhat),
                )
            append_result(row)
            print(f"  OK {method}: MSE={row['test_mse']:.3f}  R²={row['test_r2']:.3f}  "
                  f"time={row['method_wall_s']:.0f}s")
        except Exception as e:
            tb = traceback.format_exc()
            err_row = {
                "dataset": name, "method": method, "seed": int(seed),
                "status": "error",
                "error_msg": str(e),
                "error_tb": tb[-500:],
                "method_wall_s": float(time.time() - t_start),
            }
            append_result(err_row)
            print(f"  ERR {method}: {e!s:.100}")
            print(tb[-500:])


# --- aggregates ------------------------------------------------------------
def write_aggregates():
    if not RESULTS_JSONL.exists():
        return
    rows = []
    with open(RESULTS_JSONL) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception: pass
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df[df["status"] == "ok"].copy()
    if len(df) == 0:
        return

    metric_cols = ["test_mse", "test_r2", "mean_active_set",
                   "beta_lipschitz_xspace_median", "jaccard_xspace_mean",
                   "beta_lipschitz_phi_median",    "jaccard_phi_mean",
                   "time_per_test_point_ms"]
    # Tolerate missing columns from older rows (fill with NaN for aggregation).
    for c in metric_cols:
        if c not in df.columns:
            df[c] = float("nan")

    # --- CV-selected-teacher: synthetic post-hoc method (no refit) ----------
    # For each (dataset, seed), pick the distillation teacher with the largest
    # muhat.  Because the student CV loss in muhat's numerator is the same lasso
    # across teachers, argmax-muhat == argmin-teacher-CV-error, and selection
    # uses only training-fold CV (muhat) — the test set is never consulted.
    # Emit that teacher's row as method "ld_cv_selected".  If every teacher
    # reverted (muhat <= 1) the ld_* row already equals the global lasso; we
    # substitute the global_lasso row when present to make the fallback explicit.
    TEACHER_DISTILL = ["ld", "ld_xgb", "ld_tabfm"]
    if "mu_hat" in df.columns:
        syn = []
        cand = df[df["method"].isin(TEACHER_DISTILL)].dropna(subset=["mu_hat"])
        for (ds, sd), g in cand.groupby(["dataset", "seed"]):
            best = g.loc[g["mu_hat"].idxmax()]
            sel_teacher = str(best["method"])
            all_reverted = bool(best["mu_hat"] <= 1.0)
            row = best.copy()
            if all_reverted:
                gl = df[(df["dataset"] == ds) & (df["seed"] == sd)
                        & (df["method"] == "global_lasso")]
                if len(gl):
                    row = gl.iloc[0].copy()
            row["method"] = "ld_cv_selected"
            row["cv_selected_teacher"] = sel_teacher
            row["cv_all_reverted"] = all_reverted
            syn.append(row)
        if syn:
            df = pd.concat([df, pd.DataFrame(syn)], ignore_index=True)

    # Per (dataset, method): mean and std across seeds.
    per_ds = df.groupby(["dataset", "method"])[metric_cols].agg(["mean", "std"])
    per_ds.columns = ["_".join(c).strip("_") for c in per_ds.columns]
    per_ds.to_csv(AGG_DIR / "per_dataset_table.csv")

    # Rollup: per method across all (dataset, seed).
    rollup = df.groupby("method")[metric_cols].agg(["mean", "median", "std"])
    rollup.columns = ["_".join(c).strip("_") for c in rollup.columns]
    rollup.to_csv(AGG_DIR / "rollup_table.csv")


# --- main ------------------------------------------------------------------
def main():
    # If running as a SLURM array task, restrict to the single dataset
    # matching SLURM_ARRAY_TASK_ID.  Mid-run aggregates are skipped in
    # array mode to avoid concurrent writers racing on the CSVs;
    # aggregates are computed once at the end of each task and the last
    # task to finish leaves the correct aggregates on disk.
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    datasets = DATASETS
    if array_id is not None:
        idx = int(array_id)
        if not (0 <= idx < len(DATASETS)):
            raise IndexError(f"SLURM_ARRAY_TASK_ID={idx} out of range "
                             f"0..{len(DATASETS)-1}")
        datasets = [DATASETS[idx]]
        print(f"[array task {idx}] running dataset: {datasets[0]}")

    done = done_keys()
    total_planned = len(datasets) * len(SEEDS) * len(METHODS)
    print(f"Planned: {total_planned} tasks  |  already done: {len(done)}  "
          f"|  seeds: {SEEDS}")
    counter = 0
    for name in datasets:
        # load_dataset handles both .npz and CSV fallbacks in openml_data/
        # and uci_data/; don't pre-screen on .npz existence.
        for seed in SEEDS:
            run_dataset_seed(name, seed, done)
            counter += 1
            if array_id is None and counter % 2 == 0:
                print(f"  rollup: computing aggregates ...")
                write_aggregates()
        if array_id is None:
            write_aggregates()
        print(f"== done dataset {name} ==")
    write_aggregates()
    print("\nAll requested tasks finished.")


if __name__ == "__main__":
    main()
