"""Export standardized train/test splits to disk for the R / LLF comparator.

For each (dataset, seed) we write the EXACT arrays the Python benchmark feeds
its teacher-unaware methods (LOESS etc.): the split produced by
common.split_and_standardize(seed) — an 80/20 train_test_split(random_state=seed)
followed by a StandardScaler fit on the training fold only.  The R script
(llf_grf.R) reads these arrays directly, so LLF runs on byte-identical splits:
R never reseeds, and the paired comparison across methods stays valid.

Layout (under results/llf/):
    datasets.txt                          one dataset name per line, index order
                                          (R maps SLURM_ARRAY_TASK_ID -> name)
    splits/{dataset}__{seed}__train.csv   columns x0..x{p-1}, y  (X standardized)
    splits/{dataset}__{seed}__test.csv    same columns
    splits/{dataset}__{seed}__idx.json    raw train/test row indices (provenance)

Resumable: skips a split whose train+test CSVs already exist.  One dataset per
SLURM_ARRAY_TASK_ID; seeds 0-19 inside one task.  Also fine to run directly on
the login node (no GPU): `python export_splits.py`.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "benchmark_methods"))
from common import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "llf"
SPLITS = OUT / "splits"
SPLITS.mkdir(parents=True, exist_ok=True)

# MUST match experiment_explanation_benchmark.DATASETS exactly (order + names).
DATASETS = [
    "Automobile", "Servo", "Liver_Disorders", "Auto_MPG",
    "Real_Estate_Valuation", "student_performance_por", "cars",
    "QSAR_fish_toxicity", "concrete_compressive_strength",
    "Infrared_Thermography_Temperature", "socmob", "red_wine",
    "airfoil_self_noise", "auction_verification", "space_ga",
    "white_wine", "abalone",
]

TEST_SIZE = 0.20   # keep in lockstep with common.split_and_standardize


def _parse_seeds(spec):
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-"); out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return out
SEEDS = _parse_seeds(os.environ.get("LLF_SEEDS", "0-19"))


def _write_csv(path, X, y):
    header = ",".join([f"x{j}" for j in range(X.shape[1])] + ["y"])
    np.savetxt(path, np.column_stack([X, y]), delimiter=",",
               header=header, comments="")


def export_one(name, seed):
    tr_csv = SPLITS / f"{name}__{seed}__train.csv"
    te_csv = SPLITS / f"{name}__{seed}__test.csv"
    if tr_csv.exists() and te_csv.exists():
        print(f"  skip {name} seed={seed}: exists"); return
    X, y, _ = load_dataset(name)
    idx = np.arange(len(y))
    # Reproduce split_and_standardize EXACTLY: same train_test_split call (the
    # extra `idx` array rides the same random_state permutation, so the X/y
    # partition is identical), then StandardScaler fit on the training fold.
    X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(
        X, y, idx, test_size=TEST_SIZE, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    _write_csv(tr_csv, X_tr, y_tr)
    _write_csv(te_csv, X_te, y_te)
    with open(SPLITS / f"{name}__{seed}__idx.json", "w") as f:
        json.dump({"train_idx": tr_idx.tolist(), "test_idx": te_idx.tolist()}, f)
    print(f"  wrote {name} seed={seed}  train={X_tr.shape} test={X_te.shape}")


def main():
    (OUT / "datasets.txt").write_text("\n".join(DATASETS) + "\n")
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if task_id >= len(DATASETS):
        print(f"task_id {task_id} out of range"); return
    names = [DATASETS[task_id]] if task_id >= 0 else DATASETS
    for name in names:
        for seed in SEEDS:
            export_one(name, seed)


if __name__ == "__main__":
    main()
