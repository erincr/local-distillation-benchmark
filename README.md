# Local Distillation Benchmark

Local distillation fits an interpretable, per-test-point regularized linear model that is locally faithful to a strong black-box **teacher** (e.g. TabPFN). This benchmark compares it against global baselines, other local explainers and local models across 17 regression datasets.

## Methods

Each `(dataset, method, seed)` is evaluated and logged to
`results/local_explanation_benchmark/results.jsonl`.

| Group | Methods |
|-------|---------|
| Global baselines | `global_lasso`, `global_ridge` |
| Teachers (black box) | `teacher` (TabPFN), `xgboost`, `tabfm` |
| **Local distillation (ours)** | `ld` (TabPFN), `ld_xgb` (XGBoost), `ld_tabfm` (TabFM) — lasso student; `ld_ridge` / `ld_xgb_ridge` / `ld_tabfm_ridge` — ridge student |
| Distillation ablation | `ld_anchor`, `ld_weights` (the two teacher-derived components; the other two 2×2 corners are `global_lasso` and `ld`) |
| CV-selected teacher | `ld_cv_selected` — synthesized in aggregation: per split, the distillation whose teacher has the largest μ̂ (training-fold CV loss ratio); never looks at test |
| Teacher-free local | `loess`, `llf` (local linear forests, via R `grf`) |
| Other local explainers | `lime`, `maple` (post-hoc explainers of the teacher; reported for reference, excluded from the main comparison) |

## Datasets

17 regression datasets (7 UCI + 10 OpenML-CTR23), n ∈ [≈167, ≈4000] after de-duplication, p ∈ [4, 39]. The data is **not distributed with this repo** — see [`DATA_SOURCES.md`](DATA_SOURCES.md) for provenance and run `python prepare_data.py` to fetch and cache it locally.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# TabFM teacher (source-only, not on PyPI):
git clone https://github.com/google-research/tabfm.git && cd tabfm
pip install -e .[pytorch] && cd ..     # record `git rev-parse HEAD` for reproducibility

# LLF comparator needs R with the grf package:
Rscript -e 'install.packages("grf", repos="https://cloud.r-project.org")'

# Fetch the datasets (see DATA_SOURCES.md):
python prepare_data.py
```

Python ≥ 3.11. A CUDA GPU is strongly recommended (TabPFN and TabFM are faster on GPU). Devices are set via `TABPFN_DEVICE` / `TABFM_DEVICE` (`cuda` | `mps` | `cpu`).

## Running

**Single machine** — runs all datasets × seeds × methods, resumable (each result is appended immediately; a re-run skips completed work):

```bash
python experiment_explanation_benchmark.py
# subset: BENCHMARK_SEEDS="0-4" python experiment_explanation_benchmark.py
```

**SLURM cluster** — one dataset per array task. Set `--account` in the `.slurm` files first (they contain a `<your_slurm_account>` placeholder):

```bash
sbatch run_full_array.slurm       # main benchmark, array 0-16
```

**LLF comparator** (teacher-free, runs in R on identical splits):

```bash
python export_splits.py           # writes the exact standardized splits to disk
sbatch run_llf.slurm              # or: Rscript llf_grf.R   (grf local linear forests)
python merge_llf.py               # folds LLF results into results.jsonl
```

## Outputs & analysis

- `results/local_explanation_benchmark/results.jsonl` — one JSON row per
  `(dataset, method, seed)` with `test_mse`, `test_r2`, `mu_hat`, timings, etc.
- `aggregates/{per_dataset_table,rollup_table}.csv` — per-method summaries,
  including `ld_cv_selected`.
- `visualize_results.R` — reads `results.jsonl` (tolerant of malformed lines)
  and produces the paper figures into `figures/`.

## Layout

```
experiment_explanation_benchmark.py   main driver 
export_splits.py / llf_grf.R / merge_llf.py   LLF pipeline (Python splits -> R -> merge) -- done separately because GRF is not in Python
benchmark_methods/                    one module per method + shared common.py
run_*.slurm                           SLURM array scripts (set --account first)
prepare_data.py / DATA_SOURCES.md     dataset fetching + provenance
visualize_results.R / visualize_ablation.R   figures
```
