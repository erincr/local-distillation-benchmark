#!/bin/bash
#SBATCH --job-name=llf_grf
#SBATCH --partition=standard          # CPU-only; LLF/grf does not use the GPU
#SBATCH --array=0-16
#SBATCH --cpus-per-task=4             # grf is multithreaded (num.threads)
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_llf_%A_%a.out
#SBATCH --error=slurm_llf_%A_%a.err
#SBATCH --account=<your_slurm_account>               # confirm CPU-eligible via `my_accounts`

# Local Linear Forests comparator (grf::ll_regression_forest).  Reads the splits
# written by export_splits.py (run that first) and writes per-(dataset,seed)
# metrics to results/llf/llf_results.jsonl.  No Python venv needed here.
#
# Confirm the R module name with `module avail R` (e.g. R/4.4.0) and that grf is
# installed:  Rscript -e 'library(grf)'   (install with the commands in the
# writeup if it is missing).
set -euo pipefail
module load R/4.4.3        # must match the R version grf was installed under

cd "$SLURM_SUBMIT_DIR"
export LLF_SEEDS="0-19"
echo "Array task $SLURM_ARRAY_TASK_ID on $(hostname)"
Rscript llf_grf.R
