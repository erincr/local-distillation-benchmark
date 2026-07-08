#!/bin/bash
#SBATCH --job-name=export_splits
#SBATCH --partition=standard          # CPU partition (no GPU needed)
#SBATCH --array=0-16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_splits_%A_%a.out
#SBATCH --error=slurm_splits_%A_%a.err
#SBATCH --account=<your_slurm_account>               # confirm CPU-eligible via `my_accounts`

# Writes results/llf/splits/{dataset}__{seed}__{train,test}.csv for the R/LLF
# step, using the identical split as the Python benchmark.  Cheap enough to run
# on the login node instead:  source the venv, then `python export_splits.py`.
set -euo pipefail
module load python/3.11 || true
source "$HOME/ldistill-venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"
echo "Array task $SLURM_ARRAY_TASK_ID on $(hostname)"
python -u export_splits.py
