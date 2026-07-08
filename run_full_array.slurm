#!/bin/bash
#SBATCH --job-name=ldistill_array
#SBATCH --partition=spgpu
#SBATCH --array=0-16
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_array_%A_%a.out
#SBATCH --error=slurm_array_%A_%a.err
#SBATCH --account=<your_slurm_account>

set -euo pipefail
module load python/3.11 cuda/12.1 || true
source "$HOME/ldistill-venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export TABPFN_DEVICE=cuda
export TABFM_DEVICE=cuda
export TABPFN_TOKEN=""
export TABPFN_DISABLE_TELEMETRY=1

# TabFM (third teacher) loads weights from HF Hub on first load(); pre-download
# once (see setup) so array tasks stay offline and don't each hit the network.
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1

echo "Array task $SLURM_ARRAY_TASK_ID on $(hostname)"
python -u experiment_explanation_benchmark.py
