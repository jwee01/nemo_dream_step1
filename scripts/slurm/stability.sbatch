#!/bin/bash
#SBATCH --job-name=dpp1_stability
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=logs/stability-%j.out
#SBATCH --error=logs/stability-%j.err

# Usage:
#   sbatch scripts/slurm/stability.sbatch [--trials 3] [--samples 10]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source .venv/bin/activate
PYTHONPATH="$PWD" python tests/stability_test.py "$@"
