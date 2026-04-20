#!/bin/bash
#SBATCH --job-name=dpp1_smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source .venv/bin/activate
PYTHONPATH="$PWD" python tests/smoke_test.py
