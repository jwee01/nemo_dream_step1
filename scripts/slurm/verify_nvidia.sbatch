#!/bin/bash
#SBATCH --job-name=dpp1_verify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify-%j.out
#SBATCH --error=logs/verify-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source .venv/bin/activate
PYTHONPATH="$PWD" python tests/verify_nvidia_paths.py
