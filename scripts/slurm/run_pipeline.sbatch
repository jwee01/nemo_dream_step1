#!/bin/bash
#SBATCH --job-name=dpp1_pipeline
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pipeline-%j.out
#SBATCH --error=logs/pipeline-%j.err

# Usage:
#   sbatch scripts/slurm/run_pipeline.sbatch --input path/to/input.jsonl --output path/to/stage12.jsonl
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source .venv/bin/activate
PYTHONPATH="$PWD" python scripts/run_pipeline.py "$@"
