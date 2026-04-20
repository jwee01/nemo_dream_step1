#!/bin/bash
#SBATCH --job-name=dpp1_build_index
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/build_index-%j.out
#SBATCH --error=logs/build_index-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source .venv/bin/activate
python -m src.part2_cultural.index_builder
