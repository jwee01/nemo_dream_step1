#!/bin/bash
#SBATCH --job-name=dpp1_install
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/install-%j.out
#SBATCH --error=logs/install-%j.err

# Creates an isolated .venv with Python 3.11 using `uv`, then installs requirements.
# System python is 3.9.21 and `data-designer` / `nvidia-nat` need >=3.10.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install -r requirements.txt
python -c "import data_designer, nat, openai, pydantic, numpy; print('imports ok')"
