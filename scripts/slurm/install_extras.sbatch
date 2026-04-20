#!/bin/bash
#SBATCH --job-name=dpp1_extras
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/extras-%j.out
#SBATCH --error=logs/extras-%j.err

# Installs additional NAT extras (LangChain adapters) on top of existing .venv.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
uv pip install 'nvidia-nat[langchain]'
python -c "import nat.plugins.langchain; print('nat langchain plugin ok')" || \
  python -c "from langchain_nvidia_ai_endpoints import ChatNVIDIA; print('langchain-nvidia ok')"
