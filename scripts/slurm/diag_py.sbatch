#!/bin/bash
#SBATCH --job-name=dpp1_diagpy
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/diagpy-%j.out
#SBATCH --error=logs/diagpy-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "=== /usr/bin/python* ==="
ls -la /usr/bin/python* 2>&1 || true
echo "=== /opt/* /usr/local/* python candidates ==="
ls /opt 2>&1 | head -30
ls /usr/local 2>&1 | head -30
echo "=== module avail (grep python) ==="
bash -lc 'module avail 2>&1 | grep -i -E "python|conda|mamba" || true'
echo "=== conda / mamba / pyenv / uv / micromamba on PATH ==="
for cmd in conda mamba micromamba pyenv uv python3.10 python3.11 python3.12; do
  echo "--- $cmd ---"
  which $cmd 2>&1 || echo "(not found)"
done
