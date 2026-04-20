#!/bin/bash
#SBATCH --job-name=dpp1_diag
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/diag-%j.out
#SBATCH --error=logs/diag-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "=== python ==="
which python; python --version
echo "=== pip ==="
python -m pip --version
echo "=== pip config ==="
python -m pip config list
echo "=== pypi reachability ==="
python -c "import urllib.request; r=urllib.request.urlopen('https://pypi.org/simple/data-designer/', timeout=10); print('HTTP', r.status, 'bytes', len(r.read()))"
echo "=== pip index versions data-designer ==="
python -m pip index versions data-designer 2>&1 | head -20
echo "=== pip index versions nvidia-nat ==="
python -m pip index versions nvidia-nat 2>&1 | head -20
