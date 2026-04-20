#!/bin/bash
#SBATCH --job-name=dpp1_netprobe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/netprobe-%j.out
#SBATCH --error=logs/netprobe-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source .venv/bin/activate
echo "=== env proxy vars ==="
env | grep -i -E "proxy|pip" || true
echo
echo "=== getent hosts ==="
getent hosts integrate.api.nvidia.com || true
echo
PYTHONPATH="$PWD" python tests/net_probe.py
