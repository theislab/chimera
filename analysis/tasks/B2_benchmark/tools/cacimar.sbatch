#!/usr/bin/env bash
#SBATCH --job-name=b2t_cacimar
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# Usage, from chimera/: sbatch --output=<log> analysis/tasks/B2_benchmark/tools/cacimar.sbatch \
#   --h5ad <path> --cohort <cohort> --label-key <obs col> --seed <int> --out <csv> [--work <dir>]
set -euo pipefail
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
exec mamba run --no-capture-output -n cacimar_env python analysis/tasks/B2_benchmark/tools/run_cacimar.py "$@"
