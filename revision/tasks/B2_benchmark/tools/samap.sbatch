#!/usr/bin/env bash
#SBATCH --job-name=b2t_samap
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:30:00
# Usage, from chimera/: sbatch --output=<log> revision/tasks/B2_benchmark/tools/samap.sbatch \
#   --h5ad <path> --cohort <cohort> --label-key <obs col> --seed <int> --out <csv> [--work <dir>]
set -euo pipefail
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
exec mamba run --no-capture-output -n samap_env python revision/tasks/B2_benchmark/tools/run_samap.py "$@"
