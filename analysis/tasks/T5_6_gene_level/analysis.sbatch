#!/usr/bin/env bash
#SBATCH --job-name=T5_6_analysis
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Scalars, analysis, then the scoreboard with the primary tool columns (official folder untouched).
# Usage (from chimera/): sbatch --output=results/analysis/T5_6/logs/%x_%j.out analysis/tasks/T5_6_gene_level/analysis.sbatch
set -euo pipefail
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "${CHIMERA_ROOT:-/home/icb/kemal.inecik/lustre_workspace/master_chimera/chimera}"
export PYTHONPATH=$PWD
echo "sha=$(git rev-parse HEAD) start=$(date -Is)"
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T5_6_gene_level.scalars
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T5_6_gene_level.analysis
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T4_3_scoreboard.run --mode real \
  --tools results/analysis/T5_6/tools_primary.csv --outdir results/analysis/T5_6/scoreboard --workers 16
echo "end=$(date -Is)"
