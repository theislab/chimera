#!/usr/bin/env bash
#SBATCH --job-name=T5_6_analysis
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Scalars, their analysis (scalar_analysis.py), then the scoreboard with the primary tool columns.
# Usage (from chimera/): sbatch --output=results/analysis/T5_6/logs/%x_%j.out analysis/tasks/T5_6_gene_level/analysis.sbatch
set -euo pipefail
command -v mamba >/dev/null 2>&1 || PATH="$(dirname "${MAMBA_EXE:-${CONDA_EXE:-/nonexistent/mamba}}"):$PATH"
command -v mamba >/dev/null 2>&1 || { echo "mamba not found: put its bin folder on PATH before submitting" >&2; exit 2; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "${CHIMERA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 2
[[ -d analysis/tasks ]] || { echo "submit from the repository root or set CHIMERA_ROOT" >&2; exit 2; }
export PYTHONPATH=$PWD
echo "sha=$(git rev-parse HEAD) start=$(date -Is)"
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T5_6_gene_level.scalars
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T5_6_gene_level.scalar_analysis
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T4_3_scoreboard.run --mode real \
  --tools results/analysis/T5_6/tools_primary.csv --outdir results/analysis/T5_6/scoreboard --workers 16
echo "end=$(date -Is)"
