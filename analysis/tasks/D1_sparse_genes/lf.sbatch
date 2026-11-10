#!/usr/bin/env bash
#SBATCH --job-name=d1_lat
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=03:00:00
#SBATCH --exclude=cpusrv114
# Usage (from chimera/): sbatch --array=<rows>%N -c 8 --mem=16G --output=results/analysis/D1/logs/%x_%A_%a.out analysis/tasks/D1_sparse_genes/lf.sbatch
source analysis/slurm/_body.sh
set -e
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.D1_sparse_genes.labelfree latent --index "$SLURM_ARRAY_TASK_ID"
