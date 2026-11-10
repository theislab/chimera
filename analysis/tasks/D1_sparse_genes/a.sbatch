#!/usr/bin/env bash
#SBATCH --job-name=d1_a
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=03:00:00
# Usage (from chimera/): sbatch -c 16 --mem=16G --output=results/analysis/D1/logs/%x_%j.out analysis/tasks/D1_sparse_genes/a.sbatch run|goldfree
source analysis/slurm/_body.sh
set -e
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.D1_sparse_genes.a_masked "$1"
