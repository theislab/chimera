#!/usr/bin/env bash
#SBATCH --job-name=d1_a
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=03:00:00
# Usage (from chimera/): sbatch -c 16 --mem=16G --output=results/revision/D1/logs/%x_%j.out revision/tasks/D1_sparse_genes/a.sbatch run|goldfree
source revision/slurm/_body.sh
set -e
mamba run --no-capture-output -n chimera_env python -m revision.tasks.D1_sparse_genes.a_masked "$1"
