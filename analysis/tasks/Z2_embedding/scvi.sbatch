#!/usr/bin/env bash
#SBATCH --job-name=z2_scvi
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=results/analysis/Z2/logs/%x_%A_%a.out
source analysis/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.Z2_embedding.scvi_train row "$SLURM_ARRAY_TASK_ID"
