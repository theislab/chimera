#!/usr/bin/env bash
#SBATCH --job-name=z2_scvi
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/revision/Z2/logs/%x_%A_%a.out
source revision/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python -m revision.tasks.Z2_embedding.scvi_train row "$SLURM_ARRAY_TASK_ID"
