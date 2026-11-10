#!/usr/bin/env bash
#SBATCH --job-name=y5_seeds
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
source analysis/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.Y5_aspects.seeds row "$SLURM_ARRAY_TASK_ID"
