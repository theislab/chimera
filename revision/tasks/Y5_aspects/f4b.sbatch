#!/usr/bin/env bash
#SBATCH --job-name=y5_f4b
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
source revision/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python -m revision.tasks.Y5_aspects.f4b "$SLURM_ARRAY_TASK_ID"
