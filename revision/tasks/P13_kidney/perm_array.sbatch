#!/usr/bin/env bash
#SBATCH --job-name=p13_perm
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=23:50:00
source revision/slurm/_body.sh
mamba run --no-capture-output -n revision_de_env python revision/tasks/P13_kidney/de.py --perm "$SLURM_ARRAY_TASK_ID" --n-cpus 16
