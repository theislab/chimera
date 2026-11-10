#!/usr/bin/env bash
#SBATCH --job-name=p13_loo
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=23:50:00
# Leave-one-out primary DE fits, H-D1 sensitivity (pre-specified analysis plan, section 8c).
# One task per type. Usage: sbatch --array=0-8 analysis/tasks/P13_kidney/loo_array.sbatch
source analysis/slurm/_body.sh
mamba run --no-capture-output -n revision_de_env python analysis/tasks/P13_kidney/de.py \
    --loo "$SLURM_ARRAY_TASK_ID" --n-cpus "${SLURM_CPUS_PER_TASK:-8}"
