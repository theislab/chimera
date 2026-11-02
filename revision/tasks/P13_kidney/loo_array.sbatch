#!/usr/bin/env bash
#SBATCH --job-name=p13_loo
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=23:50:00
# Leave-one-out primary DE fits for the H-D1 sensitivity (plan section 8c), one task per type.
# Usage: sbatch --array=0-8 revision/tasks/P13_kidney/loo_array.sbatch
source revision/slurm/_body.sh
mamba run --no-capture-output -n revision_de_env python revision/tasks/P13_kidney/de.py \
    --loo "$SLURM_ARRAY_TASK_ID" --n-cpus "${SLURM_CPUS_PER_TASK:-8}"
