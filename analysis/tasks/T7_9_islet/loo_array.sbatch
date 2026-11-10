#!/usr/bin/env bash
#SBATCH --job-name=t79_loo
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=23:50:00
# Leave-one-out primary DE fits for the H-D1 sensitivity (plan section 8b), one task per DE-able type.
# Usage: sbatch --array=0-5 --output=results/analysis/T7_9/logs/%x_%A_%a.out analysis/tasks/T7_9_islet/loo_array.sbatch
source analysis/slurm/_body.sh
mamba run --no-capture-output -n revision_de_env python analysis/tasks/T7_9_islet/de.py \
    --loo "$SLURM_ARRAY_TASK_ID" --n-cpus "${SLURM_CPUS_PER_TASK:-8}"
