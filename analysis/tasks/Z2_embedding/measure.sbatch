#!/usr/bin/env bash
#SBATCH --job-name=z2_meas
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/analysis/Z2/logs/%x_%A_%a.out
# Usage: sbatch --export=ALL,COHORTS="a b c" --array=0-N analysis/tasks/Z2_embedding/measure.sbatch
source analysis/slurm/_body.sh
read -r -a cs <<<"$COHORTS"
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.Z2_embedding.measure cohort "${cs[$SLURM_ARRAY_TASK_ID]}"
