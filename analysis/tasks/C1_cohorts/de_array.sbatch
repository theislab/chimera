#!/usr/bin/env bash
#SBATCH --job-name=c1_de
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=23:50:00
# sbatch --array=0-K%L de_array.sbatch <case> <n_chunks>; omit the array for the primary fits (give only <case>)
source analysis/slurm/_body.sh
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  mamba run --no-capture-output -n revision_de_env python analysis/tasks/C1_cohorts/de_case.py --case "$1" --perm-task "$SLURM_ARRAY_TASK_ID" --n-chunks "$2" --n-cpus "${SLURM_CPUS_PER_TASK:-4}"
else
  mamba run --no-capture-output -n revision_de_env python analysis/tasks/C1_cohorts/de_case.py --case "$1" --n-cpus "${SLURM_CPUS_PER_TASK:-4}"
fi
