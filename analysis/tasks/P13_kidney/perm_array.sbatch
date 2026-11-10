#!/usr/bin/env bash
#SBATCH --job-name=p13_perm
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=23:50:00
# Permuted-condition DE for the H-D4 / H-D1 label null (plan section 8c): 9 types x N_CHUNKS tasks.
# Usage: sbatch --array=0-$((9*N_CHUNKS-1))%16 analysis/tasks/P13_kidney/perm_array.sbatch <N_CHUNKS>
source analysis/slurm/_body.sh
mamba run --no-capture-output -n revision_de_env python analysis/tasks/P13_kidney/de.py \
    --perm-task "$SLURM_ARRAY_TASK_ID" --n-chunks "${1:-1}" --n-cpus "${SLURM_CPUS_PER_TASK:-8}"
