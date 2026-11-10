#!/usr/bin/env bash
#SBATCH --job-name=chimera_cpu
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=23:50:00
# One command as a CPU job. Submit from the repository root (or set CHIMERA_ROOT):
#   sbatch --output=results/analysis/<task>/logs/%x_%j.out analysis/slurm/cpu.sbatch <command...>
# Partition and QOS above are those of the authors' cluster; change them for yours.
source analysis/slurm/_body.sh
"$@"
