#!/usr/bin/env bash
#SBATCH --job-name=chimera_rev
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=23:50:00
# Usage: sbatch --output=results/revision/<task>/logs/%x_%j.out revision/slurm/cpu.sbatch <command...>
source revision/slurm/_body.sh
"$@"
