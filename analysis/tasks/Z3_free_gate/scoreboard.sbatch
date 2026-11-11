#!/usr/bin/env bash
#SBATCH --job-name=z3_sb
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=06:00:00
# Usage (from chimera/): sbatch -c 40 --mem=48G --output=results/analysis/Z3/logs/%x_%j.out analysis/tasks/Z3_free_gate/scoreboard.sbatch <line>
source analysis/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.Z3_free_gate.scoreboard --line "$1" --workers 38
