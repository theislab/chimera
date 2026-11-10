#!/usr/bin/env bash
#SBATCH --job-name=b2sb
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
# Usage (from chimera/): sbatch --array=0-N%K --output=results/analysis/B2/logs/%x_%A_%a.out analysis/tasks/B2_benchmark/score_b.sbatch
source analysis/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.B2_benchmark.score_b --index "$SLURM_ARRAY_TASK_ID"
