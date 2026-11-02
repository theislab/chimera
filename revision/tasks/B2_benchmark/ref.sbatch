#!/usr/bin/env bash
#SBATCH --job-name=b2ref
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=06:00:00
# Scores the reused unaltered fits; never changes their run folders.
source revision/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python -m revision.tasks.B2_benchmark.score_run --manifest "$1" --index "$SLURM_ARRAY_TASK_ID"
