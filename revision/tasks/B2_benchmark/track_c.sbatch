#!/usr/bin/env bash
#SBATCH --job-name=b2c
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=12:00:00
# Usage (from chimera/): sbatch --array=... -c 2 --mem=8G --nice=10 --output=results/revision/B2/logs/%x_%A_%a.out revision/tasks/B2_benchmark/track_c.sbatch <baselines|unit|lists|y4>
source revision/slurm/_body.sh
PY="mamba run --no-capture-output -n chimera_env python"
case "$1" in
  baselines) $PY -m revision.tasks.B2_benchmark.baselines_c "$SLURM_ARRAY_TASK_ID" ;;
  unit) $PY -m revision.tasks.B2_benchmark.track_c unit --index "$SLURM_ARRAY_TASK_ID" ;;
  lists) $PY -m revision.tasks.B2_benchmark.track_c lists --index "$SLURM_ARRAY_TASK_ID" ;;
  y4) $PY -m revision.tasks.B2_benchmark.track_c y4 ;;
esac
