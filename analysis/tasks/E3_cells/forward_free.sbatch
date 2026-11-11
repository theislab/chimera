#!/usr/bin/env bash
#SBATCH --job-name=e3_fw
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=02:00:00
# E3 forward passes of the Z4 per-gene gate fits (checkpoint only).
# Usage (from chimera/): sbatch --array=0-62%12 -c 4 --mem=12G --output=results/analysis/E3/logs/%x_%A_%a.out analysis/tasks/E3_cells/forward_free.sbatch
source analysis/slurm/_body.sh
mamba run --no-capture-output -n chimera_env python analysis/tasks/E3_cells/forward_free.py run --row "$SLURM_ARRAY_TASK_ID" --threads "${SLURM_CPUS_PER_TASK:-4}"
