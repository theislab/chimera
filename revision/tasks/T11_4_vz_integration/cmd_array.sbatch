#!/usr/bin/env bash
#SBATCH --job-name=rev_vz_base
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch --array=0-N --output=results/revision/T11/logs/%x_%A_%a.out \
#   revision/tasks/T11_4_vz_integration/cmd_array.sbatch <manifest.tsv>
# Manifest: header line, then "<name><TAB><arguments to training/baseline_runner.py>".
source revision/slurm/_body.sh
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$1")
[[ -n "$row" ]] || { echo "no manifest row for task $SLURM_ARRAY_TASK_ID"; exit 2; }
name=$(cut -f1 <<<"$row"); args=$(cut -f2- <<<"$row")
echo "$name: $args"
# shellcheck disable=SC2086
mamba run --no-capture-output -n chimera_env python training/baseline_runner.py $args
