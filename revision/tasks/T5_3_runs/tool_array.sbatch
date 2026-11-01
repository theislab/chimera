#!/usr/bin/env bash
#SBATCH --job-name=chimera_T5_3
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch --array=0-N%K --output=results/revision/T5_3/logs/%x_%A_%a.out \
#   revision/tasks/T5_3_runs/tool_array.sbatch revision/tasks/T5_3_runs/<manifest>.tsv
# Manifest: header line, then "tool<TAB>stem<TAB>labels<TAB>seed<TAB>work<TAB>dest".
source revision/tasks/T5_2_adapters/_tool_body.sh
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$1")
[[ -n "$row" ]] || { echo "no manifest row for task $SLURM_ARRAY_TASK_ID"; exit 2; }
IFS=$'\t' read -r tool stem labels seed work dest <<<"$row"
run_tool "$tool" "$stem" "$labels" "$seed" "$work" "$dest"
