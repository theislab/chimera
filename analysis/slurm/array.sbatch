#!/usr/bin/env bash
#SBATCH --job-name=chimera_rev_arr
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=23:50:00
# Usage: sbatch --array=0-N --output=results/analysis/<task>/logs/%x_%A_%a.out \
#          analysis/slurm/array.sbatch <manifest.tsv>
# Manifest: header line, then "<run_name><TAB><run_experiment args>".
source analysis/slurm/_body.sh
manifest="$1"
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$manifest")
name=$(cut -f1 <<<"$row")
args=$(cut -f2- <<<"$row")
# shellcheck disable=SC2086
train_run "$name" $args
