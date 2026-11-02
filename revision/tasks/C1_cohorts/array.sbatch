#!/usr/bin/env bash
#SBATCH --job-name=c1_train
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=23:50:00
#SBATCH --exclude=cpusrv09,cpusrv18
# Usage: sbatch --array=0-N --output=results/revision/C1/logs/%x_%A_%a.out revision/tasks/C1_cohorts/array.sbatch <manifest.tsv>
source revision/slurm/_body.sh
manifest="$1"
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$manifest")
name=$(cut -f1 <<<"$row")
args=$(cut -f2- <<<"$row")
# shellcheck disable=SC2086
train_run "$name" $args
