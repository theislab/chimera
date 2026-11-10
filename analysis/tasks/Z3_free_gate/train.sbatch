#!/usr/bin/env bash
#SBATCH --job-name=z3_free
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
#SBATCH --exclude=cpusrv114
# Z3 Config M (free per-gene gate) seeds; recipe = canonical recipe of the cohort with --config M.
# Usage (from chimera/): sbatch --array=<i> -c 40 --mem=<mem from manifest> --output=results/analysis/Z3/logs/%x_%A_%a.out analysis/tasks/Z3_free_gate/train.sbatch
source analysis/slurm/_body.sh
M=analysis/tasks/Z3_free_gate/manifest.tsv; I=$SLURM_ARRAY_TASK_ID
run=$(python3 -c "import csv;r=list(csv.DictReader(open('$M'),delimiter='\t'));print(r[$I]['run_name'])")
args=$(python3 -c "import csv;r=list(csv.DictReader(open('$M'),delimiter='\t'));print(r[$I]['args'])")
train_run "$run" $args
