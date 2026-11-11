#!/usr/bin/env bash
#SBATCH --job-name=z4_free
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Z4 Config M (free per-gene gate) seeds on every harmonized cohort; recipe = the cohort's chosen canonical recipe with --config M (as Z3_free_gate).
# Usage (from chimera/): sbatch --array=<rows>%<n> -c 8 --mem=<mem of those rows> --output=results/analysis/Z4/logs/%x_%A_%a.out analysis/tasks/Z4_free_gate_all/train.sbatch
source analysis/slurm/_body.sh
M=${Z4_MANIFEST:-analysis/tasks/Z4_free_gate_all/manifest.tsv}; I=$SLURM_ARRAY_TASK_ID
run=$(python3 -c "import csv;r=list(csv.DictReader(open('$M'),delimiter='\t'));print(r[$I]['run_name'])")
args=$(python3 -c "import csv;r=list(csv.DictReader(open('$M'),delimiter='\t'));print(r[$I]['args'])")
train_run "$run" $args
