#!/usr/bin/env bash
#SBATCH --job-name=b1repair
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=cpusrv114
# Direct readouts on the altered object of a fit that left no scores; the fit is never rerun.
source revision/slurm/_body.sh
set -e
M=revision/tasks/B1_ground_truth/repair.tsv; I=$SLURM_ARRAY_TASK_ID
PY="mamba run --no-capture-output -n chimera_env python"
run=$(python3 -c "import csv;print(list(csv.DictReader(open('$M'),delimiter='\t'))[$I]['run'])")
h5=data/revision/B1/h5ad/$run.h5ad
[[ -f "$h5" ]] || $PY -m revision.tasks.B1_ground_truth.step --manifest "$M" --index "$I"
$PY -m revision.tasks.B1_ground_truth.score_run --manifest "$M" --index "$I" --h5ad "$h5"
rm -f "$h5"
