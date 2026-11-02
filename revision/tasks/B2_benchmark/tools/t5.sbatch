#!/usr/bin/env bash
#SBATCH --job-name=b2t5
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch --array=... -c 16 --mem=48G --nice=10 --output=results/revision/B2/logs/%x_%A_%a.out revision/tasks/B2_benchmark/tools/t5.sbatch <saturn|camex>
source revision/slurm/_body.sh
export DGLBACKEND=pytorch
TOOL="$1"; I=$SLURM_ARRAY_TASK_ID; M=revision/tasks/B2_benchmark/tools/tool_grid.tsv
run=$(awk -F'\t' -v i=$((I+2)) 'NR==i{print $5}' $M); seed=$(awk -F'\t' -v i=$((I+2)) 'NR==i{print $4}' $M)
U=$(echo "$TOOL" | tr a-z A-Z)
[[ -f results/revision/B2/tool_scores/$U/$run.csv ]] && { echo done; exit 0; }
mamba run --no-capture-output -n chimera_env python -m revision.tasks.B2_benchmark.tools.prep_inputs --manifest $M --index $I || exit 1
unset PYTHONPATH
mamba run --no-capture-output -n ${TOOL}_env python revision/tasks/B2_benchmark/tools/run_${TOOL}_b2.py --run "$run" --seed "$seed"
