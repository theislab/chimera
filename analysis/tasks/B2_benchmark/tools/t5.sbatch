#!/usr/bin/env bash
#SBATCH --job-name=b2t5
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch --array=... -c 16 --mem=48G --nice=10 --output=results/analysis/B2/logs/%x_%A_%a.out analysis/tasks/B2_benchmark/tools/t5.sbatch <saturn|camex>
source analysis/slurm/_body.sh
export DGLBACKEND=pytorch
TOOL="$1"; I=$SLURM_ARRAY_TASK_ID; M=analysis/tasks/B2_benchmark/tools/tool_grid.tsv
run=$(awk -F'\t' -v i=$((I+2)) 'NR==i{print $5}' $M); seed=$(awk -F'\t' -v i=$((I+2)) 'NR==i{print $4}' $M)
U=$(echo "$TOOL" | tr a-z A-Z)
[[ -f results/analysis/B2/tool_scores/$U/$run.csv ]] && { echo done; exit 0; }
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.B2_benchmark.tools.prep_inputs --manifest $M --index $I || exit 1
unset PYTHONPATH
rc=0; mamba run --no-capture-output -n ${TOOL}_env python analysis/tasks/B2_benchmark/tools/run_${TOOL}_b2.py --run "$run" --seed "$seed" || rc=$?
mkdir -p results/analysis/B2/attempts/$U; echo "$rc" > results/analysis/B2/attempts/$U/$run.rc
exit $rc
