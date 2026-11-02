#!/usr/bin/env bash
#SBATCH --job-name=b2t5b
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Several rows of tool_grid.tsv in one job, one after the other. Usage (from chimera/): sbatch -c 8 --mem=32G tools/t5_bundle.sbatch <saturn|camex> i j k ...
source revision/slurm/_body.sh
export DGLBACKEND=pytorch
TOOL="$1"; shift; M=revision/tasks/B2_benchmark/tools/tool_grid.tsv; U=$(echo "$TOOL" | tr a-z A-Z); PP=$PYTHONPATH
for I in "$@"; do
  run=$(awk -F'\t' -v i=$((I+2)) 'NR==i{print $5}' $M); seed=$(awk -F'\t' -v i=$((I+2)) 'NR==i{print $4}' $M)
  [[ -f results/revision/B2/tool_scores/$U/$run.csv ]] && continue
  export PYTHONPATH=$PP
  mamba run --no-capture-output -n chimera_env python -m revision.tasks.B2_benchmark.tools.prep_inputs --manifest $M --index "$I" || { echo "FAILED prep $run"; continue; }
  unset PYTHONPATH
  mamba run --no-capture-output -n ${TOOL}_env python revision/tasks/B2_benchmark/tools/run_${TOOL}_b2.py --run "$run" --seed "$seed" || echo "FAILED $run"
  rm -rf "data/revision/B2/tool_inputs/$run/camex" "data/revision/B2/tool_inputs/$run/saturn" "data/revision/B2/tool_inputs/$run/object.h5ad.tmp"
done
