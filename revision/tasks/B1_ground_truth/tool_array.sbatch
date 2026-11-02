#!/usr/bin/env bash
#SBATCH --job-name=b1tool
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch --array=0-N%K -c <cpus> --mem=<mem> --output=results/revision/B1/logs/%x_%A_%a.out \
#   revision/tasks/B1_ground_truth/tool_array.sbatch <saturn|camex> revision/tasks/B1_ground_truth/grid_tools.tsv
source revision/tasks/T5_2_adapters/_tool_body.sh
set -e
tool="$1"; M="$2"; I=$SLURM_ARRAY_TASK_ID
export PYTHONPATH=$PWD B1_TOOL=$tool
T=revision/tasks/B1_ground_truth/tools.py
get() { python3 -c "import csv;print(list(csv.DictReader(open('$M'),delimiter='\t'))[$I]['$1'])"; }
run=$(get run); stem=$(get stem); seed=$(get seed)
out=results/revision/B1/tool_scores/${tool}__$run.parquet
if [[ -f "$out" ]]; then echo "already scored $tool $run"; exit 0; fi
W=data/revision/B1/tools/$tool/$run
if [[ ! -f "$W/altered.h5ad" ]]; then mamba run --no-capture-output -n chimera_env python -m revision.tasks.B1_ground_truth.tools build --manifest "$M" --index "$I"; fi
t0=$SECONDS
if [[ $tool == saturn ]]; then
  rm -rf "$W/saturn_work" "$W/saturn_final"
  mamba run --no-capture-output -n saturn_env python $T prep_saturn --manifest "$M" --index "$I"
  mamba run --no-capture-output -n saturn_env python revision/tasks/T5_2_adapters/run_saturn.py --stem "$stem" --labels cell_type --seed "$seed" \
    --work "$W/saturn_work" --dest "$W/saturn_final" --inputs "$W/saturn_in" --hv-genes 40000 --method-name saturn_1to1
  echo "tool_elapsed_s=$((SECONDS - t0))"
  mamba run --no-capture-output -n chimera_env python -m revision.tasks.B1_ground_truth.tools score_saturn --manifest "$M" --index "$I"
else
  rm -rf "$W/camex_work" "$W/camex_final"
  mamba run --no-capture-output -n camex_env python $T prep_camex --manifest "$M" --index "$I"
  mamba run --no-capture-output -n camex_env python revision/tasks/T5_2_adapters/run_camex.py --stem "$stem" --seed "$seed" \
    --work "$W/camex_work" --dest "$W/camex_final" --inputs "$W/camex_in"
  echo "tool_elapsed_s=$((SECONDS - t0))"
  mamba run --no-capture-output -n chimera_env python -m revision.tasks.B1_ground_truth.tools score_camex --manifest "$M" --index "$I"
fi
rm -rf "$W"
