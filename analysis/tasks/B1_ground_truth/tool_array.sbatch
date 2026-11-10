#!/usr/bin/env bash
#SBATCH --job-name=b1tool
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch --array=0-N%K -c <cpus> --mem=<mem> --output=results/analysis/B1/logs/%x_%A_%a.out \
#   analysis/tasks/B1_ground_truth/tool_array.sbatch <saturn|camex> analysis/tasks/B1_ground_truth/grid_tools.tsv
source analysis/tasks/T5_2_adapters/_tool_body.sh
set -e
tool="$1"; M="$2"; I=$SLURM_ARRAY_TASK_ID
export PYTHONPATH=$PWD B1_TOOL=$tool
T=analysis/tasks/B1_ground_truth/tools.py
get() { python3 -c "import csv;print(list(csv.DictReader(open('$M'),delimiter='\t'))[$I]['$1'])"; }
run=$(get run); stem=$(get stem); seed=$(get seed)
out=results/analysis/B1/tool_scores/${tool}__$run.parquet
if [[ -f "$out" ]]; then echo "already scored $tool $run"; exit 0; fi
W=data/analysis/B1/tools/$tool/$run
# attempt record for completion.py, once the tool itself starts; a node that cannot even start the job leaves none
mkdir -p results/analysis/B1/attempts/$tool
record_rc() { local rc=$?; [[ $rc -eq 127 || -f "$out" ]] || echo "$rc" > "results/analysis/B1/attempts/$tool/$run.rc"; }  # 127: the node, not the tool
trap record_rc EXIT
if [[ ! -f "$W/altered.h5ad" ]]; then mamba run --no-capture-output -n chimera_env python -m analysis.tasks.B1_ground_truth.tools build --manifest "$M" --index "$I"; fi
t0=$SECONDS
if [[ $tool == genespectra || $tool == cacimar || $tool == samap ]]; then
  # the adapters of the overview benchmark (analysis/tasks/B2_benchmark/tools), own env, same primary scalar
  lk=$(get label_key); cohort=$(get cohort)
  mkdir -p "$W/out"
  mamba run --no-capture-output -n ${tool}_env python analysis/tasks/B2_benchmark/tools/run_$tool.py --h5ad "$W/altered.h5ad" --cohort "$cohort" \
    --label-key "$lk" --seed "$seed" --out "$W/out/$tool.csv" --work "$W/work"
  echo "tool_elapsed_s=$((SECONDS - t0))"
  mamba run --no-capture-output -n chimera_env python -m analysis.tasks.B1_ground_truth.tools score_csv --manifest "$M" --index "$I"
elif [[ $tool == saturn ]]; then
  rm -rf "$W/saturn_work" "$W/saturn_final"
  mamba run --no-capture-output -n saturn_env python $T prep_saturn --manifest "$M" --index "$I"
  mamba run --no-capture-output -n saturn_env python analysis/tasks/T5_2_adapters/run_saturn.py --stem "$stem" --labels cell_type --seed "$seed" \
    --work "$W/saturn_work" --dest "$W/saturn_final" --inputs "$W/saturn_in" --hv-genes 40000 --method-name saturn_1to1
  echo "tool_elapsed_s=$((SECONDS - t0))"
  mamba run --no-capture-output -n chimera_env python -m analysis.tasks.B1_ground_truth.tools score_saturn --manifest "$M" --index "$I"
else
  rm -rf "$W/camex_work" "$W/camex_final"
  mamba run --no-capture-output -n camex_env python $T prep_camex --manifest "$M" --index "$I"
  mamba run --no-capture-output -n camex_env python analysis/tasks/T5_2_adapters/run_camex.py --stem "$stem" --seed "$seed" \
    --work "$W/camex_work" --dest "$W/camex_final" --inputs "$W/camex_in"
  echo "tool_elapsed_s=$((SECONDS - t0))"
  mamba run --no-capture-output -n chimera_env python -m analysis.tasks.B1_ground_truth.tools score_camex --manifest "$M" --index "$I"
fi
rm -rf "$W"
