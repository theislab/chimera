#!/usr/bin/env bash
#SBATCH --job-name=b2add
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Added tools on settings' objects, one after the other. Usage (from chimera/): sbatch -c N --mem=M tools/added.sbatch <genespectra|cacimar|samap> <cohort> <table key> rep setting [setting ...]
source revision/slurm/_body.sh
TOOL="$1"; C="$2"; TABLE="$3"; REP="$4"; shift 4
declare -A NAME=([genespectra]=GeneSpectra [cacimar]=CACIMAR [samap]=SAMap)
O=results/revision/B2/tool_scores/${NAME[$TOOL]}; mkdir -p "$O"; W=data/revision/B2/tool_work; mkdir -p "$W"; PP=$PYTHONPATH
for S in "$@"; do
  run=b2-$C-$S-r$REP
  [[ -f $O/$run.csv ]] && continue
  h=$W/${TOOL}_$run.h5ad
  export PYTHONPATH=$PP
  mamba run --no-capture-output -n chimera_env python -m revision.tasks.B2_benchmark.tools.prep_object "$C" "$S" "$REP" "$h" || { echo "FAILED prep $run"; continue; }
  unset PYTHONPATH
  rc=0; mamba run --no-capture-output -n ${TOOL}_env python revision/tasks/B2_benchmark/tools/run_${TOOL}.py --h5ad "$h" --cohort "$TABLE" --label-key b2_label --seed $((142 + REP)) --out "$O/$run.csv" --work "$W/${TOOL}_$run" || { rc=$?; echo "FAILED $run"; }
  mkdir -p results/revision/B2/attempts/${NAME[$TOOL]}; echo "$rc" > results/revision/B2/attempts/${NAME[$TOOL]}/$run.rc
  rm -rf "$h" "$W/${TOOL}_$run"
done
