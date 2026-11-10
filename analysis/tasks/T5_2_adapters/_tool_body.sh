# Shared body for SATURN / CAMEX jobs. Sourced by the sbatch files in T5_2_adapters and T5_3_runs.
set -uo pipefail
command -v mamba >/dev/null 2>&1 || PATH="$(dirname "${MAMBA_EXE:-${CONDA_EXE:-/nonexistent/mamba}}"):$PATH"
command -v mamba >/dev/null 2>&1 || { echo "mamba not found: put its bin folder on PATH before submitting" >&2; exit 2; }
N=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N
export VECLIB_MAXIMUM_THREADS=$N NUMBA_NUM_THREADS=$N PYTHONUNBUFFERED=1 MPLBACKEND=Agg DGLBACKEND=pytorch
# The repository root is $CHIMERA_ROOT if set, else found from the location of this file.
cd "${CHIMERA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}" || exit 2
A=analysis/tasks/T5_2_adapters
echo "host=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} cpus=$N sha=$(git rev-parse HEAD) start=$(date -Is)"

# run_tool <saturn|camex> <stem> <labels|-> <seed> <work> <dest>
run_tool() {
  local tool=$1 stem=$2 labels=$3 seed=$4 work=$5 dest=$6
  if [[ -f "$dest/z_mean.npy" && -f "$dest/config.json" ]]; then echo "finished: $dest"; return 0; fi
  if [[ -e "$work" ]]; then
    mkdir -p "$(dirname "$work")/_failed"
    mv "$work" "$(dirname "$work")/_failed/$(basename "$work")_${SLURM_JOB_ID:-x}"
  fi
  local rc=0 t0=$SECONDS
  if [[ $tool == saturn ]]; then
    mamba run --no-capture-output -n saturn_env python $A/run_saturn.py \
      --stem "$stem" --labels "$labels" --seed "$seed" --work "$work" --dest "$dest" || rc=$?
  elif [[ $tool == saturn_1to1 ]]; then
    mamba run --no-capture-output -n saturn_env python $A/run_saturn.py \
      --stem "$stem" --labels "$labels" --seed "$seed" --work "$work" --dest "$dest" \
      --inputs "data/analysis/T5_inputs/$stem/saturn_1to1" --hv-genes 40000 --method-name saturn_1to1 || rc=$?
  else
    mamba run --no-capture-output -n camex_env python $A/run_camex.py \
      --stem "$stem" --seed "$seed" --work "$work" --dest "$dest" || rc=$?
  fi
  echo "rc=$rc elapsed_s=$((SECONDS - t0)) end=$(date -Is)"
  return $rc
}
