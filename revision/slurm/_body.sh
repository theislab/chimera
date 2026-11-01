# Shared job body, sourced by cpu.sbatch and array.sbatch.
set -uo pipefail
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "${CHIMERA_ROOT:-/home/icb/kemal.inecik/lustre_workspace/master_chimera/chimera}" || exit 2
export PYTHONPATH=$PWD

# Write training/runs/$1/_provenance.json next to the run: SHA, argv, lock hash, job, host.
provenance() {
  local run_dir="training/runs/$1"; shift
  mkdir -p "$run_dir"
  python3 - "$run_dir" "$@" <<'PY'
import hashlib, json, os, pathlib, socket, subprocess, sys
lock = pathlib.Path("revision/envs/chimera_env.linux-64.lock.txt")
json.dump({
    "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "argv": sys.argv[2:],
    "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "host": socket.gethostname(),
}, open(sys.argv[1] + "/_provenance.json", "w"), indent=2)
PY
}

# train_run <run_name> <args to run_experiment.py after --run-name>
# Exit 1 from run_experiment.py means expectations failed, not that the run failed.
train_run() {
  local name="$1"; shift
  if [[ -e "training/runs/$name" ]]; then
    if [[ -f "training/runs/$name/summary.json" ]]; then echo "finished: $name"; return 0; fi
    mkdir -p training/runs/_failed
    mv "training/runs/$name" "training/runs/_failed/${name}_${SLURM_JOB_ID:-x}"
  fi
  provenance "$name" "$@"
  local rc=0
  mamba run --no-capture-output -n chimera_env python training/run_experiment.py \
    --run-name "$name" --accelerator cpu "$@" || rc=$?
  [[ -f "training/runs/$name/summary.json" && -f "training/runs/$name/final_state.npz" ]] || exit "${rc:-1}"
}
