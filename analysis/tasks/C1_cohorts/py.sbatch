#!/usr/bin/env bash
#SBATCH --job-name=c1_py
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:50:00
#SBATCH --exclude=cpusrv09,cpusrv18,cpusrv63,cpusrv114
# sbatch py.sbatch <script under C1_cohorts> <args...>; with --array, {T} in an argument becomes the line of $C1_LIST number task+1.
source analysis/slurm/_body.sh
args=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${C1_LIST:-}" ]]; then
  v=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$C1_LIST")
  args=("${args[@]//\{T\}/$v}")
  # shellcheck disable=SC2206
  args=(${args[@]})
fi
mamba run --no-capture-output -n chimera_env python "analysis/tasks/C1_cohorts/${args[0]}" "${args[@]:1}"
