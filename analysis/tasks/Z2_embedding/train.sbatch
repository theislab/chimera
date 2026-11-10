#!/usr/bin/env bash
#SBATCH --job-name=z2_train
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=results/analysis/Z2/logs/%x_%A_%a.out
# Throttle to about a sixth of the per-user CPU cap: 8 tasks at 8 CPUs, with scVI at 2. Memory about 1.3 x observed peak (liver H+Mouse 1.4G); testis rows pass --mem on the command line. Usage: sbatch --array=<rows>%8
source analysis/slurm/_body.sh
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" analysis/tasks/Z2_embedding/factorial.tsv)
name=$(cut -f1 <<<"$row"); mode=$(cut -f5 <<<"$row"); args=$(cut -f6- <<<"$row")
if [[ -f "training/runs/$name/summary.json" ]]; then echo "finished: $name"; exit 0; fi
if [[ -e "training/runs/$name" ]]; then mkdir -p training/runs/_failed; mv "training/runs/$name" "training/runs/_failed/${name}_${SLURM_JOB_ID:-x}"; fi
provenance "$name" "$mode" $args
rc=0
# shellcheck disable=SC2086
mamba run --no-capture-output -n chimera_env python -m analysis.tasks.Z2_embedding.train_patched "$mode" --run-name "$name" --accelerator cpu $args || rc=$?
[[ -f "training/runs/$name/summary.json" && -f "training/runs/$name/final_state.npz" ]] || exit "${rc:-1}"
