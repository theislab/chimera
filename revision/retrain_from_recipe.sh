#!/usr/bin/env bash
# Train one canonical recipe again from configs/canonical_recipes.tsv (see REPRODUCE.md).
#
# Usage, from the repository root, inside the environment:
#   bash revision/retrain_from_recipe.sh <cohort> <seed> [run_name] [accelerator]
#   bash revision/retrain_from_recipe.sh liv_hmouse 42 retrain-liv_hmouse-s42 cpu
#
# <cohort> is a key of the recipe table (liv_hmonkey, liv_hmouse, liv_hzeb, liv_4sp, test_hmac,
# test_hmouse, test_3sp). The run goes to training/runs/<run_name>/ (default
# retrain-<cohort>-s<seed>) and is never written over. Accelerator defaults to cpu. Set
# PRINT_ONLY=1 to print the command without running it.
set -euo pipefail

cohort="${1:?cohort}"; seed="${2:?seed}"
run="${3:-retrain-$cohort-s$seed}"; acc="${4:-cpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

field() { awk -F'\t' -v c="$cohort" -v f="$1" 'NR > 1 && $1 == c {print $f}' configs/canonical_recipes.tsv; }
h5ad="$(field 5)"; epochs="$(field 8)"; overrides="$(field 9)"
[[ -n $h5ad ]] || { echo "unknown cohort $cohort"; exit 2; }
[[ -e training/runs/$run ]] && { echo "training/runs/$run exists; choose another run name"; exit 2; }

# shellcheck disable=SC2206
cmd=("${PYTHON:-python}" training/run_experiment.py --run-name "$run" --config A --max-epochs "$epochs"
     --adata "data/processed/$h5ad" --seed "$seed" --accelerator "$acc" $overrides)
echo "${cmd[*]}"
[[ ${PRINT_ONLY:-0} == 1 ]] || "${cmd[@]}"
