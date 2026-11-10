#!/usr/bin/env bash
#SBATCH --job-name=g1_seed
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --exclude=cpusrv114
#SBATCH --output=results/analysis/G1/logs/%x_%j.out
# G1, seed fix in training/train_and_diagnose.py set_seed(). usage: sbatch seedfix.sbatch <base tree> <fixed tree>. One node, one thread.
#   base, split pinned to the seed by the harness  vs  fixed, nothing pinned (twice): identical means the fix only seeds the split, and new fits repeat.
set -uo pipefail
command -v mamba >/dev/null 2>&1 || PATH="$(dirname "${MAMBA_EXE:-${CONDA_EXE:-/nonexistent/mamba}}"):$PATH"
command -v mamba >/dev/null 2>&1 || { echo "mamba not found: put its bin folder on PATH before submitting" >&2; exit 2; }
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=$1; fixed=$2; here=$PWD/analysis/tasks/G1_guard; data=$PWD/data/processed
hostname
fit() { local tree=$1 name=$2 pin=$3; shift 3
  G1_PIN_SPLIT_SEED=$pin mamba run --no-capture-output -n chimera_env python "$here/bitproof_run.py" "$tree" --run-name "$name" --accelerator cpu --max-epochs 3 --seed 42 "$@" 2>&1 | grep -E "Key metrics|Error|error" | tail -2; }
L="--config A --adv-weight 200 --free-bits 1.0 --adata $data/camex_human_mouse.h5ad"
T="--config A --adv-weight 200 --free-bits 1.0 --celltype-conditional-mmd --celltype-key cell_ontology_class --prior-clamp-max 0.85 --adata $data/camex_testis_human_mouse.h5ad"
for g in "liv $L" "test $T"; do set -- $g; n=g1seed-$1; shift
  fit "$base" "$n-a" 42 "$@"; fit "$fixed" "$n-a" none "$@"; fit "$fixed" "$n-b" none "$@"
done
mamba run --no-capture-output -n chimera_env python - "$base" "$fixed" <<'PY'
import sys, json
from pathlib import Path
sys.argv = ["x", sys.argv[1], sys.argv[2], "/dev/null"]
src = open("analysis/tasks/G1_guard/bitproof_compare.py").read().split("rows = []")[0]
exec(src)
import pandas as pd
rows = []
for g in ("g1seed-liv", "g1seed-test"):
    a, f1, f2 = load(before, g + "-a"), load(after, g + "-a"), load(after, g + "-b")
    for lab, x, y in (("fixed vs fixed (repeat, nothing pinned)", f1, f2), ("base, split pinned to --seed vs fixed", a, f1)):
        rows.append(dict(group=g, compare=lab, sha_x=digest(x[0])[:16], sha_y=digest(y[0])[:16], **same(x, y)))
d = pd.DataFrame(rows)
d["identical"] = (d.n_tensors == d.n_tensors_equal) & (d.n_final_arrays == d.n_final_arrays_equal) & d.state_keys_equal & d.rng_equal
d.to_csv("results/analysis/G1/seedfix_proof.csv", index=False)
print(d.to_string())
PY
