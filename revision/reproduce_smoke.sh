#!/usr/bin/env bash
# Reproduction smoke test for a release checkout (see REPRODUCE.md).
#
# Usage, from the repository root, inside chimera_env:
#   bash revision/reproduce_smoke.sh            # tests + Δα tables + gene identity
#   RUN_CHECKPOINT=1 bash revision/reproduce_smoke.sh   # also load one checkpoint on CPU
#   RUN_FIGURE=1 bash revision/reproduce_smoke.sh       # also execute 03_alpha_master
#
# Expects the deposited bundles unpacked in place: data/processed/ and
# training/runs/ (see REPRODUCE.md). Writes nothing outside a temporary folder,
# except the figures that notebook 03 writes when RUN_FIGURE=1.
# Exit code 0 means every check passed.
set -euo pipefail

ROOT="${CHIMERA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export CHIMERA_ROOT="$ROOT" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg
PY="${PYTHON:-python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

step() { printf '\n== %s\n' "$*"; }

step "inputs present"
for f in data/processed/ortholog_table.csv data/processed/ortholog_table_testis.csv \
         configs/canonical_recipes.tsv; do
  [[ -f $f ]] || { echo "missing $f"; exit 2; }
done
# Fields 5, 10, 11 of the recipe table: h5ad, seeds, run_name (overrides may be empty).
awk -F'\t' 'NR > 1 {print $5, $10, $11}' configs/canonical_recipes.tsv | while read -r h5ad seeds run; do
  [[ -f data/processed/$h5ad ]] || { echo "missing data/processed/$h5ad"; exit 2; }
  for s in ${seeds//,/ }; do
    d="training/runs/${run//\{seed\}/$s}"
    [[ -f $d/final_state.npz && -f $d/model/model.pt ]] || { echo "missing $d"; exit 2; }
  done
done
echo ok

step "test suite"
"$PY" -m pytest -q -p no:cacheprovider tests

step "Δα tables from final_state.npz"
"$PY" - <<'PY'
import csv
import numpy as np
from scipy.stats import spearmanr
from revision.common import gene_ids

rows = list(csv.DictReader(open("configs/canonical_recipes.tsv"), delimiter="\t"))
recipe = {r["cohort"]: r for r in rows}

def final_state(cohort, seed=42):
    run = recipe[cohort]["run_name"].replace("{seed}", str(seed))
    return dict(np.load(f"training/runs/{run}/final_state.npz", allow_pickle=False))

fs = final_state("liv_hmonkey")
a, p = fs["alpha"], fs["prior"]
got = (len(a), round(float(a.mean()), 3), round(float(a.std()), 3), round(float(spearmanr(a, p)[0]), 3))
print("liver H+Monkey s42: n, mean α, std α, Spearman(α, π) =", got)
assert got == (8430, 0.784, 0.142, 0.770), got

fs = final_state("liv_4sp")
d = fs["alpha"] - fs["prior"]
up, down = round(100 * float((d > 0.05).mean()), 1), round(100 * float((d < -0.05).mean()), 1)
print(f"liver 4-sp s42: Δα > +0.05 = {up} %, Δα < -0.05 = {down} %")
assert (up, down) == (7.7, 35.8), (up, down)

import anndata as ad
for cohort, stem in gene_ids.H5ADS.items():
    adata = ad.read_h5ad(f"data/processed/{stem}.h5ad", backed="r")
    rho = gene_ids.assert_table_matches(adata, gene_ids.load_table(cohort))
    n_alpha = len(final_state(cohort)["alpha"])
    assert n_alpha == adata.n_vars, (cohort, n_alpha, adata.n_vars)
    print(f"{cohort:12s} genes={adata.n_vars:6d}  table match Spearman={rho:.3f}")
    adata.file.close()
PY

if [[ ${RUN_CHECKPOINT:-0} == 1 ]]; then
  step "checkpoint load on CPU (liver H+Monkey s42)"
  "$PY" - <<'PY'
import anndata as ad
import numpy as np
from scipy.stats import spearmanr
from chimera import ChimeraModel

run = "training/runs/iter90-150ep-s42"
adata = ad.read_h5ad("data/processed/camex_human_monkey.h5ad")
model = ChimeraModel.load(f"{run}/model", adata=adata, accelerator="cpu")
alpha = np.asarray(model.get_alpha(), dtype=float).ravel()
ref = np.load(f"{run}/final_state.npz", allow_pickle=False)["alpha"]
rho = float(spearmanr(alpha, ref)[0])
print(f"per-gene α Spearman vs final_state = {rho:.4f}")
assert rho >= 0.999, rho
PY
fi

if [[ ${RUN_FIGURE:-0} == 1 ]]; then
  step "render notebook 03_alpha_master"
  "$PY" -m papermill notebooks/03_alpha_master.ipynb "$TMP/03_alpha_master.ipynb" \
    --cwd notebooks -k "${KERNEL:-chimera_env}"
fi

step "all checks passed"
