#!/usr/bin/env bash
# Cross-dataset robustness sweep — 2026-04-17
#
# Run the canonical 2-species iter16 recipe on every H+<other> subset
# prepared under data/processed/. The recipe is *unchanged* from
# workflow.md §"Canonical recipes" — the sweep asks whether that recipe
# is robust to the choice of second species or whether it over-fits to
# the Human+Monkey pair.
#
# Sequential execution (MPS is a single device). Each run takes ~7 min.
# Results are under training/runs/xds-*.

set -uo pipefail
# NB: no `-e`. `run_experiment.py` exits 1 when any expectation fails, which
# is an *expected* outcome of a cross-dataset sweep (we want to see which
# datasets the default recipe generalises to, not abort on the first one
# that doesn't). Each run's summary.json is authoritative.

cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"   # run inside chimera_env, or set PYTHON

CONFIG_FLAGS=(
  --config D --max-epochs 100
  --adv-weight 100 --adv-lr 1e-4
  --lambda-fit 0.1 --lambda-anchor 0.001 --lambda-res 0.01
  --warmup-adversarial-epochs 5 --warmup-fqr-epochs 20 --warmup-kl-epochs 20
  --n-latent 10 --n-hidden 128 --accelerator mps --seed 42
)

run_one() {
  local name="$1"
  local adata="$2"
  echo "=== $(date '+%H:%M:%S')  $name  ==="
  "$PY" training/run_experiment.py \
    --run-name "$name" \
    --adata "$adata" \
    "${CONFIG_FLAGS[@]}"
}

run_one xds-hmouse-iter16     data/processed/camex_human_mouse.h5ad
run_one xds-hzebrafish-iter16 data/processed/camex_human_zebrafish.h5ad

echo
echo "=== sweep complete $(date '+%H:%M:%S') ==="
"$PY" training/summarize_runs.py \
  iter16-slow-classifier xds-hmouse-iter16 xds-hzebrafish-iter16 \
  > training/runs/xds_sweep_summary.md
echo "wrote training/runs/xds_sweep_summary.md"
