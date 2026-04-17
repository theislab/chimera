#!/bin/bash
# GIN-input ablation sweep (round2 §1526 & Issue 15.3).
# Four 100-epoch runs on H+Mo subset, all with iter16 config:
#   1. gin-abl-baseline:    reference (all 8 informative features + 64 ESM-2 PCA)
#   2. gin-abl-no-esm2:     zero the 64 ESM-2 PCA dims (scalar features only)
#   3. gin-abl-no-expr:     zero the 3 expression stats (tests Problem 1 fix)
#   4. gin-abl-no-scalars:  zero all 5 scalar features (ESM-2 only)
# Runtime: ~5 min/run on MPS = ~20 min total.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=/Users/kemalinecik/tools/apps/mamba/bin/python
COMMON_ARGS="--max-epochs 100 --adv-lr 1e-4 --seed 42 --config D --adv-weight 100 --lambda-fit 0.1 --lambda-anchor 0.001 --lambda-res 0.01 --warmup-adversarial-epochs 5 --warmup-fqr-epochs 20 --warmup-kl-epochs 20 --n-latent 10 --n-hidden 128 --accelerator mps"

echo "=== [1/4] gin-abl-baseline (no patch) ==="
$PY training/run_experiment.py --run-name gin-abl-baseline $COMMON_ARGS 2>&1 | tee training/runs/gin-abl-baseline.log | tail -5

echo "=== [2/4] gin-abl-no-esm2 ==="
$PY training/run_experiment.py --run-name gin-abl-no-esm2 $COMMON_ARGS --gin-feature-op zero_esm2 2>&1 | tee training/runs/gin-abl-no-esm2.log | tail -5

echo "=== [3/4] gin-abl-no-expr ==="
$PY training/run_experiment.py --run-name gin-abl-no-expr $COMMON_ARGS --gin-feature-op zero_expr_stats 2>&1 | tee training/runs/gin-abl-no-expr.log | tail -5

echo "=== [4/4] gin-abl-no-scalars ==="
$PY training/run_experiment.py --run-name gin-abl-no-scalars $COMMON_ARGS --gin-feature-op zero_all_scalars 2>&1 | tee training/runs/gin-abl-no-scalars.log | tail -5

echo
echo "=== GIN input sweep complete ==="
ls training/runs/gin-abl-*/summary.json
