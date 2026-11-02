#!/usr/bin/env bash
# Local pass over cohorts, skipping finished latents. Re-run after the scVI array ends.
export PYTHONPATH=$PWD OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 PYTHONUNBUFFERED=1
for c in "$@"; do
  mamba run --no-capture-output -n chimera_env python -m revision.tasks.Z2_embedding.measure cohort "$c" 2>&1 | grep -v Warning
done
