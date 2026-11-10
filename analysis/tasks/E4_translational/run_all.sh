#!/usr/bin/env bash
# E4 pipeline, from chimera/: bash analysis/tasks/E4_translational/run_all.sh   (about 25 min on 16 local cores; no SLURM)
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
R="mamba run -n chimera_env python -m analysis.tasks.E4_translational"
bash analysis/tasks/E4_translational/fetch.sh
$R.annot
$R.q1_q3_stats tables q1 q23
$R.names
$R.summarize
$R.q1_followup
$R.figures
bash figures/build.sh    # composes the figures whose panels changed (figures/specs/supp_tissues.py)
