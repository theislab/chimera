#!/usr/bin/env bash
# H1, one pass, re-runnable. Run from chimera/.
set -euo pipefail
export PYTHONPATH=$PWD OMP_NUM_THREADS=2
R="mamba run --no-capture-output -n chimera_env python"
D=revision/tasks/H1_harmonise
$R $D/tables.py --jobs 8
$R $D/recurrence.py --jobs 16
$R $D/gsea.py --jobs 5
$R $D/archetypes.py --jobs 5
$R $D/drivers.py
$R $D/figures.py
$R $D/tex.py
