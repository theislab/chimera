#!/usr/bin/env bash
# Official runs of T2.3-T2.7 in chimera_env, on the current node (CPU).
# Outputs: results/analysis/T2_{3,4,5,6,7}/, each with _provenance.json and DONE last.
# Optional: REF_TABLES=<folder with the earlier per-cohort CSVs> adds the full-precision
# reproduction check to T2.4. THREADS sets the torch threads of T2.5 (default 4).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ENV="${ENV_NAME:-chimera_env}"
py() { mamba run --no-capture-output -n "$ENV" python "$@"; }

if [[ -n "$(git status --porcelain -- analysis/common analysis/tasks/T2_3_tissue analysis/tasks/T2_4_covariates \
      analysis/tasks/T2_5_celltype_margin analysis/tasks/T2_6_matched analysis/tasks/T2_7_consistency)" ]]; then
  echo "uncommitted changes in the T2 scripts; commit first so _provenance.json records the code" >&2
  exit 1
fi

py analysis/tasks/T2_3_tissue/run.py
py analysis/tasks/T2_4_covariates/run.py ${REF_TABLES:+--reference-tables "$REF_TABLES"}
py analysis/tasks/T2_6_matched/run.py
py analysis/tasks/T2_7_consistency/run.py
py analysis/tasks/T2_5_celltype_margin/run.py --threads "${THREADS:-4}"
