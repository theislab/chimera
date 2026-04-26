#!/bin/bash
# iter120 end-to-end eval pipeline. Idempotent — re-running is safe.
# Usage: bash training/_iter120_eval.sh
set -e

RUN=testis-hmac-iter120-pgbias-lr1e-3-200ep-s42
DIR=training/runs/${RUN}
ADATA=data/processed/camex_testis_human_macaque.h5ad

echo "=== iter120 eval: ${RUN} ==="

# 1. Confirm training finished
if [ ! -f "${DIR}/run_summary.json" ]; then
    echo "ERROR: ${DIR}/run_summary.json not found — training not complete?"
    exit 1
fi
echo "✓ Training summary exists"

# 2. compare_vs_scvi.py against scVI(donor=sample_batch) baseline
echo ""
echo "--- compare_vs_scvi (donor baseline) ---"
mamba run -n chimera_env python training/compare_vs_scvi.py \
    --chimera-run-name ${RUN} \
    --adata ${ADATA} \
    --batch-key sample_batch

# 3. gin_signed_diagnostics.py for per-gene per-class metrics
echo ""
echo "--- gin_signed_diagnostics ---"
mamba run -n chimera_env python training/gin_signed_diagnostics.py ${DIR}

# 4. UMAP aspect check
echo ""
echo "--- umap_check ---"
mamba run -n chimera_env python training/umap_check.py ${DIR} --adata ${ADATA}

# 5. Summary of results
echo ""
echo "=== iter120 RESULTS SUMMARY ==="
echo ""
echo "report.md (E1-E7+E9):"
grep -E "^\| E[0-9]" ${DIR}/report.md 2>/dev/null | head -20
echo ""
echo "integration_vs_scvi.md (E8):"
cat ${DIR}/integration_vs_scvi.md 2>/dev/null | grep -A1 -E "E8\.|gate" | head -30
echo ""
echo "gin_signed_diagnostics.json:"
cat ${DIR}/gin_signed_diagnostics.json 2>/dev/null
echo ""
echo "umap_aspect.json:"
cat ${DIR}/umap_aspect.json 2>/dev/null

echo ""
echo "=== DONE ==="
