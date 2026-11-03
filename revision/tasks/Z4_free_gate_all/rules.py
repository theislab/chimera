"""Z4 free gate on every harmonised cohort: cohorts, recipes, seeds, health rule and readout definitions.

Exploratory, beside the canonical reading; nothing here changes a status, row or modifier. Rule text in the private ledger
Z4_free_gate_all.md, committed before any free-gate fit of a new cohort was read.

Free gate = Config M (`--config M`: no GIN, a free per-gene c_g, package unchanged), exactly as Z3 launched it: the cohort's chosen
canonical recipe (the argv of its kept H1 runs) with `--config A` replaced by `--config M`, nothing else changed. Outputs are read
from final_state.npz (alpha, prior, S_g_ema, z_mean), never through ChimeraModel.load (which does not restore per_gene_correction).

Cohorts: the 26 of H1 (rules.cohorts()). Liver H+Mouse, testis H+Mouse and liver 4 species reuse Z3's free-gate fits (read only);
the other 23 are trained here, seeds 42 and 43; a third seed (44) only where one of the two fails health.
"""

from __future__ import annotations

from revision.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "revision"
OUT = RES / "Z4"
TASK = REPO / "revision" / "tasks" / "Z4_free_gate_all"
RUNS = REPO / "training" / "runs"

SEEDS = (42, 43)
SPARE = 44
PREFIX = "z4"

# Z3's free-gate fits, read only: candidates in seed order, first ones passing the health rule below are used (Z3 lines.json free5 picks
# the same ones under the same rule).
Z3_FREE = {
    "liv_hmouse": {42: "rev-ctl-liv_hmouse-m-s42", 45: "rev-ctl-liv_hmouse-m-s45", 46: "rev-ctl-liv_hmouse-m-s46",
                   **{s: f"z3-liv_hmouse-m-s{s}" for s in (47, 48, 49)}},
    "test_hmouse": {s: f"z3-test_hmouse-m-s{s}" for s in (42, 43, 44, 45, 46)},
    "liv_4sp": {s: f"z3-liv_4sp-m-s{s}" for s in (42, 43, 44, 45, 46, 47, 48, 49)},
}
Z3_MAX_SEEDS = 5  # Z3's lines take the first 5 healthy seeds

# Recipes of cohorts whose canonical runs predate run provenance (configs/canonical_recipes.tsv, checked against their config.json):
# overrides after --adata. Every other cohort takes the argv of its first kept H1 run from _provenance.json.
NO_PROVENANCE = {
    "liv_hmonkey": "--max-epochs 150 --adata data/processed/camex_human_monkey.h5ad",
    "liv_hzeb": "--max-epochs 100 --adata data/processed/camex_human_zebrafish.h5ad --lambda-mmd 10",
    "test_hmac": "--max-epochs 200 --adata data/processed/camex_testis_human_macaque.h5ad --adv-weight 100 --free-bits 1.0 --lambda-mmd 50 "
                 "--celltype-conditional-mmd --celltype-key cell_type --prior-clamp-max 0.85",
    "test_3sp": "--max-epochs 100 --adata data/processed/camex_testis_unified.h5ad --adv-weight 100 --lambda-anchor 0.001 --free-bits 1.0 "
                "--lambda-mmd 50 --celltype-conditional-mmd --celltype-key cell_ontology_class --prior-clamp-max 0.85",
}

# Health rule for free-gate seeds (Z3 amendment 1, unchanged): finite alpha, prior, z and S_g_ema, no E2.1; 10 of 10 z dimensions with
# Var >= 0.01; alpha > 0.95 and alpha < 0.05 each in at most 10% of genes (seeds above 5% have their saturated genes flagged, as H1);
# the gate follows its own count evidence, Spearman(alpha - prior, S_g_ema) >= 0.5 on count-supported genes. The raw-prior tail
# criterion of the canonical rule is reported, not applied: it tests that the GIN left the prior; a free gate's spread is |S_g|.
# Integration enters health: kNN mixing entropy at least half the median of the cohort's kept canonical seeds and adversary accuracy at
# most the canonical maximum + 0.10 (H1 integration cache, same function, k = 15). A seed failing this is 'poorly mixed', reported, and
# does not count toward the two seeds; Z3's liver H+Mouse line (3 integrated, 2 poorly mixed) is taken as Z3 fixed it.
SAT_MAX, SAT_FLAG, ZVAR_MIN, N_ACTIVE, RHO_MIN = 0.10, 0.05, 0.01, 10, 0.5
TAIL_DELTA = 0.05
MIX_FRAC, ADV_MARGIN = 0.5, 0.10
EPS = 1e-6  # count support |S_ema| >= EPS (X4, Z1, H1)
