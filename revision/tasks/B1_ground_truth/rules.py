"""B1 Track A (semi-synthetic spike-in) and Track D: definitions, fixed before any run is scored.

Design in README.md, predictions in PREDICTIONS.md. Nothing in the chimera package changes; Chimera is
retrained with the cohort's canonical recipe. Alteration code in alter.py, scoring in score_run.py and
metrics.py, tidy aggregation in aggregate.py.
"""

from __future__ import annotations

import itertools
import zlib

from revision.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "revision" / "B1"
H5AD_DIR = REPO / "data" / "revision" / "B1" / "h5ad"
TOOL_DIR = REPO / "data" / "revision" / "B1" / "tools"
TASK_DIR = REPO / "revision" / "tasks" / "B1_ground_truth"

# ---------------------------------------------------------------------------
# Cohorts and recipes (unchanged canonical recipes: 07_EXPERIMENT_MATRIX, T3.2, P13)
# ---------------------------------------------------------------------------
COHORTS = {
    "liv_hmouse": dict(
        h5ad="data/processed/camex_human_mouse.h5ad", gene_table="liv_hmouse", label_key="cell_type",
        library_key="batch", coarse=False,
        recipe="--config A --max-epochs 150 --adv-weight 200 --free-bits 1.0",
        none_runs={45: "rev-liv_hmouse-s45", 46: "rev-liv_hmouse-s46", 47: "rev-liv_hmouse-s47",
                   48: "rev-liv_hmouse-s48", 49: "rev-liv_hmouse-s49", 50: "rev-liv_hmouse-s50"}),
    "test_hmouse": dict(
        h5ad="data/processed/camex_testis_human_mouse.h5ad", gene_table="test_hmouse", label_key="cell_type",
        library_key="sample_batch", coarse=True,
        recipe="--config A --max-epochs 200 --adv-weight 200 --free-bits 1.0 --celltype-conditional-mmd "
               "--celltype-key cell_ontology_class --prior-clamp-max 0.85",
        none_runs={45: "rev-test_hmouse-s45", 46: "rev-test_hmouse-s46", 47: "rev-test_hmouse-s47",
                   48: "rev-test_hmouse-s48", 49: "rev-test_hmouse-s49", 50: "rev-test_hmouse-s50"}),
    "kidney": dict(
        h5ad="data/revision/disease_kidney/kidney_healthy_train.h5ad", gene_table="kidney",
        label_key="cell_ontology_class", library_key="sample_batch", coarse=False,
        recipe="--config A --max-epochs 200 --batch-key batch --adv-weight 300 --lambda-anchor 0.003 "
               "--lambda-mmd 50 --free-bits 1.0 --celltype-conditional-mmd --celltype-key cell_ontology_class "
               "--prior-clamp-max 0.85",
        none_runs={42: "rev-kid-grid3-s42", 43: "rev-kid-s43", 44: "rev-kid-s44", 45: "rev-kid-s45",
                   46: "rev-kid-s46"}),
}
SPECIES = ("Human", "Mouse")

# ---------------------------------------------------------------------------
# Grid: full factorial of alteration x effect x expression band x share x seed, on all three cohorts
# ---------------------------------------------------------------------------
ALTERATIONS = {
    # q: share of the move from the observed profile to the swapped profile
    "pattern": [0.25, 0.5, 1.0],
    # f: fold; the altered species' gene is thinned to 1/f, which raises the other species' level f-fold
    "level": [2, 4, 8],
    # e: share of the gene's counts removed in its top type
    "loss": [0.5, 0.9, 1.0],
}
NULLS = {"sham_swap": [1.0], "thin_both": [4]}
BANDS = ["low", "midlow", "midhigh", "high"]  # quartiles of species-balanced mean log1p CP10K, all genes
SHARES = [0.02, 0.05, 0.10]
SEEDS = [42, 43]  # one seed drives gene draw, alteration and training of a run
IDENTITY_BANDS = ["id_low", "id_mid", "id_high"]  # tertiles of pi_raw over all genes; draws are balanced on them

# Selection
MIN_COUNTS = 20        # gene needs >= 20 counts in each species, otherwise an alteration is empty
MIN_CONTRAST_LOG2 = 1  # pattern, loss, and their controls: top vs bottom shared type >= 2-fold in the altered species
MIN_TYPE_CELLS = 30    # shared types: >= 30 cells in both species (the T4.2 rule)
MAX_PAIRS_FRAC = 0.5   # altered + control cannot exceed the eligible genes of the band

# Main settings for the tools (SATURN, CAMEX): one share, strongest informative effects, all bands, both seeds
TOOL_SHARE = 0.05
TOOL_ROWS = [("pattern", 1.0), ("level", 4), ("level", 8), ("loss", 1.0), ("sham_swap", 1.0)]
TOOL_NONE_SEEDS = [42, 43]

# Health (discard rule, fixed now): a run is discarded if alpha, z or S_g is non-finite or fewer than 10 of 10
# latent dims have Var(z_mean) >= 0.01. Mixing, tails and saturation are recorded for every run, never used
# to discard, and read in a sensitivity (well-mixed runs only).
ACTIVE_VAR = 0.01
N_LATENT = 10


def useed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def run_name(cohort, alt, effect, band, share, seed) -> str:
    return f"b1-{cohort}-{alt}{effect}-{band}-sh{int(round(share * 100)):02d}-s{seed}"


def grid(cohort: str | None = None) -> list[dict]:
    rows = []
    for c in ([cohort] if cohort else COHORTS):
        fam = {**ALTERATIONS, **NULLS}
        for alt, effs in fam.items():
            for e, b, sh, s in itertools.product(effs, BANDS, SHARES, SEEDS):
                rows.append(dict(cohort=c, alt=alt, effect=e, band=b, share=sh, seed=s,
                                 run=run_name(c, alt, e, b, sh, s), reuse=""))
        for s, r in COHORTS[c]["none_runs"].items():
            rows.append(dict(cohort=c, alt="none", effect=0, band="all", share=0.0, seed=s, run=r, reuse=r))
    return rows
