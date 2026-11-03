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
# Grid: full factorial of alteration x effect x expression band x share x seed on all three cohorts (BENCHMARK_PLAN, second
# amendment). Only cells that add no information leave: a share whose gene set equals a smaller share's (make_grid: duplicate_of).
# Central setting: strongest effect of each alteration, band midhigh, share 5%, seed 42. Tiers set the submission order, so central
# settings and main effects finish before interactions:
#   0 central effects and nulls at the central setting
#   1 main effects, one factor at a time: effect, band, share, seed
#   2 effect x band and nulls x band, share 5%, seed 42
#   4 the remaining shares, seed 42
#   5 seed 43
# Tier 3 is the tools' main settings, tier 6 their other rows (make_queue.py).
# Scope: tiers 0, 1, 2, 3, 10, 11 run and scored; 4, 5, 6 planned, not run (owner decision, BENCHMARK_PLAN.md).
# ---------------------------------------------------------------------------
ALTERATIONS = {
    # q: share of the move from the observed profile to the swapped profile
    "pattern": [0.25, 0.5, 1.0],
    # f: fold; the altered species' gene is thinned to 1/f, which raises the other species' level f-fold
    "level": [2, 4, 8],
    # e: share of the gene's counts removed in its top type
    "loss": [0.5, 0.9, 1.0],
}
CENTRAL_EFFECT = {"pattern": 1.0, "level": 8, "loss": 1.0, "sham_swap": 1.0, "thin_both": 4}
NULLS = {"sham_swap": [1.0], "thin_both": [4]}
BANDS = ["low", "midlow", "midhigh", "high"]  # quartiles of species-balanced mean log1p CP10K, all genes
CENTRAL_BAND = "midhigh"
CENTRAL_SHARE = 0.05
SHARES = [0.02, 0.05, 0.10]
SEEDS = [42, 43]  # one seed drives gene draw, alteration and training of a run
IDENTITY_BANDS = ["id_low", "id_mid", "id_high"]  # tertiles of pi_raw over all genes; draws are balanced on them

# Selection
MIN_COUNTS = 20        # gene needs >= 20 counts in each species, otherwise an alteration is empty
MIN_CONTRAST_LOG2 = 1  # pattern, loss, and their controls: top vs bottom shared type >= 2-fold in the altered species
MIN_TYPE_CELLS = 30    # shared types: >= 30 cells in both species (the T4.2 rule)
MAX_PAIRS_FRAC = 0.5   # altered + control cannot exceed the eligible genes of the band

# Tools. Main settings, tier 3: every alteration type (central effects) and the sham null, bands midhigh and low, share 5%, seed 42, plus the unaltered object.
# Tier 6: also level 4, sham_swap, all bands, seed 43.
TOOL_SHARE = 0.05
TOOL_ROWS = [("pattern", 1.0), ("level", 4), ("level", 8), ("loss", 1.0), ("sham_swap", 1.0)]
TOOL_MAIN_ROWS = [("pattern", 1.0), ("level", 8), ("loss", 1.0), ("sham_swap", 1.0)]
TOOL_MAIN_BANDS = ["midhigh", "low"]
TOOL_NONE_SEEDS = [42, 43]

# Health (discard rule, fixed now): a run is discarded if alpha, z or S_g is non-finite or fewer than 10 of 10
# latent dims have Var(z_mean) >= 0.01. Mixing, tails and saturation are recorded for every run, never used
# to discard, and read in a sensitivity (well-mixed runs only).
ACTIVE_VAR = 0.01
N_LATENT = 10


# Gate arms (coordinator request after the interim, predictions in PREDICTIONS.md before scoring). Same altered objects as the main grid, recipe otherwise
# unchanged. free: Config M, no GIN, one free c_g per gene. hybrid: GIN plus a per-gene bias (iter119). Tiers: 10 central alterations and nulls x 2 seeds with
# unaltered fits, 11 the tier 1 main effects.
ARMS = {"free": lambda r: r.replace("--config A", "--config M"), "hybrid": lambda r: r + " --use-per-gene-bias"}
ARM_METHODS = {"S_g": "S_g", "Delta_alpha": "D", "Delta_alpha_raw": "D_raw", "c_g": "correction", "B1z": "B1z", "S_fwd": "S_fwd"}
ARM_SEEDS = [42, 43]


def useed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def run_name(cohort, alt, effect, band, share, seed) -> str:
    return f"b1-{cohort}-{alt}{effect}-{band}-sh{int(round(share * 100)):02d}-s{seed}"


def tier_of(alt, effect, band, share, seed) -> int:
    central_e = effect == CENTRAL_EFFECT[alt]
    central_b, central_s, central_seed = band == CENTRAL_BAND, share == CENTRAL_SHARE, seed == 42
    n_off = (not central_e) + (not central_b) + (not central_s) + (not central_seed)
    if n_off == 0:
        return 0
    if n_off == 1:
        return 1
    if central_s and central_seed:
        return 2
    return 4 if central_seed else 5


def grid(cohort: str | None = None) -> list[dict]:
    rows = []
    for c in ([cohort] if cohort else COHORTS):
        fam = {**ALTERATIONS, **NULLS}
        for alt, effs in fam.items():
            for e, b, sh, s in itertools.product(effs, BANDS, SHARES, SEEDS):
                rows.append(dict(cohort=c, alt=alt, effect=e, band=b, share=sh, seed=s, tier=tier_of(alt, e, b, sh, s),
                                 run=run_name(c, alt, e, b, sh, s), reuse=""))
        for s, r in COHORTS[c]["none_runs"].items():
            rows.append(dict(cohort=c, alt="none", effect=0, band="all", share=0.0, seed=s, tier=0, run=r, reuse=r))
    return rows


def arm_run(arm: str, main_run: str) -> str:
    return main_run.replace("b1-", f"b1{arm}-", 1)


def arm_grid() -> list[dict]:
    rows = []
    main = grid()
    for arm, rec in ARMS.items():
        for c, cfg in COHORTS.items():
            for s in ARM_SEEDS:
                rows.append(dict(cohort=c, alt="none", effect=0, band="all", share=0.0, seed=s, tier=10, arm=arm, run=f"b1{arm}-{c}-none-s{s}",
                                 reuse="", recipe=rec(cfg["recipe"])))
        for r in main:
            if r["alt"] == "none" or r["tier"] > 1:
                continue
            c0 = r["effect"] == CENTRAL_EFFECT[r["alt"]] and r["band"] == CENTRAL_BAND and r["share"] == CENTRAL_SHARE
            rows.append({**r, "tier": 10 if c0 else 11, "arm": arm, "run": arm_run(arm, r["run"]), "recipe": rec(COHORTS[r["cohort"]]["recipe"])})
    return rows

RUN_TIERS = {0, 1, 2, 3, 10, 11}
