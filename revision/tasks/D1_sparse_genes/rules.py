"""D1 (EXPLORATORY): does Chimera return a usable reading on genes with too few counts for a direct method?

Everything here is fixed and committed before any association between a readout and a gold on these gene sets is computed for this task.
Nothing changes a status, row or modifier of the plan, and nothing changes in the chimera package. Vocabulary: CHIMERA_MECHANISM.md: a masked gene's
delta-alpha is imputed from features, never a count call.

Gene sets (per cohort; flags from H1 tables, O1 audit).
  masked     |S_ema| < 1e-6 (S exactly 0 in all seeds, X4). ppass in the table gives the chance of passing the mask in one batch.
  clean      not flagged T3 (o1_T2, wrong row, false zero). PRIMARY universe: a false zero is an artefact of the join, not sparsity. All genes are a sensitivity.
  no_direct  masked and B1 undefined (no pair of species has a shared type with the T4.2 detection rule): no direct pattern reading.
  sp_sparse  masked and under 20 expressing cells in at least one species of the cohort.
Split. Human chromosome, Ensembl 110, as Y4: discovery odd autosomes, replication even autosomes. X, Y, MT, scaffolds in neither; they stay in the "all" row.

Golds (rules_c of B2 Track C): liver G_H3 (TS vs TMS), G_H1 (EVEE), G_lvl_liver; testis G_H4, G_H4_primates, G_H2; kidney G_kidney(+fine, nephron), G_evee_kidney, G_lvl_kidney;
islet G_pancreas (+noacinar, fine), G_lvl_pancreas; bone marrow G_coe. Eye cohorts have no gold: gold-free rows only.
Statistic. rho_p of 05 section 3 (rank, residualise on spline C or C_pat, Pearson), ESM-block bootstrap; pi on C without pi.

Methods on masked genes (oriented higher = more conserved).
  Chimera   D (consensus), S_fwd (forward margin at the posterior mean, no mask; a count reading, exploratory, B1 Track A), S (0 by construction: coverage 0).
  Sequence  pi.
  Direct    B1 (coverage partial), B1c, B1z, Sdir, Lfc: counts on the same training cohort.
  Feature   B3a (frozen: 72 features to E), B3b (frozen: 72 features to D, cross-fitted, so an emulator of Chimera's own output);
            trained on unmasked genes only, predicted on masked genes, never seen: R_S (target S), R_Sdir, R_B1, R_D; suffix _p = protein columns only (69: all but 66-68), _72 = all.
            kNN_S, kNN_Sdir, kNN_B1, kNN_D: mean over the KNN nearest unmasked genes in the 64 ESM PCs (gold-free nearest-protein-neighbour control);
            Block_S: mean S of the unmasked genes of the gene's ESM module; Para_S: S of the closest human paralog (Compara 110) that is unmasked in the cohort.
Coverage: every row has n; denominators are written in coverage.csv.

Depth (b). Binomial thinning of all counts, B2 Track B code and recipe. Settings depth25 (B2 fits where they exist), depth10, depth5. Cohorts: liv_hmouse, test_hmouse (B2) and liv_4sp.
  sparse_f   unmasked at full depth, S exactly 0 in every replicate at fraction f, clean.
  Replicates: per-gene mean over healthy replicates, as B2.
  Metric 1, own agreement: rho of M_f with M_full given C, and raw, on genes both define, with share of the set defined.
  Metric 2, common target: rho of M_f with S_full and with Sdir_full given C (the count reading at full depth, measured on these genes).
  Metric 3, gold: rho_p(M_f, gold | C), covariates unperturbed.
  Depth comparators: D, S_fwd, B1, B1c, Sdir, Lfc, B3a, and R_S_f, R_Sdir_f, kNN_S_f built inside each depth from its unmasked genes.
  Ceiling for D: unaltered seed against the consensus of the others, on the same genes.

Held-out genes (c). A fold is a set of whole chromosomes holding about HOLD_SHARE of genes: fold A from odd autosomes (discovery), fold B from even autosomes (replication), so
  no tandem neighbour is in training. Counts of held-out genes are set to zero in the object; S_g is then 0 by the package's own mask, exactly as a masked gene; p_null recomputed; recipe unchanged.
  Arm P: --gin-feature-op zero_expr_stats (the gate reads protein columns only, for every gene): the honest case.
  Arm E: 72 features, held-out genes keep their true expression columns (diagnostic: what expression columns add; not available for an unmeasured gene). From each arm-E fit also
  D_E0: same fitted gate, forward on held-out genes with expression columns at their mean (0).
  Scored on held-out genes, clean, against measured D, S, Sdir, B1 (full object) and golds, beside R_*_p, kNN_* trained on the fit's other genes only, and pi. Ceiling as above.
  Paralog sensitivity: drop held-out genes with a paralog outside the fold at >= PARALOG_ID identity.

Decision rules.
  signal(cell)    cohort x gold, clean masked genes: rho_p >= 0.03 with CI lower > 0 on odd, and same sign with CI excluding 0 on even.
  feature-matched D is within 0.03 of, or below, any of R_S_p, R_Sdir_p, R_B1_p, B3a, kNN_S, kNN_Sdir, Block_S (delta lower CI not above 0, or delta < 0.03), on odd or even. B3b is scored and is not a competitor.
  target          target = Chimera if R_S_p beats R_Sdir_p and R_B1_p by >= 0.03 with CI lower > 0 on odd and on even. Otherwise the target is any count reading.
  Verdict         differentiates Chimera: a signal cell with D beating every comparator of "feature-matched" and every direct score defined there, and on depth or held-out the same.
                  matches feature-only models: a signal cell, feature-matched. not usable: no signal cell in masked genes and none in held-out genes.
"""

from __future__ import annotations

import zlib

from revision.common import gene_ids
from revision.tasks.B2_benchmark import rules as B2
from revision.tasks.B2_benchmark import rules_c as RC

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "revision" / "D1"
H5AD_DIR = REPO / "data" / "revision" / "D1" / "h5ad"
TASK = REPO / "revision" / "tasks" / "D1_sparse_genes"
SEED = 20261102

ALL_COHORTS = list(RC.COHORTS_C)
GOLD_COHORTS = [c for c, v in RC.COHORTS_C.items() if v["golds"]]
INFORMATIVE = ["liv_4sp", "test_hmouse", "test_hmac", "test_3sp"]   # mechanism file and coordinator: features carry masked D here
WEAK = ["liv_hmouse"]

COV_C = B2.COV_C
KNN = 15
MIN_EXPR_CELLS = 20
N_DRAWS = 10_000           # plan section 3; depth and held-out grids 2,000 as B2
N_DRAWS_GRID = 2_000
STRATUM_MIN = B2.STRATUM_MIN
SIG = 0.03
PROT_COLS = [i for i in range(72) if i not in (66, 67, 68)]
ESM_COLS = list(range(64))

DEPTH_COHORTS = ["liv_hmouse", "test_hmouse", "liv_4sp"]
DEPTHS = {"depth25": 0.25, "depth10": 0.10, "depth5": 0.05}
HOLD_SHARE = 0.20
PARALOG_ID = 40.0
HOLD_SEEDS = {0: 242, 1: 243, 2: 244, 3: 245}  # 2, 3: 4-sp fold A only, both first seeds failed the health rule
ARMS = {"P": "--gin-feature-op zero_expr_stats", "E": "", "L": ""}  # L: anchor appended by anchor_arg.py, from revision/common/depth_anchor.py

COHORT_FIT = {
    "liv_hmouse": dict(cpus=8, mem="6G"),
    "test_hmouse": dict(cpus=12, mem="10G"),
    "liv_4sp": dict(cpus=12, mem="10G"),
}
LIV4 = dict(h5ad="data/processed/camex_pancreas_unified.h5ad", gene_table="liv_4sp", label_key="cell_type", coarse_map=False,
            library_key="batch", donor_key=None, batch_key="batch", model_key=None, cpus=12, mem="10G",
            recipe="--config A --max-epochs 200 --adv-weight 200 --lambda-anchor 0.001 --lambda-mmd 100 --free-bits 1.0",
            nolabel_recipe=None, ref_runs={})


def register() -> None:
    """Depth settings and 4-sp join the B2 registries of this process. B2 files stay as they are; the depth kind and its code are B2's."""
    B2.COHORTS.setdefault("liv_4sp", LIV4)
    B2.ALL_COHORTS.setdefault("liv_4sp", LIV4)
    B2.SETTINGS.setdefault("depth10", dict(family="sampling", kind="depth", frac=0.10))
    B2.SETTINGS.setdefault("depth5", dict(family="sampling", kind="depth", frac=0.05))


def useed(*parts) -> int:
    return zlib.crc32("|".join(map(str, ("D1",) + parts)).encode()) & 0x7FFFFFFF


# Committed test L (coordinator, 2026-11-02, 21:25), before any label-free cluster margin exists. Chosen after S_fwd was read on masked genes, so its comparators are new and its reading of S_fwd is not blind.
# Question: is S_fwd's label-free reading of sparse genes specific to Chimera, or does any latent with clusters give it?
# Sdir_scvi: X6 Sdir with Leiden clusters (15-NN, resolution 1, T4.2 code) of scVI(batch=species, 100 epochs, 10 latent, 128 hidden, training/compare_vs_scvi.py) in place of labels, mean over seeds 42-44 at full
# depth (S_fwd is a mean over 3 fits), one seed per replicate at depth 25%. Sdir_pca: same on 30 PCs of log1p CP10K, 2,000 HVGs by species, genes centred within species, deterministic.
# Same cells, same counts. Cohorts liv_hmouse, liv_4sp, test_hmouse, kidney. Sets: masked clean genes; sparse_f at depth25. Golds in L_GOLDS, primary = first of each cohort.
# Statistic rho_p given C_pat with ref S_fwd, 10,000 ESM-block draws, odd then even. Labelled Sdir and Lfc beside.
# leads: S_fwd - max(Sdir_scvi, Sdir_pca) >= 0.03 with CI lower > 0 on odd and on even. trails: the reverse. Otherwise ties.
# The candidate holds only if S_fwd leads on >= 2 of 4 cohorts on primary golds, masked genes, and trails on none; depth 25% read the same way where a healthy fit exists. A depth cell with no finished fit is "Chimera did not finish".
L_COHORTS = ["liv_hmouse", "liv_4sp", "test_hmouse", "kidney"]
L_GOLDS = {"liv_hmouse": ["pattern", "level_atlas"], "liv_4sp": ["pattern", "level_atlas"], "test_hmouse": ["pattern", "pattern_primates"], "kidney": ["pattern", "pattern_fine", "level_atlas"]}
L_SEEDS = [42, 43, 44]
L_HVG = 2000
L_PCS = 30
