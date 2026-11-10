"""Y1 (EXPLORATORY, Family 3): gene x tissue view of the count margin. Definitions, fixed before any association.

Nothing here changes a status, row or modifier of the plan. Every test below is chosen after the scoreboard. Predictions and
the verdict rule: chimera-private ledger Y1_gene_tissue.md.

Tissues and readouts. Human and mouse in common in every tissue; gene identity is the human Ensembl ID through
analysis.common.gene_ids (never a symbol, never ortho_N). Models stand as they are (no retuning).
  liver        liv_hmouse     pair fit, 10 frozen seeds (T3.3)              S = mean S_g_ema
  testis       test_hmouse    pair fit, 10 frozen seeds (T3.3)              S = mean S_g_ema
  kidney       kidney         pair fit, 5 healthy seeds (P13)               S = mean S_g_ema
  islet        islet          pair fit, 5 healthy seeds (T7.9)              S = mean S_g_ema
  eye          van_zyl hmouse pair fit, 2 healthy seeds s42, s43 (T11)      S = mean S_g_ema
  bone_marrow  human+lemur+mouse fit, 2 healthy seeds s42, s46 (T12.3)      S = ppass x mean over the human and mouse
                                                                            rows of the T2.5 forward margin, mean over seeds
D = alpha - prior as anchored, mean over the same seeds, centred on the cohort median (the fit's one D; on bone marrow the
fit is three-species). D_raw beside it: alpha - pi_raw, centred; on clamped recipes it nearly equals -pi on the raw scale, so D_raw
enters the variance split only with pi_raw in the design.
B1 = cross-species pseudobulk profile correlation E (T4.2) of the human-mouse pair over the tissue's shared groups,
raw, residualised here. Bone marrow uses E_human_mouse, not the pooled three-pair value. K per tissue 4, 6, 13, 8, 9, 8.
Sdir, Sdir_pattern, Sdir_level = the X4 no-model direct margin and its exact split, on the same cells and groups as B1
(human and mouse cells only on bone marrow; X4 cache on liver and testis).

Covariates per tissue, computed on human and mouse cells: expr = mean of species mean log1p CP10K, det = mean of species
detection, tau = T4.3 tau over the human-mouse shared types, amp = SD over shared types of log2 CPM, mean of species,
dom = top minus mean of other shared types of log2 CPM, mean of species. prior_raw and log_cds_length are the gene's own.

Comparable: within a tissue, ranks. Not comparable: raw S across tissues (scale follows depth and counts; sign only after
within-tissue centring; zero depends on cohort), raw B1 across tissues (K differs). So every cross-tissue statistic uses
z_gt = within-tissue standardised residual of rank(score) on the spline design of the tissue's covariates. A tissue main effect
is zero by construction after this; "gene vs tissue" therefore reads as gene vs gene x tissue, split from noise by reliabilities.
The tissue share before centring is reported on raw values as a statement about scale only.

Count support. A cell (g, t) has count support if |S| >= MASK_EPS (not masked) and ppass >= PPASS_MIN (passes the 20-cell
rule in at least half of the 256-cell batches). This is the primary matrix: the T2.5 margin_support flag needs Delta-alpha
and truncates |S| at its median, which would select on the outcome and on expression. Strict support (|S| >= tissue median
of unmasked |S|, sign of S stable over the tissue's seeds) is the sensitivity for (b) and the rule for naming in (d).
"""

from __future__ import annotations

from pathlib import Path

from analysis.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "analysis"
OUT = RES / "Y1"
ATLAS = REPO / "data" / "analysis" / "Y1"
TASK = REPO / "analysis" / "tasks" / "Y1_gene_tissue"

SEED = 20261102
MASK_EPS = 1e-6
PPASS_MIN = 0.5
N_DRAWS = 10_000
N_NULL = 1_000

TISSUES = ["liver", "testis", "kidney", "islet", "bone_marrow", "eye"]
# Gold -> tissue trained on the same organ. Independent atlas pair or held-out species; bone marrow (mouse side is TMS
# marrow and spleen, the training source) and eye (no mouse atlas in TMS) have none, so they enter only as other-tissue sources.
GOLD_TISSUE = {"liver": "liver", "kidney": "kidney", "pancreas": "islet", "testis": "testis"}
GOLD_COLUMN = {"liver": "G_H3", "kidney": "G_kidney", "pancreas": "G_pancreas", "testis": "G_H4"}

# Covariates of the target tissue T (columns of the tissue table, spline of rank, df 4, section 3).
C_EXPR_DET = ["expr", "det"]
C_PAT = ["prior_raw", "expr", "det", "log_cds_length", "tau"]  # + H4_nsp for the testis gold

# (c) contrast. rho(X_t, G_T | C) = rank partial correlation on one universe, one design for all sources t.
#   Delta_T(X) = rho(X_T, G_T) - mean_{t != T} rho(X_t, G_T)
#   Universe U_T: G_T and C of T defined, count support in all six tissues, B1 defined in all six (same genes for S and B1).
#   Family: Delta_T(S), Delta_T(B1), Delta_T(S) - Delta_T(B1) over the 4 golds = 12 tests, BH, p from the joint block draws
#   (N_DRAWS, ESM modules of the kidney vocabulary, genes outside it fall back to the testis modules).
#   Crossed interaction over the four matched tissues: mean diag - mean row - mean col + grand mean of rho(X_t, G_T), own
#   universes, same draws.
# Sensitivities, fixed: (i) C_PAT + every source tissue's expr, det, tau; (ii) matched variability: C_PAT + pattern amplitude
# in T, atlas amplitude, atlas expression and detection; (iii) (ii) + dominant-type enrichment in T; (iv) pairwise universes
# (support only in T and t); (v) unweighted margin (equal weight per shared type, composition); (vi) per-seed S_T (seed
# range), liver integrated seeds, kidney all five seeds; (vii) pancreas gold without acinar (K = 5, islet has no acinar).
# (d) classes on z_gt given expr, det, among genes with count support in >= MIN_TISSUES tissues:
Z_STABLE = 0.5
Z_FLIP = 1.0
MIN_TISSUES = 5
#   shared_all: z >= +Z_STABLE in every supported tissue; not_shared_all: z <= -Z_STABLE in every supported tissue;
#   flip: some tissue z >= +Z_FLIP and another z <= -Z_FLIP. Expected counts: z permuted within tissue inside 20 expression
#   bins (mean expr rank over tissues), N_NULL draws. Same classes for B1.
#   Named only with strict support and sign-stable S in the tissues read, S above/below the tissue median as the class says.

# (b) Variance split on z, one row per score and design. Designs: ED = expr, det (the question); ED_pi = + prior_raw;
# ED_tau = + tau (matched variability). Universes: all count-supported cells; complete cases in all six tissues (S and D); B1 defined.
#   gene share = mean over tissue pairs of the Pearson of z; rel_t = cell-half reliability on the fixed model (S; 1 forward pass per
#   tissue, label x half groups), cell-half (B1, N_HALVES splits, unit minimum halved, T4.3 baseline_splits rule), seed split (D, S beside);
#   noise share = 1 - mean rel_t; gene x tissue share = mean rel_t - gene share. All on the complete-case universe; block CI on the
#   gene share. ESM-free part of D: residual of z(D) on a 5-fold ridge on the 64 PCs, as X3, gene share again.
N_HALVES = 20
# (c2) Tissue contrast within gene: for two golds T, T', rho of (z S_T - z S_T') with (z G_T - z G_T'), z on each tissue's C_PAT
#   (+ H4_nsp on testis), genes with both golds and support in both tissues, 6 pairs, for S, B1, D, Sdir; BH over each score's six.
#   Flip contrast: among genes with z S_T - z S_T' beyond +/- 1 SD of that difference, mean(z G_T - z G_T') high minus low, SD units.
# (d) Direct analogue of flips: over cells (g, t_hi, t_lo) of flip genes, shares of (z S hi - z S lo) by covariance with the same
#   difference of z(Sdir_pattern), z(-Sdir_level cost oriented as S); against matched non-flip genes; by which tissue is t_lo.
#   Moves: rho of (z S hi - z S lo) with amp hi - amp lo, |shift| lo - |shift| hi, expr hi - expr lo, given all genes with support in both.
LIT_PANEL = {  # literature cases fixed before reading: gene, tissue expected low relative to the others, reason
    "SLC2A2": ("islet", "GLUT2 is the mouse beta-cell transporter, low in human beta cells"),
    "MAFB": ("islet", "in adult human beta and alpha cells, adult mouse alpha only"),
    "GAD1": ("islet", "GAD67 in mouse beta cells, GAD65 in human"),
    "ARG1": ("bone_marrow", "human neutrophil granule protein, mouse myeloid pattern differs; shared hepatocyte enzyme in liver"),
    "CD4": ("bone_marrow", "on human monocytes and macrophages, not on mouse ones"),
    "NOS2": ("bone_marrow", "inducible in mouse macrophages, weak in human"),
    "PDK4": ("liver", "X2 hepatocyte level gap in fasting state"),
    "RPL3": ("testis", "X4 level gap between species"),
}
TISSUE_REGULATORS = ["HNF4A", "HNF1A", "HNF1B", "FOXA2", "CEBPA", "NR1H4", "PDX1", "NKX6-1", "MAFA", "ISL1", "NEUROD1", "PAX6", "ARX",
                     "RFX6", "PAX2", "PAX8", "LHX1", "GATA3", "TFCP2L1", "SOX9", "DMRT1", "WT1", "RUNX1", "GATA1", "SPI1", "IKZF1", "CRX",
                     "OTX2", "SOX17", "ERG", "TAL1"]
METABOLIC = ["CPS1", "ASS1", "ASL", "OTC", "PCK1", "G6PC1", "FBP1", "ALDOB", "GCK", "GLS", "GOT1", "HAL", "SDS", "ACOX1", "CAT", "ALAS1",
             "PDK2", "LDHA", "LDHB", "SCD", "FASN", "HMGCS1"]
IMMUNE = ["PTPRC", "CD74", "TYROBP", "FCER1G", "LYZ", "C1QA", "C1QB", "C1QC", "MERTK", "STAB1", "CSF1R", "CD14", "CD68", "MRC1", "CD3E",
          "CD79A", "MS4A1", "NKG7", "GZMB", "IL7R", "CCR7", "TREM2"]
