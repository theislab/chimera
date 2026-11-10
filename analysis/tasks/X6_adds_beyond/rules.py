"""EXPLORATORY: definitions, fixed before any association of this task with a gold was computed.

Every test here was chosen after the decision row, its modifiers and the Family 3 values were known, and after
X4_where_differs, so none is pre-specified and none changes a
scoreboard output, status, row or modifier (row 4, PATTERN-ONLY, REVERSED-TESTIS stand).

Cohorts: liver H+Mouse (liv_hmouse), testis H+Mouse (test_hmouse). Gene tables are the X4_where_differs tables
(results/analysis/X4/genes), so D, D_raw, S, B1b_raw, B1c_raw, B1z_raw, B2b, B3a, golds, covariates and
ESM-module blocks are exactly those of T4_3_scoreboard / X4_where_differs. Sdir is the X4_where_differs direct margin (no model, cell labels).

Statistic. rho_p as T4_3_scoreboard (stats.py): mid-ranks within the universe, OLS residuals on intercept + natural
cubic spline (df 4, rank quartile knots) of each covariate, Pearson of residuals. CI: block bootstrap on
ESM-module blocks, 2,000 draws, percentile, all scores of one unit on the same draws (so deltas and
replicate means have CIs). Conditioning on a score = adding its rank spline to the design, as X4_where_differs.

Golds and base covariates (rules of X4_where_differs / T4_3_scoreboard):
    H1 liver EVEE level, C           H3 liver TS vs TMS pattern, C_pat
    H2 testis EVEE level, C          H4 testis Murat 8 held-out species, C_pat + H4_nsp
    H4p testis Murat primates, C_pat + H4_npri
C = rank pi_raw, expression, detection, CDS length; C_pat = C + tau over shared types.

Part 1, incremental information (one universe per gold: genes with gold, covariates and every one of
D, D_raw, S, Sdir, B1b, B1c, B1z, B2b, B3a; this equals the X4_where_differs 'given_panel' universe).
    simple baselines P = {B1b, B1c, B2b, B1z, B3a}
    D, D_raw, S, Sdir   : given C* + all of P
    D                   : given C* + P + S;   S : given C* + P + D;   S : given C* + P + Sdir
    each B in P         : given C* + P minus B
    variant             : the same with dN/dS and LOEUF added to the conditioning set (smaller universe)
    reproduction        : X4_where_differs panel (B1b, B1c, B1z, B2b, no B3a) for D_raw, S, Sdir must give its numbers
                          (liver H3 D 0.068, S 0.095; testis H4 S 0.055) before anything else is read.
Testis D is the scoreboard D (clamped prior); D_raw (raw prior) is reported beside it.

Part 2, label degradation. Recomputed with each label set: B1b_raw (E), B1c_raw (L), B2b, and also the
X4_where_differs direct margin Sdir (added here because Sdir is the labeled likelihood margin;
defined now, before any association). Same code paths as T4_2_gene_baselines (pseudobulk per library,
log2 1+CPM, detection >= 1% in one group in both species, E on groups shared by the pair, L and B2b on
groups shared by all species) and X4_where_differs (Sdir). Minimum cells per (species, group, library): 30 for label
groups, 20 for Leiden clusters (T4_2_gene_baselines's cluster rule).
    full     : the T4_2_gene_baselines labels (liver obs.cell_type; testis T4_2_gene_baselines coarse map SG, SC, rSD, eSD, Sertoli,
               Other_somatic). Must reproduce T4_2_gene_baselines B1b_raw, B1c_raw, B2b and X4_where_differs Sdir.
    coarse3  : liver epithelial = {Hepatocytes, Cholangio}, endothelial = {Endothelial},
               stromal_immune = {fibroblasts, Kupffer cells, Macrophages, B Plasma cells, HsPCs};
               testis germ_early = {SG, SC}, germ_late = {rSD, eSD}, somatic = {Sertoli, Other_somatic}.
               B2b needs K >= 4 under the plan; with K = 3 it is computed anyway as an exploratory
               cosine and flagged (column B2b, condition coarse3, note K3).
    leiden   : no annotation. Per species separately: normalize_total 1e4, log1p; 2,000 HVGs (scanpy
               'seurat' flavour); scale (max 10); PCA 30 (arpack, seed 0); 15-NN; Leiden resolution 1.0
               (igraph, 2 iterations, seed 0). Matching across species: per cluster with >= 20 cells the
               mean log1p CP10K of every gene; within each species each gene z-scored across clusters;
               markers = per cluster the 50 genes of highest z among genes detected in >= 10% of the
               cluster's cells; M = union over clusters and species; r_ij = Pearson of z on M between
               human cluster i and mouse cluster j; matched = reciprocal best hits with r >= 0.3, each
               matched pair one shared label; unmatched clusters keep species-own labels (not shared).
               Known bias, stated: genes chosen as matching markers are partly the genes whose
               profiles agree, as in any marker-based matching.
    noise10, noise25 : in each species a fraction q (0.10, 0.25) of cells, chosen uniformly without
               replacement, gets a different label drawn from the other types present in that species
               with probability proportional to their abundance (exactly q wrong). Three replicates
               (r0, r1, r2). Testis noise acts on the coarse-map labels.
    leiden_r02 : sensitivity, the same clusters and matching with r >= 0.2. Added after reading
               the testis match table (spermatid and Sertoli clusters reached best r 0.19-0.30 and stayed
               unmatched at 0.3) and before any degradation association was computed; no gold was read.
Universe U_deg per gold: genes with gold, covariates, D, S, B1z_raw and every condition's B1b_raw,
B1c_raw, B2b, Sdir defined. Own-universe values are a sensitivity analysis.
Scores on U_deg: rho_p given C* (the head-to-head estimand) of D, S, B1z (label-free on liver; testis
Chimera and its latent were trained with labels through cell-type-conditional MMD, so testis is not
label-free) and of every degraded score; deltas S - B and D - B on the same draws; noise replicates also
averaged per draw. Secondary: S and D given C* + that condition's B1b, B1c, B2b + B1z + B3a.
Golds scored: liver H3 and H1; testis H4 (primates H4p beside it).

Order of work: definitions fixed before labels were built; labels and
degraded scores built and checked before any gold was joined.
"""

from __future__ import annotations

from analysis.tasks.X4_where_differs import rules as X4

SEED = 20261103
DRAWS = 2000
COHORTS = ["liv_hmouse", "test_hmouse"]
C, C_PAT = X4.C, X4.C_PAT
GOLDS = {
    "liv_hmouse": [("G_H1", "H1 liver EVEE level", C), ("G_H3", "H3 liver TS/TMS pattern", C_PAT)],
    "test_hmouse": [("G_H2", "H2 testis EVEE level", C), ("G_H4", "H4 testis Murat pattern", C_PAT + ["H4_nsp"]),
                    ("G_H4_primates", "H4 testis primates pattern", C_PAT + ["H4_npri"])],
}
DEG_GOLDS = {"liv_hmouse": ["G_H3", "G_H1"], "test_hmouse": ["G_H4", "G_H4_primates"]}
PANEL = ["B1b_raw", "B1c_raw", "B2b", "B1z_raw", "B3a"]
X4_PANEL = ["B1b_raw", "B1c_raw", "B1z_raw", "B2b"]
SEQ = ["dNdS", "LOEUF"]

COARSE3 = {
    "liv_hmouse": {"Hepatocytes": "epithelial", "Cholangio": "epithelial", "Endothelial": "endothelial",
                   "fibroblasts": "stromal_immune", "Kupffer cells": "stromal_immune",
                   "Macrophages": "stromal_immune", "B Plasma cells": "stromal_immune", "HsPCs": "stromal_immune"},
    "test_hmouse": {"SG": "germ_early", "SC": "germ_early", "rSD": "germ_late", "eSD": "germ_late",
                    "Sertoli": "somatic", "Other_somatic": "somatic"},
}
LEIDEN_RMIN = {"leiden": 0.3, "leiden_r02": 0.2}
LEIDEN = dict(n_top_genes=2000, n_pcs=30, n_neighbors=15, resolution=1.0, seed=0, min_cells=20,
              n_markers=50, marker_det=0.10, r_min=0.3)
NOISE = {"noise10": 0.10, "noise25": 0.25}
NOISE_REPS = 3
MIN_CELLS = {"full": 30, "coarse3": 30, "leiden": 20, "noise": 30}
DEG_SCORES = ["B1b_raw", "B1c_raw", "B2b", "Sdir"]


def conditions() -> list[str]:
    out = ["full", "coarse3", "leiden", "leiden_r02"]
    for k in NOISE:
        out += [f"{k}_r{r}" for r in range(NOISE_REPS)]
    return out
