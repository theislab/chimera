"""EXPLORATORY: outputs that only Chimera's architecture produces, tested on independent data.

Definitions fixed here before any validation of this task is computed. All its tests were chosen after the decision row,
its modifiers, the Family 3 values and X4_where_differs were known, so none is pre-specified in the sense of the analysis plan, and none changes a scoreboard
output, status, row or modifier. The per-cell residual share below is a new,
model-derived per-cell output; it is not delta-alpha, and nothing here is a "single-cell resolution" claim
for delta-alpha.

Decoder (chimera/_module.py::forward):  log mu_cg = a_cg + r_cg + log L_c, with
    a_cg = shared(z_c)_g + b_{batch(c), g}          shared path (species-blind decoder) plus batch offset
    r_cg = (1 - alpha_g) * residual(h(z_c), s_c, batch_c)_g     species-specific path
z_c = posterior mean (deterministic). mu_full = exp(a + r) L, mu_sh = exp(a) L (= get_shared_only_expression),
NB with the trained per-batch theta. Model loaded with ChimeraModel.load(accelerator="cpu") as in T2_5_celltype_margin.

Lead A. Per-cell residual share (three definitions; A1 primary)
    A1  tv_c   = 1/2 sum_g |p_full,cg - p_sh,cg|, p = mu / sum_g mu. Share of the cell's predicted composition
               that the species-specific path reassigns. In [0, 1]; independent of depth.
    A2  gain_c = (1 / L_c) sum_g [log NB(x_cg | mu_full) - log NB(x_cg | mu_sh)]. Gain of the full model over the
               shared-only reconstruction, nats per count. Uses the counts.
    A3  lshare_c = sum_g |r_cg| / (sum_g |r_cg| + sum_g |a_cg - abar_g|), abar_g = mean over cells of a_cg: share of
               the cell's predicted log-expression deviation carried by the residual, unweighted over genes.
State level: mean over cells of each (species, type); a type's value is the unweighted mean over the cohort's
species present in it with >= MIN_CELLS cells. Types: liver obs.cell_type, with human Macrophages and mouse
Kupffer cells pooled as "macrophage" where needed for the H3 map; testis the T4_2_gene_baselines coarse map (SG, SC, rSD, eSD,
Sertoli, Other_somatic).

Lead A baselines (same cells)
    kNN species purity: k = K_NN Euclidean neighbors (self excluded) in the latent; purity_c = share of
        neighbors of the cell's own species; normalized npur_c = (purity_c - f_s) / (1 - f_s), f_s = species
        share of the cohort (0 = random mixing, 1 = separated). In Chimera z (posterior mean) and in scVI z
        (training/runs/baselines_scvi/<h5ad>-s42-e100-nl10-bk_<library key>, batch key = library, as Chimera).
    Direct per-type divergence from the training counts: T4_2_gene_baselines pseudobulk (libraries >= 30 cells, mean of
        log2(1 + CPM)); per species pair sharing the type, 1 - Spearman across genes detected (>= 1% of cells)
        in the type in both species; mean over pairs.

Independent per-type divergence (golds; never seen by any model)
    Testis, Murat 2023 held-out species (bonobo, chimpanzee, gorilla, gibbon, marmoset; opossum, platypus,
        chicken), Murat obs.cell_type mapped to the same six coarse types (Spermatogonia SG, Spermatocytes SC,
        Round_spermatids rSD, Elongated_spermatids eSD, Sertoli, Other_somatic), Ensembl 1:1 orthologs to human
        (T4_1_gold union map), library pseudobulk as T4_3_scoreboard (>= 30 cells, mean log2(1 + CPM), library size over mapped
        genes). Per pair and type, 1 - Spearman across genes detected in both. Tier per cohort, fixed:
        test_hmac -> primate-primate pairs (PP); test_hmouse -> primate-outgroup (PO); test_3sp -> all pairs.
        Platypus has no Sertoli; its pairs skip Sertoli. Expectation (Murat 2023): spermatid stages diverge most.
    Liver, TS (10x 3' v3) vs TMS (T4_3_scoreboard h3_profiles.npz, K = 6 types): per type 1 - Spearman across genes of
        TS_x vs TMS_x, genes detected >= 1% in the type in both. Map: Hepatocytes -> hepatocyte, Endothelial ->
        endothelial, fibroblasts -> stellate, Macrophages + Kupffer cells -> macrophage. Used for liv_hmouse and
        liv_4sp (human and mouse cells only); liv_hmonkey and liv_hzeb have no per-type gold for their pair and
        are descriptive only.

Test A-i (type level). Spearman across matched types between the type value (A1; A2, A3 sensitivity) and the
    independent divergence; the same for each baseline. Exact one-sided permutation p over all orderings;
    prediction rho > 0. With 4 liver types the smallest p is 1/24; these are descriptive.
Test A-ii (gene x type). r̄_gsk = mean over cells of (s, k) of r_cg; R_gk = SD over species of r̄_gsk (two species:
    |difference| / 2): the species divergence the residual posits for g in k. Independent I_gk: liver
    |TS_x - TMS_x - median_k| (genes detected in k in both atlases); testis mean over the tier's pairs of
    |x_s1 - x_s2 - median_pair,k| (genes detected in k in both; >= 3 pairs). Direct baseline Ddir_gk = SD over
    species of the training pseudobulk x_sgk after removing each species' median offset in k. rho_p within each
    matched type, covariates = spline of the mean training pseudobulk level and detection in k; block bootstrap
    over the T4_3_scoreboard gene modules (X4_where_differs rho_unit); prediction rho > 0; head-to-head R minus Ddir. Cohorts with X4_where_differs
    gene tables (liv_hmouse, liv_4sp, test_hmouse, test_3sp).
Drivers. In the matched type with the highest independent divergence, genes ranked by R_gk; named only if T2_5_celltype_margin
    margin_support, unmasked S, counts in every species (X4_where_differs nameable without the seed-sign condition is not
    used: the X4_where_differs ``nameable`` column is required).

Lead B. Species-resolved margins in one joint fit
    S_gs = mean over cells of species s of l_cg (T2_5_celltype_margin forward pass, z = posterior mean): the species term of
        S_g = mean_s S_gs. Type-balanced S^tb_gs = mean over types shared by all species of S_gsk (T2_5_celltype_margin S_gsk;
        testis cell_ontology_class, liver cell_type) as sensitivity.
    Testis 3-sp: Bt_g = (S_gHuman + S_gMacaque) / 2 - S_gMouse. Gold Gt_g = zbar_PP - zbar_PO, the tier means of
        G_H4 (T4_3_scoreboard H4 pipeline, 4 types, >= 3 primates and >= 2 outgroups). Covariates C_pat + H4_nsp.
        Prediction rho_p > 0 (mouse fits the shared model worse where held-out primates agree with each other
        more than with outgroups). Baselines from the same counts: B1b pair contrast
        E(H,Mac) - [E(H,Mo) + E(Mac,Mo)] / 2 (T4_2_gene_baselines E, coarse types, each pair's shared types); direct margin
        contrast (Cspdir_H + Cspdir_Mac) / 2 - Cspdir_Mo (X4_where_differs cache); and E(H,Mac) alone.
    Liver 4-sp: Bl_g = S_gMouse - S_gMonkey. Gold G_H3 (TS vs TMS), covariates C_pat. Prediction rho_p > 0.
        Secondary Bz_g = S_gMouse - S_gZebrafish (X4_where_differs's contrast; zebrafish fit not separable from mixing).
        Baselines: E(H,Mo) - E(H,Mon) and E(H,Mo) - E(H,Zeb) from the 4-sp counts; Cspdir contrasts.
    Universe: unmasked S (|S_g_ema| >= 1e-6) and counts in every species (nonzero_all_species); all genes as
        sensitivity. rho_p and block CI as X4_where_differs (common.rho_unit), ref = the Chimera contrast.
    Mixing, explicit: per seed and species the mean normalized kNN purity in Chimera z; the Chimera contrast is
        reported per seed beside it.

Seeds. Lead A and B forward passes on the canonical runs and the seeds listed in SEEDS (all exist and were
trained before X7). A seed result is reported as range; nothing is selected by seed.

Changed before any Chimera output of this task was read (only the golds had been computed): Murat's
primate objects carry the ten fine stages (Undifferentiated/Differentiated Spermatogonia, Leptotene/Zygotene/
Pachytene Spermatocytes, Early/Late Round, Elongated spermatids, Sertoli, Other_somatic), the outgroups only the
coarse types. The coarse map therefore also maps the fine labels (T4_2_gene_baselines TESTIS_COARSE). Added test A-i fine:
test_hmac (human and macaque both carry the ten fine stages), type value per fine stage against the PP-tier
divergence per fine stage across the five held-out primates (10 pairs), same statistic; and the same for
test_3sp restricted to human and macaque cells.
"""

from __future__ import annotations

K_NN = 30
MIN_CELLS = 30
DET_MIN = 0.01
MASK_EPS = 1e-6
SEED = 20261102
SCVI_SEED = 42

LIVER_H3_MAP = {"Hepatocytes": "hepatocyte", "Endothelial": "endothelial", "fibroblasts": "stellate",
                "macrophage": "macrophage"}
LIVER_MACRO = {"Macrophages": "macrophage", "Kupffer cells": "macrophage"}
TESTIS_TYPES = ["SG", "SC", "rSD", "eSD", "Sertoli", "Other_somatic"]
MURAT_COARSE = {"Spermatogonia": "SG", "Spermatocytes": "SC", "Round_spermatids": "rSD",
                "Elongated_spermatids": "eSD", "Sertoli": "Sertoli", "Other_somatic": "Other_somatic",
                "Undifferentiated_Spermatogonia": "SG", "Differentiated_Spermatogonia": "SG",
                "Leptotene_Spermatocytes": "SC", "Zygotene_Spermatocytes": "SC", "Pachytene_Spermatocytes": "SC",
                "Early_Round_spermatids": "rSD", "Late_Round_spermatids": "rSD"}
FINE_STAGES = ["Undifferentiated_Spermatogonia", "Differentiated_Spermatogonia", "Leptotene_Spermatocytes",
               "Zygotene_Spermatocytes", "Pachytene_Spermatocytes", "Early_Round_spermatids", "Late_Round_spermatids",
               "Elongated_spermatids", "Sertoli", "Other_somatic"]
TESTIS_TIER = {"test_hmac": "PP", "test_hmouse": "PO", "test_3sp": "all"}
GOLD_COHORTS_A = {"liv_hmouse": "liver", "liv_4sp": "liver", "test_hmac": "testis", "test_hmouse": "testis",
                  "test_3sp": "testis"}
GENE_TYPE_COHORTS = ["liv_hmouse", "liv_4sp", "test_hmouse", "test_3sp"]
SEEDS = {
    "liv_hmonkey": [43, 44],
    "liv_hmouse": [43, 44, 45, 46],
    "liv_4sp": [43, 44, 45, 46],
    "test_hmouse": [43, 44, 45, 46],
    "test_3sp": [43, 44, 45, 46],
}
LIB_KEY = {"liver": "batch", "testis": "sample_batch"}
