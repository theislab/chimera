"""H1: one reading across tissues. Registry and definitions, fixed before any cross-tissue association was computed.

Exploratory throughout. Nothing here changes a status, row or modifier of the plan. Vocabulary: CHIMERA_MECHANISM.md.

Cohorts. Reference: canonical liver and testis. New: bone marrow, kidney healthy, islet healthy, Van Zyl eye (full, H+Mouse,
H+M. mulatta, primates; H+Pig and H+M. fascicularis join when their seeds exist). Seeds: canonical = the T3.3 seed sets of the
paper; new cohorts = every seed passing the health rule (finished, no NaN, 10 of 10 active z dimensions, both raw-prior tails at
>= 2% of genes, saturation at most 10%); 2 or more seeds suffice, failing runs are listed in runs.tsv and not analysed.

Per-gene readouts, one schema (tables.py):
  pi_raw      unclamped protein-identity prior (conservation_prior_raw.npy, else the run's prior);
  pi_clamped  the prior the run anchors on (equal to pi_raw on unclamped recipes);
  alpha_s<k>  alpha per kept seed;  alpha = their mean;
  delta_alpha mean alpha - pi_clamped, uncentred;
  D           delta_alpha minus its cohort median;  D_raw = mean alpha - pi_raw minus its cohort median;
  S_ema       mean S_g_ema over kept seeds;
  count_support  |S_ema| >= 1e-6 (X4, Z1);
  strict_support count_support, |S_ema| >= cohort median over supported genes, sign(S_ema) = sign(D), sign of S_g_ema and of
              alpha_seed - pi_clamped - seed median stable over every kept seed;  named genes need strict_support. Saturated genes (alpha > 0.95 in any
              seed) are flagged and may be named. Amendment, after the first gene lists: the first rule also excluded saturated genes, which on clamped recipes removes the
              whole upper tail near the ceiling (pi_clamped 0.85 + 0.10), including most lineage genes; their sign is backed by counts, only their rank among themselves is capped;
  expr, detection  species-balanced mean log1p CP10K and detection on the training object (cohorts.gene_covariates).

Sign. Compared across cohorts only after centring, and in the set analysis only against genes matched on the prior and on
expression in the same cohort.

Set analysis (recurrence.py). Sets: the 50 MSigDB Hallmarks (Enrichr 2020) and the X3 gene classes in CLASSES, by human Ensembl ID
(Hallmark symbols through the cohort's human symbols). Universe: count-supported genes of the cohort. A set enters a cohort with at
least MIN_SET genes in the universe. Value: inverse-normal transform of the within-cohort rank of the readout. Set score: mean over
members. Null: values permuted among genes of the same stratum, within cohort, independently in each cohort, N_PERM draws, same draws
for all sets. Strata: PI_BINS quantile bins of pi_raw x EXPR_BINS quantile bins of expr (pi_raw and dN/dS as readouts: expr only).
z = (score - null mean) / null SD; p two-sided empirical; q BH over the sets of the cohort.
Recurrence statistic, per set over the six primary tissues: R = max(number of tissues with z >= Z_CALL, number with z <= -Z_CALL).
Its null: the same statistic on the permuted draws. p_rec = (1 + #{R_null >= R}) / (N_PERM + 1); q_rec BH over all sets.
A set is called recurrent when R >= R_MIN and q_rec < 0.05.
Tissue-specific: called in one tissue only (q < 0.05 there, |z| < Z_CALL in every other), and the contrast z_t - mean z_others beyond
its permutation null, BH over all such candidates, q < 0.05.
Readouts beside D: S_ema (count evidence, no features), B1 (pseudobulk profile correlation, baseline), pi_raw and dN/dS (sequence,
baseline). D_raw and all-gene universes are sensitivities. Because every fit's GIN reads the same ESM coordinates, recurrence on D may
come from protein features; recurrence on S_ema does not. Both are reported.

Post hoc sensitivities, added after 48 of 72 sets recurred on D and the sequence baselines recurred about as often, which says the gene-level null is a low bar for coherent sets: (i) D_tau, S_tau: strata
also on cell-type specificity tau (5 bins), since the margin rewards having a pattern; (ii) D_block, S_block: a null that keeps ESM-module structure (block_scores, 2,000 draws) on values centred within strata. The main-figure list
is the primary rule; survival under (i) and (ii) is shown beside it.

Separate axis (recurrence.axis_table), per cohort: Spearman of D with pi_raw and with log10 dN/dS (Compara human-mouse, T4.3 gold table), on all genes
and on count-supported genes; the same for S_ema and D_raw; rho(pi_raw, dN/dS) as the reference; partial rho(D, dN/dS | pi_raw, expr, detection).

Main thread. Human and mouse are in common in every tissue, so the six-tissue reading uses the H+Mouse fit of each tissue; bone marrow
has only the three-species fit (human, lemur, mouse). The same six as Y1 (which owns gene x tissue variance and tissue-matched tests).

Lineage panels below are fixed from textbook markers before any readout of these tissues was read in H1; they decide which genes are
looked at for "what the tissue adds", not which are significant. Only strict-support genes are named.
"""

from __future__ import annotations

import os
from pathlib import Path

from revision.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "revision"
OUT = Path(os.environ.get("H1_OUT", RES / "H1"))  # H1_OUT: dry runs only
RUNS = REPO / "training" / "runs"
SEED = 20261102
MASK_EPS = 1e-6
TAIL = 0.05
SAT_HI = 0.95
SAT_MAX = 0.10
TAIL_MIN = 0.02

MIN_SET = 15
N_PERM = 10_000
PI_BINS = 5
EXPR_BINS = 10
Z_CALL = 1.96
R_MIN = 3

P = REPO / "data" / "processed"
VZ = REPO / "data" / "revision" / "van_zyl"
_H = {s: f"rev-liv_hmouse-s{s}" for s in range(45, 52)}
_T = {s: f"rev-test_hmouse-s{s}" for s in range(43, 52)}

# name -> tissue, label, species shown, h5ad, ortholog table key, cell-type key, integration label key, seeds tried
COHORTS = {
    "liv_hmonkey": dict(tissue="liver", label="Liver H+Monkey", species="human, macaque", h5ad=P / "camex_human_monkey.h5ad", table="liv_hmonkey",
                        ct="cell_type", role="reference", tried={42: "iter90-150ep-s42", 43: "iter90-150ep-s43", 44: "iter90-150ep-s44"}),
    "liv_hmouse": dict(tissue="liver", label="Liver H+Mouse", species="human, mouse", h5ad=P / "camex_human_mouse.h5ad", table="liv_hmouse", ct="cell_type",
                       role="reference", tried={42: "xds-hmouse-iter90-150ep-s42", 43: "xds-hmouse-iter90-150ep-s43", 44: "xds-hmouse-iter90-150ep-s44", **_H}),
    "liv_hzeb": dict(tissue="liver", label="Liver H+Zebrafish", species="human, zebrafish", h5ad=P / "camex_human_zebrafish.h5ad", table="liv_hzeb", ct="cell_type",
                     role="reference", tried={42: "xds-hzeb-iter85-s42"}),
    "liv_4sp": dict(tissue="liver", label="Liver 4 species", species="human, macaque, mouse, zebrafish", h5ad=P / "camex_pancreas_unified.h5ad", table="liv_4sp",
                    ct="cell_type", role="reference",
                    tried={42: "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42", **{s: f"rev-liv_4sp-s{s}" for s in range(43, 47)}}),
    "test_hmac": dict(tissue="testis", label="Testis H+Macaque", species="human, macaque", h5ad=P / "camex_testis_human_macaque.h5ad", table="test_hmac",
                      ct="cell_ontology_class", role="reference", tried={42: "testis-hmac-iter109-pclamp085-200ep-s42"}),
    "test_hmouse": dict(tissue="testis", label="Testis H+Mouse", species="human, mouse", h5ad=P / "camex_testis_human_mouse.h5ad", table="test_hmouse",
                        ct="cell_ontology_class", role="reference", tried={42: "testis-hmouse-iter111-pclamp-coc-200ep-s42", **_T}),
    "test_3sp": dict(tissue="testis", label="Testis 3 species", species="human, macaque, mouse", h5ad=P / "camex_testis_unified.h5ad", table="test_3sp",
                     ct="cell_ontology_class", role="reference",
                     tried={42: "testis-unified-iter115-pclamp-coc-100ep-s42", **{s: f"rev-test_3sp-s{s}" for s in range(43, 47)}}),
    "bone_marrow": dict(tissue="bone marrow", label="Bone marrow 3 species", species="human, mouse lemur, mouse", h5ad=gene_ids.h5ad_path("bone_marrow"),
                        table="bone_marrow", ct="cell_ontology_class", role="new",
                        tried={42: "rev-bm2-grid5-s42", **{s: f"rev-bm2-s{s}" for s in range(43, 47)}}),
    "kidney": dict(tissue="kidney", label="Kidney H+Mouse", species="human, mouse", h5ad=gene_ids.h5ad_path("kidney"), table="kidney", ct="cell_ontology_class",
                   role="new", tried={42: "rev-kid-grid3-s42", **{s: f"rev-kid-s{s}" for s in range(43, 47)}}),
    "islet": dict(tissue="islet", label="Islet H+Mouse", species="human, mouse", h5ad=REPO / "data/revision/disease_islet/islet_healthy_train.h5ad", table="testis",
                  ct="cell_ontology_class", role="new", tried={42: "rev-isl-grid14-s42", **{s: f"rev-isl-s{s}" for s in range(43, 47)}}),
    "vz_hmouse": dict(tissue="eye", label="Eye H+Mouse", species="human, mouse", h5ad=VZ / "van_zyl_hmouse.h5ad", table="van_zyl", ct="cell_ontology_class", role="new",
                      tried={42: "rev-vz-hmouse-grid1-s42", 43: "rev-vz-hmouse-s43", 44: "rev-vz-hmouse-s44"}),
    "vz_hmmul": dict(tissue="eye", label="Eye H+M. mulatta", species="human, M. mulatta", h5ad=VZ / "van_zyl_hmmul.h5ad", table="van_zyl", ct="cell_ontology_class",
                     role="new", tried={42: "rev-vz-hmmul-grid1-s42", 43: "rev-vz-hmmul-s43", 44: "rev-vz-hmmul-s44"}),
    "vz_primates": dict(tissue="eye", label="Eye primates", species="human, M. fascicularis, M. mulatta", h5ad=VZ / "van_zyl_primates.h5ad", table="van_zyl",
                        ct="cell_ontology_class", role="new", tried={42: "rev-vz-primates-grid2-s42", 43: "rev-vz-primates-s43", 44: "rev-vz-primates-s44"}),
    "vz_full": dict(tissue="eye", label="Eye 5 species", species="human, M. fascicularis, M. mulatta, mouse, pig", h5ad=VZ / "van_zyl_unified.h5ad", table="van_zyl",
                    ct="cell_ontology_class", role="new", tried={42: "rev-vz-grid1-s42", **{s: f"rev-vz-s{s}" for s in range(43, 47)}}),
}
# Added once Z1 lists their selected seeds: one row per run in extra_runs.tsv (cohort, seed, run); see README.
PENDING = {
    "vz_hpig": dict(tissue="eye", label="Eye H+Pig", species="human, pig", h5ad=VZ / "van_zyl_hpig.h5ad", table="van_zyl", ct="cell_ontology_class", role="new", tried={}),
    "vz_hmfas": dict(tissue="eye", label="Eye H+M. fascicularis", species="human, M. fascicularis", h5ad=VZ / "van_zyl_hmfas.h5ad", table="van_zyl", ct="cell_ontology_class",
                     role="new", tried={}),
}
PRIMARY = {"liver": "liv_hmouse", "testis": "test_hmouse", "bone marrow": "bone_marrow", "kidney": "kidney", "islet": "islet", "eye": "vz_hmouse"}
TISSUE_ORDER = ["liver", "testis", "bone marrow", "kidney", "islet", "eye"]
NEW_TISSUES = ["bone marrow", "kidney", "islet", "eye"]

CLASSES = ["TF", "TF_homeodomain", "TF_C2H2_ZF", "TF_nuclear_rec", "protein_kinase", "receptor", "GPCR", "ion_channel", "SLC_transporter",
           "chromatin_reg", "enzyme", "mitochondrial"]

# What each tissue may add: textbook lineage markers, fixed before reading. Looked at, not tested as lists of hits.
LINEAGE = {
    "liver": {"hepatocyte nutrient programme": ["CPS1", "ARG1", "ASL", "ASS1", "OTC", "PCK1", "G6PC1", "PAH", "ALB", "APOA1", "CYP3A4", "HNF4A", "ALDOB", "TAT"],
              "liver endothelium and support": ["STAB2", "LYVE1", "DCN", "COL1A1", "PDGFRB", "MARCO", "CD5L"]},
    "testis": {"germline": ["DDX4", "DAZL", "SYCP3", "SYCP1", "PIWIL1", "STRA8", "PRM1", "PRM2", "TNP1", "ACRV1", "ZPBP2", "SPO11", "TEX101"],
               "somatic": ["SOX9", "AMH", "CYP17A1", "STAR", "INSL3", "WT1"]},
    "bone marrow": {"B and plasma": ["CD19", "MS4A1", "CD79A", "CD79B", "PAX5", "EBF1", "JCHAIN", "PRDM1", "XBP1", "SDC1"],
                    "T and NK": ["CD3D", "CD3E", "CD4", "CD8A", "IL7R", "GZMB", "NKG7", "KLRD1", "NCAM1", "TBX21", "EOMES"],
                    "myeloid": ["CD14", "LYZ", "SPI1", "CEBPA", "IRF8", "CSF1R", "MPO", "ELANE", "S100A8", "S100A9", "CSF3R"],
                    "erythroid and progenitor": ["GATA1", "KLF1", "HBB", "ALAS2", "EPOR", "CD34", "KIT", "GATA2", "MEIS1"]},
    "kidney": {"proximal tubule": ["LRP2", "CUBN", "SLC34A1", "SLC5A2", "SLC22A6", "SLC22A8", "ALDOB", "PCK1", "HNF4A"],
               "loop and distal tubule": ["SLC12A1", "UMOD", "CLDN16", "SLC12A3", "CALB1", "TRPM6", "PVALB"],
               "collecting duct": ["AQP2", "AQP3", "SCNN1G", "SLC4A1", "FOXI1", "ATP6V1B1"],
               "glomerulus": ["NPHS1", "NPHS2", "PODXL", "WT1", "EHD3", "PTPRB"]},
    "islet": {"endocrine": ["INS", "GCG", "SST", "PPY", "IAPP", "PDX1", "NKX6-1", "MAFA", "MAFB", "ARX", "SLC2A2", "GCK", "ABCC8", "KCNJ11", "CHGA", "NEUROD1", "ISL1"],
              "exocrine and duct": ["CPA1", "PRSS1", "CELA3A", "PNLIP", "CTRB1", "KRT19", "SOX9", "CFTR", "HNF1B"],
              "stroma": ["PECAM1", "PLVAP", "COL1A1", "PDGFRB", "ACTA2"]},
    "eye": {"trabecular meshwork and aqueous outflow": ["MYOC", "MGP", "CHI3L1", "ANGPTL7", "DCN", "TMEFF2", "NELL2", "EDN3", "FABP4", "BMP5"],
            "Schlemm's canal and vessels": ["TEK", "ANGPT1", "ANGPT2", "PECAM1", "PLVAP", "FLT4", "POSTN", "NOS3"],
            "ciliary muscle and pigment": ["DES", "MYH11", "ACTA2", "TYR", "PMEL", "MLANA", "DCT", "RPE65"],
            "glaucoma-linked": ["OPTN", "TBK1", "CYP1B1", "FOXC1", "PITX2", "LTBP2", "CAV1", "CDKN2B", "SIX6", "LMX1B"]},
}


def load_extra():
    """Register H+Pig and H+M. fascicularis seeds from extra_runs.tsv when present (cohort, seed, run)."""
    import pandas as pd

    f = Path(os.environ.get("H1_EXTRA", Path(__file__).resolve().parent / "extra_runs.tsv"))
    out = dict(COHORTS)
    if f.exists():
        t = pd.read_csv(f, sep="\t")
        for c, g in t.groupby("cohort"):
            out[c] = {**PENDING[c], "tried": {int(r.seed): str(r.run) for r in g.itertuples()}}
    return out


def cohorts():
    return load_extra()
