"""Y5 (EXPLORATORY, Family 3): what else Chimera reveals, beyond the single-gene score against golds.

Fixed here before any Y5 association is computed. All Y5 tests are chosen after G3 and X1-X7 were known, so none
is pre-specified in the sense of the analysis plan; none changes a status, row or modifier. Vocabulary follows
ledgers/CHIMERA_MECHANISM.md. Predictions and the reading of each statistic are in ledgers/Y5_aspects.md.

Hold-out. Genes are split by human chromosome: discovery = odd autosomes, replication = even autosomes. X, Y, MT
and scaffolds are in neither half (X enters family F6 only, where the odd/even rule cannot apply and the replication
is an independent cohort). Where cells are the unit: independent cohort (TS, TMS) or held-out libraries.

Survivor rule (every family). BH across ALL discovery p-values of Chimera readouts over ALL families (one global
list, `tried.csv`), q < 0.05; replication half same sign and p < 0.05 one-sided; a direct baseline or another tool is
reported beside it on the same genes or cells. Effects with |rho_p| < 0.05 in either half are called "replicated,
small". Named genes: X4 `nameable` only (margin-supported, seed-stable).

Seeds. G-V training-health gate (T12_2b select_recipe.gate; amendment a82eaad): no NaN, saturation <= 5% per tail,
10/10 active z dims, both tails >= 2% of genes. Read on the canonical cohorts (results/analysis/Y5/seed_health.csv, health
quantities only) the literal gate leaves: liv_4sp 42-46; liv_hmouse 42-45, 47, 49, 50; test_3sp 43, 44, 46; none of
test_hmouse, test_hmac (the negative tail is measured against the 0.85-clamped prior, where it is empty by construction),
liv_hmonkey (saturation 5.8-6.5%) or liv_hzeb (no negative tail). Rule, fixed on those quantities only: a survivor needs the
literal-gate seeds. On testis that means test_3sp 43, 44, 46. test_hmouse and test_hmac seeds pass the gate with the tail
measured on the raw prior (`pass_GV_rawtail`, all 9 and 1): they are reported as a labelled sensitivity beside test_3sp and
cannot create a survivor. liv_hmonkey and liv_hzeb are descriptive only. Poorly mixed healthy seeds stay and are reported
(the damage from poor mixing falls on D, not on S).

Readouts. S = mean of S_g_ema over healthy seeds. D_raw = mean(alpha - pi_raw) over healthy seeds, centred on the cohort
median. c_full = MLP(f); c_esm = MLP(f with expr3 columns set to 0, the scaled mean); c_expr = MLP(f with esm64 and
esm_var set to 0), mean over healthy seeds. These two are the GIN's protein-only and expression-only responses;
identity scalars stay in both. A masked gene's D is feature extrapolation: analyses of S use unmasked genes.

Statistic. rho_p = rank partial correlation given C4 = spline(pi_raw) + spline(log_cp10k_spmean) +
spline(detection_spmean) + spline(log_cds_length) (T4.3 stats, X4 rho_unit); length as a property is tested without
the length term. Sensitivity C_pat = C4 + tau_shared, because S and D first reward having a pattern (P12). CI and p from
the block bootstrap over the T4.3 ESM gene modules (500 draws in scans, 2000 for survivors), p two-sided normal approx.

Comparators beside every gene result: B1b, B1c, B2b from T4.2; Sdir (X4 labelled no-model margin; pattern gain minus
level cost) where X4 has it. Cell results: scVI, scANVI, sysVI, Harmony, PCA, SATURN, CAMEX latents on the same cells.
"""

from __future__ import annotations

ODD = {str(i) for i in range(1, 23, 2)}
EVEN = {str(i) for i in range(2, 23, 2)}
DRAWS_SCAN, DRAWS_SURVIVOR = 500, 2000
ALPHA_Q = 0.05
SMALL = 0.05

PRIMARY = ["liv_hmouse", "liv_4sp", "test_3sp"]             # literal G-V seeds; X4 tables (Sdir, gold, block)
SENS_RAWTAIL = ["test_hmouse"]                               # raw-prior tail; X4 tables; cannot create a survivor
DESCRIPTIVE = ["liv_hmonkey", "liv_hzeb", "test_hmac"]       # literal G-V fails; no X4 table
EXTERNAL_TISSUE = {"vz_hmouse": "results/analysis/T11/readout/genes_hmouse.parquet",
                   "vz_full": "results/analysis/T11/readout/genes_full.parquet",
                   "vz_hmmul": "results/analysis/T11/readout/genes_hmmul.parquet",
                   "bm": "results/analysis/P12/readout/bm_genes.parquet"}

# ----------------------------------------------------------------- F1 protein properties (UniProt 2026 pull, human reviewed)
# binary = present/absent; frac = residues in the annotated feature / length; dens = features per 100 residues.
PROPERTIES = {
    "signal_peptide": "binary ft_signal",
    "transmembrane": "binary ft_transmem",
    "disorder_frac": "frac ft_region text 'Disordered'",
    "domain_dens": "dens ft_domain",
    "compbias_frac": "frac ft_compbias",
    "coiled_frac": "frac ft_coiled",
    "ptm_dens": "dens ft_mod_res",
    "active_site": "binary ft_act_site",
    "dna_zn_binding": "binary ft_dna_bind or ft_zn_fing",
    "loc_nucleus": "cc_subcellular_location contains Nucleus",
    "loc_cytoplasm": "contains Cytoplasm",
    "loc_mitochondrion": "contains Mitochondrion",
    "loc_secreted": "contains Secreted",
    "loc_plasma_membrane": "contains Cell membrane",
    "loc_er_golgi": "contains Endoplasmic reticulum or Golgi",
    "gravy": "mean Kyte-Doolittle hydropathy of the sequence",
    "ncpr": "net charge per residue of the sequence",
    "log_length": "log protein length; tested with C3 (no CDS length term)",
    "paralog_count": "log1p count of human within-species paralogs, Compara 110",
    "ppi_degree": "log1p STRING v12 physical degree, score >= 700",
    "ortholog_breadth": "count of 7 deep taxa with a Compara 110 ortholog (chicken, anole, xenopus, zebrafish, coelacanth, spotted gar, lamprey)",
}
F1_READOUTS = ["S", "D_raw", "c_esm", "c_expr"]
F1_COMPARATORS = ["B1b", "B1c", "B2b", "Sdir"]
# A property is tissue-replicating if it survives in liver and in testis (H+Mouse pair primary) with one sign.

# ----------------------------------------------------------------- F2 phylogenetic depth (Murat held-out species only)
# Million years to the last common ancestor (TimeTree median values, fixed here).
MYA = {("bonobo", "chimpanzee"): 2.8, ("gorilla", "chimpanzee"): 8.6, ("gorilla", "bonobo"): 8.6,
       ("gibbon", "chimpanzee"): 19.5, ("gibbon", "bonobo"): 19.5, ("gibbon", "gorilla"): 19.5,
       ("marmoset", "chimpanzee"): 43.0, ("marmoset", "bonobo"): 43.0, ("marmoset", "gorilla"): 43.0,
       ("marmoset", "gibbon"): 43.0}
PRIMATES = ["bonobo", "chimpanzee", "gorilla", "gibbon", "marmoset"]
for _p in PRIMATES:
    MYA[("opossum", _p)] = 160.0
    MYA[("platypus", _p)] = 180.0
    MYA[("chicken", _p)] = 318.0
MYA[("opossum", "platypus")] = 180.0
MYA[("chicken", "opossum")] = 318.0
MYA[("chicken", "platypus")] = 318.0
DEPTH_BINS = {"lt10": (0, 10), "10_45": (10, 45), "160_180": (150, 190), "318": (300, 330)}
DEPTH_MIN_PAIRS = 1
# Depth bins are evolutionary distance among species no model saw. Pair-level identity: Compara 110 human-to-species
# identity of the deep member, held in the covariates for the 160_180 and 318 bins.
DEPTH_CLASSES = ["cls_TF", "cls_TF_homeodomain", "cls_TF_C2H2_ZF", "cls_protein_kinase", "cls_receptor", "cls_GPCR",
                 "cls_enzyme", "cls_EC6", "cls_mitochondrial", "cls_ion_channel", "cls_SLC_transporter", "cls_chromatin_reg"]

# ----------------------------------------------------------------- F3 cell states inside an author label
KNN_CROSS = 30
TOOLS = ["scvi", "scanvi", "sysvi", "harmony", "pca", "saturn", "camex"]
# Literature markers in human symbols, mapped through the cohort's ortholog table; split by chromosome.
# Hepatocyte zonation: Halpern 2017 (Nature), Aizarani 2019 (Nature), MacParland 2018 (Nat Commun), Ben-Moshe 2019.
ZONE = {
    "hepatocyte": {
        "up": ["GLUL", "CYP2E1", "CYP1A2", "OAT", "AXIN2", "LECT2", "SLC1A2", "RGN", "CYP27A1", "CYP7A1", "NOTUM",
               "TBX3", "RNF43", "ZNRF3", "LGR5", "ADH4", "CYP3A4", "ALDH1A1"],                       # pericentral
        "dn": ["CPS1", "HAL", "SDS", "ASS1", "ASL", "PCK1", "G6PC1", "ARG1", "ALDOB", "ALB", "PIGR", "HAMP",
               "APOA1", "FBP1", "GLS2", "CTSC", "TAT", "ITIH4"]},                                     # periportal
    # Sinusoidal vs non-sinusoidal endothelium: MacParland 2018, Aizarani 2019, Halpern 2018 (Nat Biotechnol).
    "endothelial": {
        "up": ["STAB2", "STAB1", "LYVE1", "CLEC4G", "CLEC4M", "OIT3", "MRC1", "GPR182", "FCGR2B", "CD36"],
        "dn": ["VWF", "PECAM1", "CD34", "AQP1", "ACKR1", "SELE", "EFNB2", "SOX17", "PRSS23", "CLU"]},
    # Stellate vs portal fibroblast, mesothelial and mural: Dobie 2019 (Cell Rep), Ramachandran 2019 (Nature).
    "fibroblast": {
        "up": ["LRAT", "RELN", "PDGFRB", "COLEC11", "HGF", "NGFR", "LUM", "CXCL12"],
        "dn": ["MSLN", "UPK3B", "ELN", "PI16", "MYH11", "ACTA2", "TAGLN", "KRT19"]},
    # Resident Kupffer programme vs monocyte-like: Guilliams 2022 (Cell), MacParland 2018.
    "macrophage": {
        "up": ["MARCO", "CD5L", "VSIG4", "SLC40A1", "CD163", "FOLR2", "TIMD4", "C1QA", "C1QB"],
        "dn": ["S100A8", "S100A9", "VCAN", "FCN1", "CCR2", "ITGAL", "PLAC8", "SELL"]},
}
LIVER_TYPE = {"hepatocyte": ["Hepatocytes"], "endothelial": ["Endothelial"], "fibroblast": ["fibroblasts"],
              "macrophage": ["Macrophages", "Kupffer cells"]}
# Fine testis stages sit inside the coarse classes used by conditional MMD (cell_ontology_class). Ordinal rank within class.
STAGE_RANK = {"Undifferentiated_Spermatogonia": 1, "Differentiated_Spermatogonia": 2,
              "Leptotene_Spermatocytes": 3, "Zygotene_Spermatocytes": 4, "Pachytene_Spermatocytes": 5,
              "Early_Round_spermatids": 6, "Late_Round_spermatids": 7}
STAGE_CLASS = {"Undifferentiated_Spermatogonia": "SG", "Differentiated_Spermatogonia": "SG",
               "Leptotene_Spermatocytes": "SC", "Zygotene_Spermatocytes": "SC", "Pachytene_Spermatocytes": "SC",
               "Early_Round_spermatids": "rSD", "Late_Round_spermatids": "rSD"}

# ----------------------------------------------------------------- F4 modules
PROFILE_MIN_CELLS = 30
PATHWAY_SIZE = (10, 300)
STRING_THR = 700
MODULE_K = 15
MODULE_RES = 1.0

# ----------------------------------------------------------------- F5 lineage cases fixed before reading (descriptive only)
# Species x type expectations, literature; each is reported with S_gsk contrast and the pseudobulk gap, not counted in BH.
LINEAGE_CASES = [
    ("APOBEC1", "Hepatocytes", "Mouse", "ApoB mRNA editing in rodent liver, not human (Greeve 1993; Teng 1993)"),
    ("CLEC4F", "macrophage", "Mouse", "Clec4f marks mouse Kupffer cells; low in human (Guilliams 2022)"),
    ("MARCO", "macrophage", "Human", "MARCO high in human resident Kupffer cells (MacParland 2018)"),
    ("SLC22A1", "Hepatocytes", "Human", "OCT1 strongly hepatic in human; also kidney in rodent (Jonker 2003)"),
    ("FMO3", "Hepatocytes", "Mouse", "male mouse liver silences Fmo3 (Janmohamed 2004)"),
    ("PPARA", "Hepatocytes", "Mouse", "human liver PPARA lower than mouse (Palmer 1998)"),
    ("GFRA1", "SG", "Mouse", "mouse spermatogonial stem marker, weaker in primates (Fayomi 2018; Shami 2020)"),
    ("UTF1", "SG", "shared", "spermatogonial marker conserved human, monkey, mouse (Shami 2020)"),
    ("PIWIL4", "SG", "shared-primate", "primate early SG, retained in human (Shami 2020; Guo 2018)"),
    ("SOX9", "Sertoli", "shared", "conserved Sertoli identity (Shami 2020)"),
    ("PRM3", "eSD", "shared", "late-spermatid gene"),
]

# ----------------------------------------------------------------- F6 X linkage in testis
MSCI_TYPES = ["SC", "rSD", "eSD"]        # X silenced in meiosis, partly after (Turner 2007; Namekawa 2006)
