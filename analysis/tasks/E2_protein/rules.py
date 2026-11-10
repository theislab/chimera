"""E2: protein-level and gene-architecture properties of deployment (EXPLORATORY). Definitions fixed before any association.

Exploratory throughout; nothing here changes a status, row, modifier or H1/Z4 table. Vocabulary: CHIMERA_MECHANISM.md.
Extends Y5 F1 (21 UniProt/Compara/STRING properties, canonical gate, three cohorts) to all 26 harmonised cohorts, both gates and
new properties. Y5's annotation table (results/analysis/Y5/annot.parquet) is reused for the properties it already defines.

Read-outs (per gene, per cohort; both tables in H1's schema):
  D_can   canonical gate (main model): results/analysis/H1/genes/<cohort>.parquet, column D_raw (mean alpha - pi_raw, centred).
  D_free  per-gene gate: results/analysis/Z4/h1/genes/<cohort>.parquet, column D_raw; kept seeds in Z4/kept.json.
  S       count evidence S_ema of the canonical consensus, reported beside (not a third read-out of the task).
Universe: genes count-supported in both tables (count_support True in H1 and in Z4). Masked genes carry no free reading
(they sit at pi) and their canonical value is feature imputation, so they are left out of every association. Canonical on all
genes is a sensitivity only (`all` universe).

Statistic: partial Spearman rho_p of a property with a read-out, ranks residualised on a natural-spline design of covariate
ranks (T4_3 stats.design, the X3/X4/Y5 form). Covariates:
  C    pi_raw, expr (species-balanced log CP10K), detection (species-balanced), log protein length (dropped when the property is
       length itself);
  Ctau C + cell-type tau of the cohort (sensitivity: does a property act beyond having a cell-state pattern);
  Ccls C + TF, mitochondrial, enzyme indicators (sensitivity for dosage, ohnolog, essentiality, age: beyond protein class).
Uncertainty: bootstrap over paralog families (connected components of Compara 110 human paralogs at >= 40% identity, Y5's
pairs), DRAWS draws; two-sided normal p from the bootstrap SD. BH over every (cohort, property, read-out) cell of the primary
scan (C, count-supported universe).

Generality (fixed now): tissue value = median rho_p over the tissue's cohorts (10 tissues: liver, testis, bone marrow, kidney,
islet, eye, cortex, lung, skeletal muscle, tumour myeloid). A property is
  GENERAL     on a read-out if its tissue values share one sign in >= 8 of 10 tissues and BH q < 0.05 in >= 1 cohort of >= 5
              tissues;
  ROBUST      if GENERAL on D_can and on D_free with the same sign (the agreement layer);
  ONE-GATE    if GENERAL on one read-out only (reported with which);
  NEGATIVE    otherwise (reported in the short negative table).
Size: |median of tissue values| reported; below 0.03 is "general but small".

Named genes (O1 rules, H1 columns): count_support, nameable (strict support, no O1 mapping flag, no exact zero, not quiet),
not saturated_flag, sign_D_stable, in BOTH tables, and the sign of D the same in both; listed per cohort, and across cohorts
only if this holds in >= 3 cohorts of >= 2 tissues.

Complex coherence (Complex Portal, CORUM unreachable): per complex with >= 3 universe members in a cohort, the within-complex
SD of the C-residual rank of the read-out; cohort statistic = mean over complexes of (SD_obs - mean SD_null)/SD(SD_null),
null = 500 random sets of the same size drawn within pi x expression quintile strata of the members. Paralog guard: members
with a >= 40% identity paralog inside the same complex reduced to one per paralog family.

Family scans (Pfam via UniProt, Lambert TF DBD families): families with >= 15 universe genes in a cohort; mean C-residual
standardised rank of the read-out in the family; GENERAL/ROBUST as above with the per-cohort t-test p in place of bootstrap.
"""
from __future__ import annotations

DRAWS = 300
MIN_FAMILY = 15
MIN_COMPLEX = 3
NULL_SETS = 500

TISSUE_SHORT = {"liver": "liver", "testis": "testis", "bone marrow": "marrow", "kidney": "kidney", "islet": "islet",
                "eye": "eye", "cortex": "cortex", "lung": "lung", "skeletal muscle": "muscle", "tumour myeloid": "tumour"}

# Phylostrata from Compara 110 orthologs (any ortholog type) of the human gene: oldest clade with an ortholog.
# Higher = older. Eutheria is the floor: every gene in a human+mouse cohort has a mouse ortholog.
STRATA = [  # (stratum value, name, species any of which places the gene there)
    (10, "Opisthokonta", ["saccharomyces_cerevisiae"]),
    (9, "Bilateria", ["drosophila_melanogaster", "caenorhabditis_elegans"]),
    (8, "Olfactores", ["ciona_intestinalis", "ciona_savignyi"]),
    (7, "Vertebrata", ["petromyzon_marinus", "eptatretus_burgeri"]),
    (6, "Gnathostomata", ["callorhinchus_milii"]),
    (5, "Euteleostomi", ["danio_rerio", "lepisosteus_oculatus", "latimeria_chalumnae", "takifugu_rubripes", "oryzias_latipes",
                         "gasterosteus_aculeatus", "erpetoichthys_calabaricus"]),
    (4, "Tetrapoda", ["xenopus_tropicalis", "leptobrachium_leishanense"]),
    (3, "Amniota", ["gallus_gallus", "anolis_carolinensis", "taeniopygia_guttata", "chrysemys_picta_bellii",
                    "crocodylus_porosus", "sphenodon_punctatus", "podarcis_muralis", "meleagris_gallopavo"]),
    (2, "Mammalia", ["ornithorhynchus_anatinus"]),
    (1, "Theria", ["monodelphis_domestica", "sarcophilus_harrisii", "notamacropus_eugenii", "phascolarctos_cinereus",
                   "vombatus_ursinus"]),
]
EUTHERIA = 0

# Continuous properties (rank-based). Sign convention: as named (higher = more of it).
CONTINUOUS = {
    "log_length": "Y5: log protein length (UniProt canonical)",
    "n_pfam": "distinct Pfam families in the protein (UniProt xref)",
    "disorder_frac": "Y5: fraction in UniProt 'Disordered' regions (MobiDB-lite)",
    "coiled_frac": "Y5: coiled-coil fraction",
    "ptm_dens": "Y5: modified residues per 100 aa",
    "paralog_count": "Y5: log1p human paralogs (Compara 110)",
    "gene_age": "phylostratum 0 (Eutheria) .. 10 (Opisthokonta), Compara 110 orthologs",
    "ppi_degree": "Y5: log1p STRING v12 physical degree, score >= 700",
    "huri_degree": "log1p HuRI degree (yeast two-hybrid, abundance-unbiased)",
    "n_complexes": "log1p number of Complex Portal complexes",
    "gtex_tau": "tau across GTEx v8 tissues, log2(median TPM + 1)",
    "depmap_essentiality": "minus mean DepMap 24Q4 CRISPR gene effect (higher = more essential)",
    "loeuf": "gnomAD v4.1 LOEUF (in the H1 table; higher = less constrained)",
    "phaplo": "Collins 2022 pHaplo",
    "ptriplo": "Collins 2022 pTriplo",
}
# Binary properties (rank of 0/1).
BINARY = {
    "ohnolog": "2R ohnolog, OHNOLOGS v2 intermediate",
    "ohnolog_strict": "2R ohnolog, OHNOLOGS v2 strict",
    "complex_member": "member of >= 1 Complex Portal human complex",
    "common_essential": "DepMap 24Q4 inferred common essential",
    "haplo_sensitive": "pHaplo >= 0.86 (Collins 2022 threshold)",
    "triplo_sensitive": "pTriplo >= 0.94 (Collins 2022 threshold)",
    "clingen_hi3": "ClinGen haploinsufficiency score 3",
    "tf": "Lambert 2018 TF",
    "tf_c2h2_krab": "Lambert C2H2 ZF with KRAB",
    "tf_homeodomain": "Lambert DBD Homeodomain (incl. Homeodomain; POU, CUT, Paired)",
    "tf_nuclear_receptor": "Lambert DBD Nuclear receptor",
    "signal_peptide": "Y5: UniProt signal peptide",
    "transmembrane": "Y5: UniProt transmembrane",
    "hpa_secreted": "HPA 25.1 secretome location: secreted (any 'Secreted ...')",
    "hpa_secreted_blood": "HPA secretome: secreted to blood",
    "hpa_loc_nucleus": "HPA main location nucleoplasm, nucleoli, nuclear speckles/bodies/membrane, nucleus",
    "hpa_loc_cytosol": "HPA main location cytosol",
    "hpa_loc_mitochondria": "HPA main location mitochondria",
    "hpa_loc_plasma_membrane": "HPA main location plasma membrane",
    "hpa_loc_vesicles": "HPA main location vesicles",
    "hpa_loc_er_golgi": "HPA main location endoplasmic reticulum or Golgi apparatus",
    "hpa_loc_cytoskeleton": "HPA main location actin filaments, microtubules, intermediate filaments, centrosome, centriolar satellite",
    "hpa_loc_junctions": "HPA main location cell junctions or focal adhesion sites",
    "immune": "UniProt keyword Immunity (KW-0391) or Innate/Adaptive immunity",
    "cd_marker": "HPA protein class CD markers",
    "enzyme": "HPA protein class Enzymes",
    "transporter": "HPA protein class Transporters",
}
KEY_PROPS = ["log_length", "n_pfam", "disorder_frac", "paralog_count", "ohnolog", "gene_age", "ppi_degree", "huri_degree",
             "complex_member", "gtex_tau", "depmap_essentiality", "common_essential", "loeuf", "phaplo", "ptriplo", "tf",
             "tf_c2h2_krab", "tf_homeodomain", "hpa_secreted", "hpa_loc_plasma_membrane", "hpa_loc_mitochondria",
             "hpa_loc_nucleus", "immune"]
CLASS_COVS = ["tf", "hpa_loc_mitochondria", "enzyme"]
