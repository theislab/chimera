"""X2 (EXPLORATORY): gene sets and gene lists for the liver biology analysis, fixed before any D is read.

Zonation sets are written from the published papers only (human symbols, 1:1 orthologs
taken as they appear in the liver ortholog table; genes absent from the 1:1 vocabulary are
reported as missing, never replaced):

* Halpern et al. 2017, Nature 542:352 (mouse; smFISH landmark panel and named zonated genes):
  pericentral landmarks Glul, Cyp2e1; periportal landmarks Ass1, Asl, Alb, Cyp2f2; named
  pericentral Oat, Axin2, Cyp7a1, Hsd3b7; periportal Arg1, Igf1, Sox9; mid-lobule Hamp,
  Igfbp2, Cyp8b1; periportal gluconeogenesis and ureagenesis, pericentral glycolysis and
  xenobiotic metabolism (pathway statements, not tested as sets here).
* Aizarani et al. 2019, Nature 572:199 (human): periportal ALB, PCK1; central / midzonal
  CYP1A2, CYP2E1; central GLUL. Reports only limited mouse-human conservation of zonation.
* Ben-Moshe & Itzkovitz 2019, Nat Rev Gastroenterol Hepatol 16:395 (review of canonical
  zonated genes): pericentral GLUL, CYP2E1, CYP1A2, OAT, RHBG, AXIN2, LGR5, CYP7A1, LECT2, TBX3;
  periportal CPS1, ASS1, ASL, ARG1, HAL, SDS, PCK1, G6PC1, CYP2F2 (human CYP2F1), ALB.

ZONE_SCORE_* are the landmarks used to place hepatocytes on the porto-central axis (the
Halpern smFISH panel plus the Aizarani human landmarks). Sets that are tested for D exclude
nothing; tests of zone-resolved quantities report results with and without the scoring
landmarks.

The plan's reserved sets (H5/H6: Eisenberg housekeeping; HGNC ribosomal, proteasome,
spliceosome; Hallmark xenobiotic, bile acid, interferon, inflammatory; GO extracellular
region) are never formed or tested in X2.
"""

from __future__ import annotations

# ---- zonation (literature-defined; fixed before D was read) --------------------------------
# Amendment 1 (before any D was read): six of the eight scoring landmarks (GLUL, CYP2E1,
# CYP1A2, ASS1, ALB, CYP2F1) are absent from the 1:1 four-species liver vocabulary, so
# hepatocytes are placed on the porto-central axis from the raw CAMEX full-transcriptome
# objects (data/raw/camex/1liver/dataset/raw-liver-*.h5ad, same cells), with species-native
# symbols. Only landmarks OUTSIDE the vocabulary are used for the score, so no gene tested
# below also defines the zones. Zebrafish has no albumin, ASS1 or CYP2F ortholog in its raw
# object; its score is pericentral-only (glula, glulb). Secondary score adds the canonical
# review markers outside the vocabulary (PC LECT2; PP SDS, G6PC).
ZONE_SCORE_PC = ["GLUL", "CYP2E1", "CYP1A2"]
ZONE_SCORE_PP = ["ASS1", "ALB", "CYP2F1"]
ZONE_SCORE_NATIVE = {
    "Human": {"PC": ["GLUL", "CYP2E1", "CYP1A2"], "PP": ["ALB", "ASS1", "CYP2F1"]},
    "Monkey": {"PC": ["GLUL", "CYP2E1", "CYP1A2"], "PP": ["ALB", "ASS1", "CYP2F1"]},
    "Mouse": {"PC": ["Glul", "Cyp2e1", "Cyp1a2"], "PP": ["Alb", "Ass1", "Cyp2f2"]},
    "Zebrafish": {"PC": ["glula", "glulb"], "PP": []},
}
ZONE_SCORE_NATIVE_EXTRA = {
    "Human": {"PC": ["LECT2"], "PP": ["SDS", "G6PC"]},
    "Monkey": {"PC": ["LECT2"], "PP": ["SDS", "G6PC"]},
    "Mouse": {"PC": ["Lect2"], "PP": ["Sds", "G6pc"]},
    "Zebrafish": {"PC": [], "PP": []},
}
RAW_LIVER = {"Human": "raw-liver-human-Martin.h5ad", "Monkey": "raw-liver-monkey-Martin.h5ad",
             "Mouse": "raw-liver-mouse-Martin.h5ad", "Zebrafish": "raw-liver-zebrafish-ggj5.h5ad"}

ZONATION_SETS = {
    # Halpern 2017 + Aizarani 2019 named genes
    "PC_named": ["GLUL", "CYP2E1", "CYP1A2", "OAT", "AXIN2", "CYP7A1", "HSD3B7"],
    "PP_named": ["ASS1", "ASL", "ALB", "CYP2F1", "ARG1", "IGF1", "SOX9", "PCK1"],
    "mid_named": ["HAMP", "IGFBP2", "CYP8B1"],
    # canonical review list (Ben-Moshe & Itzkovitz 2019)
    "PC_canonical": ["GLUL", "CYP2E1", "CYP1A2", "OAT", "RHBG", "AXIN2", "LGR5", "CYP7A1", "LECT2", "TBX3"],
    "PP_canonical": ["CPS1", "ASS1", "ASL", "ARG1", "HAL", "SDS", "PCK1", "G6PC1", "CYP2F1", "ALB"],
}
ZONATION_SOURCES = {
    "PC_named": "Halpern 2017 (Glul, Cyp2e1 landmarks; Oat, Axin2, Cyp7a1, Hsd3b7 named); Aizarani 2019 (CYP1A2)",
    "PP_named": "Halpern 2017 (Ass1, Asl, Alb, Cyp2f2 landmarks; Arg1, Igf1, Sox9 named); Aizarani 2019 (PCK1)",
    "mid_named": "Halpern 2017 (Hamp, Igfbp2, Cyp8b1 mid-lobule)",
    "PC_canonical": "Ben-Moshe & Itzkovitz 2019 review",
    "PP_canonical": "Ben-Moshe & Itzkovitz 2019 review",
}

# ---- named genes on the submitted figures --------------------------------------------------
# Fig 4c themes, verbatim from notebooks/_alpha_biology_simpler_builder.py (H_THEMES)
FIG4C_THEMES = {
    "Urea cycle / GNG": ["CPS1", "ARG1", "ASL", "OTC", "PCK1", "PAH"],
    "mtDNA OXPHOS": ["MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND2", "MT-ND4", "MT-ATP6"],
    "Liver support": ["SELP", "SELE", "VCAM1", "F3", "CR2", "CD40LG", "COL17A1", "PDGFRB", "KIT", "C1QC"],
    "Housekeeping": ["DDX4", "RPL3", "EEF1A1", "EEF2", "ACTB", "TUBA1A", "COX4I1", "NDUFA1", "HNF4A"],
}
# urea-cycle enzymes and transporters (T2.6 list)
UREA_CYCLE = ["CPS1", "OTC", "ASS1", "ASL", "ARG1", "NAGS", "ARG2", "SLC25A15", "SLC25A13", "SLC25A2"]
# extra named genes in T2.5 landmarks and Fig 7 (CPS1)
EXTRA_NAMED = ["ASS1", "NAGS", "ALB", "APOA1"]

# ---- divergent-but-protein-conserved selection (fixed before D was read) --------------------
# high prior: prior_raw in the top third of the cohort; D in the bottom 5 %; margin support
# (T2.5 flag); sign of (alpha - prior) negative in every seed of the consensus.
DIVCONS_PRIOR_Q = 2 / 3
DIVCONS_D_Q = 0.05
