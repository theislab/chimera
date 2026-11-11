"""Cells, lineages and evolutionary contrasts (EXPLORATORY). Definitions, fixed before any association of this task.

Read-outs:
* canonical gate (main model): per-gene tables of H1_harmonise ``results/analysis/H1/genes/<c>.parquet`` (D, S_ema,
  flags); per-(species, type) S_gsk from the canonical forward passes (``build.py``).
* per-gene gate: tables of Z4_free_gate_all ``results/analysis/Z4/h1/genes/<c>.parquet`` (D_free); per-(species,
  type) S_gsk from this task's forward passes of those fits (``forward_free.py``).
* direct baselines beside every Chimera statistic: B1 (pseudobulk profile correlation, tables of H1_harmonise) and a
  per-type direct level gap (|log2 fold change| at the gene's home type, ``build.py`` pseudobulk).

Analyses
A. Home compartments. Each gene's home type = peak type in HUMAN pseudobulk only, log2(1 + CPM), peak minus mean
   over the human types >= 1 log2 and human detection in the peak type >= 10% (the rule of X1_testis_discovery, all
   tissues); types with fewer than 30 human cells are not candidates; otherwise "broad". Types map to cross-tissue
   compartments (COMPARTMENT) and tissue subgroups (SUBGROUP). Matched excess as in X1_testis_discovery (40 nearest
   non-group genes in rank space of C_pat = raw prior, species-balanced expression, detection, tau; 1,000 draws; SD
   units; two-sided p), computed for D (canonical), D_free, S_ema, B1 and the direct level gap.
B. Attribution: for genes of a home group, share of the group's positive and negative canonical margin mass in
   each type divided by the type's weight (the enrichment of X1_testis_discovery); per species the same.
C. Species-specific departures in fits with >= 3 species. For gene g, type k shared by all species (>= 30 cells
   each): dev_s = S_gsk(s) - mean over the other species. A departure of species s: S_gsk(s) is the minimum row,
   dev_s < 0 and |dev_s| >= 2 SD of the other rows (scale-free within a gene and type). Counted on genes
   count-supported in the cohort. Direct analogue: per-type pseudobulk log2(1 + CPM), species s minus the mean of
   the others, departure if |diff| >= 1 and beyond 2 SD of the others' spread (same rule). Divergence times
   (TimeTree, My from human; branch lengths in BRANCH_MY).
D. Named genes: nameable (rule of H1_harmonise) and not saturated in the read-out named, count-supported, sign-stable
   over seeds; flags of O1_orthologs respected through ``nameable``; DLPFC lower tails are reference artifacts by the
   rule of H1_harmonise.

Predictions (before reading any association of this task; what I knew: lineage z of H1_harmonise, stage excess of
X1_testis_discovery, hepatocyte level cost of X2_liver_biology and X4_where_differs, drivers of C1_cohorts (cortex
positive mass in glia and endothelium, negative in excitatory IT neurons), V3 of T11_6_vz_glaucoma (eye drivers follow
species-only types), verdicts of Z4_free_gate_all):
P1. Specialized home genes sit above matched genes on S_ema and D_free in most tissues (>= 5 of 8 thread tissues),
    mostly as pattern gain; broad genes below.
P2. Immune-home genes above matched genes in most tissues on S and D (inflammatory programs recur above,
    H1_harmonise).
P3. Exceptions are level-cost compartments: hepatocytes (liver), and spermatids on D in testis.
P4. B1 shows the same sign for most compartments; the direct level gap is larger where S is negative.
P5. Cortex: glial home genes above neuronal ones on S; excitatory neuron genes carry negative mass.
P6. DLPFC 4 primates: departures per species rise with branch length, human about chimpanzee < macaque <
    marmoset, on Chimera and on the direct analogue; microglia carry the most human departures (Ma et al. 2022).
P7. Eye 5 species: mouse and pig departures above macaques; liver 4 species: zebrafish highest.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results" / "analysis"
OUT = RES / "E3"

SPEC_MIN, DET_MIN, MIN_HUMAN_CELLS = 1.0, 0.10, 30
K_NN, B_DRAWS, SEED = 40, 1000, 0
COV_PAT = ["pi_raw", "expr", "detection", "tau"]
MIN_TYPE_CELLS = 30
DEP_SD = 2.0
DIRECT_LFC = 1.0

# Thread of H1_harmonise: one human+mouse model per tissue or the closest fit holding both; muscle sensitivity
THREAD = {"liver": "liv_hmouse", "testis": "test_hmouse", "bone marrow": "bone_marrow", "kidney": "kidney",
          "islet": "islet", "eye": "vz_hmouse", "cortex": "brain_m1_hmouse", "lung": "lung_healthy"}
SENSITIVITY = {"skeletal muscle": "muscle_ageing"}

# My from human (TimeTree medians)
MY_FROM_HUMAN = {"Chimpanzee": 6.4, "Macaque": 28.8, "Monkey": 28.8, "macaca_mulatta": 28.8, "macaca_fascicularis": 28.8,
                 "Marmoset": 43.0, "Mouse": 87.0, "mouse": 87.0, "mouse_lemur": 74.0, "Lemur": 74.0, "pig": 94.0,
                 "Zebrafish": 429.0, "Human": 0.0, "human": 0.0}
# terminal branch length in the cohort's own tree (My), for species-specific departures
BRANCH_MY = {"brain_dlpfc": {"Human": 6.4, "Chimpanzee": 6.4, "Macaque": 28.8, "Marmoset": 43.0},
             "brain_m1": {"Human": 43.0, "Marmoset": 43.0, "Mouse": 87.0},
             "vz_full": {"human": 28.8, "macaca_fascicularis": 3.7, "macaca_mulatta": 3.7, "pig": 94.0, "mouse": 87.0},
             "liv_4sp": {"Human": 28.8, "Monkey": 28.8, "Mouse": 87.0, "Zebrafish": 429.0},
             "test_3sp": {"Human": 28.8, "Macaque": 28.8, "Mouse": 87.0},
             "bone_marrow": {"human": 74.0, "mouse_lemur": 74.0, "mouse": 87.0},
             "lung_healthy": {"Human": 74.0, "Lemur": 74.0, "Mouse": 87.0},
             "vz_primates": {"human": 28.8, "macaca_fascicularis": 3.7, "macaca_mulatta": 3.7}}

SPECIALISED, IMMUNE, VASCULAR, STROMAL = "specialised", "immune", "vascular", "stromal"
_CTX_EXC = ["L2_3_IT", "L3_5_IT", "L5_IT", "L6_IT", "L5_ET", "L5_6_NP", "L6_CT", "L6B"]
_CTX_INH = ["PVALB", "SST", "SST_CHODL", "VIP", "LAMP5", "SNCG", "ADARB2"]
_CTX = {**{k: (SPECIALISED, "excitatory neuron") for k in _CTX_EXC}, **{k: (SPECIALISED, "inhibitory neuron") for k in _CTX_INH},
        "Astro": (SPECIALISED, "astrocyte"), "Oligo": (SPECIALISED, "oligodendrocyte"), "OPC": (SPECIALISED, "OPC"),
        "Micro": (IMMUNE, "microglia"), "Immune": (IMMUNE, "immune"), "Endo": (VASCULAR, "endothelial"),
        "Mural": (STROMAL, "mural"), "Peri": (STROMAL, "mural"), "VLMC": (STROMAL, "VLMC")}
_LIVER = {"Hepatocytes": (SPECIALISED, "hepatocyte"), "Cholangio": (SPECIALISED, "cholangiocyte"),
          "Endothelial": (VASCULAR, "endothelial"), "fibroblasts": (STROMAL, "stellate/fibroblast"),
          "Macrophages": (IMMUNE, "macrophage"), "Kupffer cells": (IMMUNE, "macrophage"), "B Plasma cells": (IMMUNE, "lymphoid"),
          "NK NKT and T cells": (IMMUNE, "lymphoid"), "HsPCs": (IMMUNE, "progenitor")}
_TESTIS = {"Spermatogonia": (SPECIALISED, "germline: spermatogonia"), "Spermatocytes": (SPECIALISED, "germline: spermatocyte"),
           "spermatids": (SPECIALISED, "germline: spermatid"), "Sertoli": (SPECIALISED, "Sertoli"), "somatic": (STROMAL, "other somatic")}
_BM = {"B": (IMMUNE, "lymphoid"), "T": (IMMUNE, "lymphoid"), "NK": (IMMUNE, "lymphoid"), "plasma": (IMMUNE, "lymphoid"),
       "monocyte": (IMMUNE, "myeloid"), "macrophage": (IMMUNE, "myeloid"), "neutrophil": (IMMUNE, "myeloid"),
       "erythroid": (IMMUNE, "erythroid"), "HSPC": (IMMUNE, "progenitor")}
_KID = {"pt": (SPECIALISED, "proximal tubule"), "dtl": (SPECIALISED, "thin limb"), "atl": (SPECIALISED, "thin limb"),
        "tal": (SPECIALISED, "thick ascending limb"), "dct": (SPECIALISED, "distal convoluted tubule"),
        "cnt": (SPECIALISED, "connecting tubule"), "pc": (SPECIALISED, "principal cell"), "ic": (SPECIALISED, "intercalated cell"),
        "pod": (SPECIALISED, "podocyte"), "pec": (SPECIALISED, "parietal epithelium"), "pape": (SPECIALISED, "papillary epithelium"),
        "ec": (VASCULAR, "endothelial"), "fib": (STROMAL, "fibroblast"), "vsm_p": (STROMAL, "mural"),
        "lymphoid": (IMMUNE, "lymphoid"), "myeloid": (IMMUNE, "myeloid")}
_ISLET = {"alpha": (SPECIALISED, "endocrine: alpha"), "beta": (SPECIALISED, "endocrine: beta"), "delta": (SPECIALISED, "endocrine: delta"),
          "pp": (SPECIALISED, "endocrine: PP"), "ductal": (SPECIALISED, "exocrine: ductal"), "endothelial": (VASCULAR, "endothelial"),
          "stellate": (STROMAL, "stellate"), "immune": (IMMUNE, "immune")}
_EYE = {"trabecular meshwork cell": (SPECIALISED, "outflow: trabecular meshwork"), "ciliary muscle cell": (SPECIALISED, "ciliary muscle"),
        "corneal epithelial cell": (SPECIALISED, "cornea"), "corneal endothelial cell": (SPECIALISED, "cornea"),
        "ciliary body epithelial cell": (SPECIALISED, "ciliary epithelium"), "corneal cell, unresolved": (SPECIALISED, "cornea"),
        "myoepithelial cell": (SPECIALISED, "myoepithelial"), "Schwann cell": (SPECIALISED, "Schwann cell"),
        "melanocyte": (SPECIALISED, "melanocyte"), "neuron": (SPECIALISED, "neuron"), "uveal cell": (STROMAL, "uveal"),
        "canal endothelial cell": (VASCULAR, "outflow: Schlemm canal"), "vascular endothelial cell": (VASCULAR, "endothelial"),
        "fibroblast": (STROMAL, "fibroblast"), "pericyte": (STROMAL, "mural"), "macrophage": (IMMUNE, "myeloid"),
        "mast cell": (IMMUNE, "myeloid"), "B cell": (IMMUNE, "lymphoid"), "NK/T cell": (IMMUNE, "lymphoid")}
_LUNG = {"AT1": (SPECIALISED, "alveolar epithelium: AT1"), "AT2": (SPECIALISED, "alveolar epithelium: AT2"),
         "Aerocyte": (VASCULAR, "capillary: aerocyte"), "Capillary": (VASCULAR, "capillary: general"), "Artery": (VASCULAR, "large vessel"),
         "Vein": (VASCULAR, "large vessel"), "Fibroblast_adventitial": (STROMAL, "fibroblast"), "Fibroblast_alveolar": (STROMAL, "fibroblast"),
         "Myofibroblast": (STROMAL, "fibroblast"), "Pericyte": (STROMAL, "mural"),
         **{k: (IMMUNE, "myeloid") for k in ["Macrophage_alveolar", "Macrophage_interstitial", "Monocyte", "Neutrophil", "cDC", "pDC"]},
         **{k: (IMMUNE, "lymphoid") for k in ["B", "T", "NK"]}}
_MUSCLE = {"MuSC": (SPECIALISED, "muscle stem cell"), "Tenocyte": (STROMAL, "tenocyte"), "FB": (STROMAL, "fibroblast"),
           "SMC": (STROMAL, "mural"), "Pericyte": (STROMAL, "mural"), "ArtEC": (VASCULAR, "endothelial"), "CapEC": (VASCULAR, "endothelial"),
           "VenEC": (VASCULAR, "endothelial"), "Macrophage": (IMMUNE, "myeloid"), "Monocyte": (IMMUNE, "myeloid")}
_TUM = {k: (IMMUNE, "myeloid") for k in ["DC", "Macrophage", "Monocyte", "Neutrophil", "pDC"]} | {k: (IMMUNE, "lymphoid") for k in ["B", "T", "NK"]}


def type_map(tissue: str) -> dict:
    return {"liver": _LIVER, "testis": _TESTIS, "bone marrow": _BM, "kidney": _KID, "islet": _ISLET, "eye": _EYE,
            "cortex": _CTX, "lung": _LUNG, "skeletal muscle": _MUSCLE, "tumour myeloid": _TUM}[tissue]
