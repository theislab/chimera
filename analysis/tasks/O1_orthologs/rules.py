"""Audit of the ortholog pairs of all 14 cohorts against Ensembl Compara 110. Definitions, fixed before any readout or gold is read.

Unit. A pair is (species, human gene, species gene) of a cohort's table, restricted to the cohort's genes and species. A gene of a cohort
is flagged when any of the cohort's non-human pairs is flagged; ``any species of the table`` is a separate sensitivity, because the prior
of a two-species subset of a canonical cohort averages identity over every species of the table.

Sources. Compara 110 ``protein_default`` and ``ncrna_default`` homology dumps, every species' dump (a dump writes each homology once, under
one of the two species). Compara 105 (human dump) and 116 (from Y6_uses) for the same pair. Ensembl 110 GTFs for biotype, CDS exons and names.

Pair flags, none of them uses counts:
  F_absent     pair absent from both Compara 110 dumps, under any homology type;
  F_type       pair present, homology type other than ortholog_one2one;
  F_ncrna      pair is one-to-one only in the ncRNA dump (no protein homology);
  F_biotype    human or species gene not protein_coding in the Ensembl 110 GTF, or absent from it;
  F_lowconf    is_high_confidence = 0 in Compara 110 (protein dump);
  F_nosynt     GOC = 0 and WGA = 0 (both reported), a sub-label of low support; descriptive;
  F_retro      species gene has at most 1 coding exon, human gene at least 3 (Y6_uses PH2);
  F_named_raw  species gene's name differs from the human symbol and another gene of the species in Ensembl 110 carries it (Y6_uses PH5);
  F_named      F_named_raw, unless GOC and WGA are both above 50 (ortholog rule of the pre-specified analysis plan, section 13: synteny outweighs a name);
  F_later      pair not ortholog_one2one in Compara 116 (descriptive: release drift);
  F_earlier    pair not ortholog_one2one in the Compara 105 human dump (descriptive; only where that species' pairs are in that dump).

Tiers, nested, used for every impact table:
  T1 "not one-to-one protein": F_absent | F_type | F_ncrna | F_biotype. Fails the definition the tables state.
  T2 "core": T1 | F_lowconf | F_retro | F_named. Equal to the plan's section 13 rule for the cohorts of C1_cohorts; the wrong named cases of Y6_uses should fall here.
  T3 "core + mapping": T2 | M_wrong_row | M_false_zero (mapping.py). Canonical cohorts only add anything here.
  T2L "core + drift": T2 | F_later (sensitivity: pairs gone by Compara 116; LYZ to 9530003J23Rik falls here and nowhere earlier).
  TZ "zero": T3 | gene with an exact zero count total in any species of the cohort (structural zeros: macaque testis in Y6_uses, zebrafish in four-species liver).
  R2, R3 (post hoc, after reading PODXL and SPINK2 in the named list; no gold seen): R2 = T1 | F_retro | F_named | (F_lowconf and not (GOC >= 50 and WGA >= 50)). R3 = R2 + mapping flags.
Count mapping flags (mapping.py; canonical cohorts, which were joined by gene symbol; raw matrices carry the Ensembl ID where available):
  M_wrong_row  the row picked by the table's species symbol carries an Ensembl ID that differs from the table's species ID and is a
               different gene in Ensembl 110 (its ID is in the 110 GTF);
  M_false_zero the object has zero counts for this pair while the raw matrix holds counts for it: under the table's Ensembl ID (human, mouse,
               testis macaque), under the human gene's one-to-one M. fascicularis gene (liver macaque matrix), under the same name in another case (zebrafish).

Nothing is filtered in the canonical objects. Impact runs drop flagged genes from the tests; the frozen row stays as it is.
"""

from __future__ import annotations

from pathlib import Path

from analysis.common import gene_ids
from analysis.tasks.H1_harmonise.rules import COHORTS as H1_COHORTS
from analysis.tasks.H1_harmonise.rules import load_extra as H1_load_extra

REPO = gene_ids.REPO_ROOT
DATA = REPO / "data" / "analysis" / "O1"
RES = REPO / "results" / "analysis" / "O1"
E110 = REPO / "data" / "analysis" / "ensembl110"
BM = REPO / "data" / "analysis" / "bone_marrow" / "ensembl110"
NU = REPO / "data" / "analysis" / "new_usecases" / "ensembl110"
Y6 = REPO / "data" / "analysis" / "Y6"

ENS = {"mmusculus": "mus_musculus", "mmulatta": "macaca_mulatta", "mfascicularis": "macaca_fascicularis",
       "sscrofa": "sus_scrofa", "drerio": "danio_rerio", "mmurinus": "microcebus_murinus"}
PROTEIN_110 = {"homo_sapiens": E110 / "Compara.110.protein_default.homologies.tsv.gz",
               "mus_musculus": BM / "Compara.110.protein_default.homologies.mus_musculus.tsv.gz",
               "microcebus_murinus": BM / "Compara.110.protein_default.homologies.microcebus_murinus.tsv.gz",
               "danio_rerio": NU / "Compara.110.protein_default.homologies.danio_rerio.tsv.gz",
               "macaca_mulatta": NU / "Compara.110.protein_default.homologies.macaca_mulatta.tsv.gz",
               "sus_scrofa": Y6 / "compara110/sus_scrofa/Compara.110.protein_default.homologies.tsv.gz",
               "macaca_fascicularis": Y6 / "compara110/macaca_fascicularis/Compara.110.protein_default.homologies.tsv.gz"}
NCRNA_110 = {s: DATA / f"compara110_ncrna/{s}/Compara.110.ncrna_default.homologies.tsv.gz" for s in PROTEIN_110}
PROTEIN_105_HUMAN = E110 / "Compara.105.protein_default.homologies.tsv.gz"
GTF_110 = {"hsapiens": E110 / "Homo_sapiens.GRCh38.110.gtf.gz", "mmusculus": BM / "Mus_musculus.GRCm39.110.gtf.gz",
           "mmurinus": BM / "Microcebus_murinus.Mmur_3.0.110.gtf.gz", "mmulatta": NU / "Macaca_mulatta.Mmul_10.110.gtf.gz",
           "drerio": NU / "Danio_rerio.GRCz11.110.gtf.gz", "sscrofa": Y6 / "gtf110/Sus_scrofa.Sscrofa11.1.110.gtf.gz",
           "mfascicularis": Y6 / "gtf110/Macaca_fascicularis.Macaca_fascicularis_6.0.110.gtf.gz"}
TABLES = {"liver": REPO / "data/processed/ortholog_table.csv", "testis": REPO / "data/processed/ortholog_table_testis.csv",
          "van_zyl": REPO / "data/analysis/van_zyl/ortholog_table_van_zyl.csv",
          "bone_marrow": REPO / "data/analysis/bone_marrow/ortholog_table_bone_marrow.csv",
          "kidney": REPO / "data/analysis/disease_kidney/ortholog_table_kidney.csv"}
COHORT_TABLE = {"liv_hmonkey": "liver", "liv_hmouse": "liver", "liv_hzeb": "liver", "liv_4sp": "liver", "test_hmac": "testis", "test_hmouse": "testis",
                "test_3sp": "testis", "bone_marrow": "bone_marrow", "kidney": "kidney", "islet": "testis", "vz_hmouse": "van_zyl", "vz_hmmul": "van_zyl",
                "vz_primates": "van_zyl", "vz_full": "van_zyl"}
COHORTS = {k: dict(h5ad=Path(v["h5ad"]), table=COHORT_TABLE[k], label=v["label"]) for k, v in H1_COHORTS.items()}
# The two eye pair fits that H1_harmonise lists beside the 14 (extra_runs.tsv): flagged too, so its recurrence code can run; reported apart.
_EXTRA = {k: v for k, v in H1_load_extra().items() if k not in COHORTS}
for k, v in _EXTRA.items():
    COHORT_TABLE[k] = "van_zyl"
    COHORTS[k] = dict(h5ad=Path(v["h5ad"]), table="van_zyl", label=v["label"])
OBS_SPECIES = {"human": "hsapiens", "monkey": "mmulatta", "macaque": "mmulatta", "mouse": "mmusculus", "zebrafish": "drerio", "mouse_lemur": "mmurinus",
               "macaca_fascicularis": "mfascicularis", "macaca_mulatta": "mmulatta", "pig": "sscrofa"}
FLAGS_T1 = ["F_absent", "F_type", "F_ncrna", "F_biotype"]
FLAGS_T2 = FLAGS_T1 + ["F_lowconf", "F_retro", "F_named"]
CANONICAL = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]
RAW = {"liver": {"Human": "raw-liver-human-Martin.h5ad", "Monkey": "raw-liver-monkey-Martin.h5ad", "Mouse": "raw-liver-mouse-Martin.h5ad",
                 "Zebrafish": "raw-liver-zebrafish-ggj5.h5ad"},
       "testis": {"Human": "raw-testis-human-Murat.h5ad", "Macaque": "raw-testis-macaque-Murat.h5ad", "Mouse": "raw-testis-mouse-Murat.h5ad"}}
RAW_DIR = {"liver": REPO / "data/raw/camex/1liver/dataset", "testis": REPO / "data/raw/camex/2testis/dataset"}
RAW_ID_COL = {"gene_ids", "gene"}
TIERS = ["T1", "T2", "T2L", "T3", "TZ"]
SEED = 20261102
