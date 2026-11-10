"""B2 Track C: cohorts, golds, lists and strata, fixed before any readout met them. A new cohort joins by adding one row here and rerunning track_c.py."""

from __future__ import annotations

from analysis.tasks.B2_benchmark.rules import COV_C, COV_PAT

P4 = COV_PAT + ["H4_nsp"]
P4P = COV_PAT + ["H4_npri"]
# kind, gold column, covariates. Pattern golds C_pat (+ number of species on Murat), level golds C.
LIVER = [("pattern", "G_H3", COV_PAT), ("level", "G_H1", COV_C), ("level_atlas", "G_lvl_liver", COV_C)]
TESTIS = [("pattern", "G_H4", P4), ("pattern_primates", "G_H4_primates", P4P), ("level", "G_H2", COV_C)]
KIDNEY = [("pattern", "G_kidney", COV_PAT), ("pattern_fine", "G_kidney_fine", COV_PAT), ("pattern_nephron", "G_kidney_nephron", COV_PAT),
          ("level", "G_evee_kidney", COV_C), ("level_atlas", "G_lvl_kidney", COV_C)]
ISLET = [("pattern", "G_pancreas", COV_PAT), ("pattern_noacinar", "G_pancreas_noacinar", COV_PAT), ("pattern_fine", "G_islet_fine", COV_PAT),
         ("pattern_fine_noacinar", "G_islet_noacinar", COV_PAT), ("level_atlas", "G_lvl_pancreas", COV_C)]
MARROW = [("pattern_coe", "G_coe", COV_C)]  # Shay COE on bulk lineages; independent of the training cells. C, as T12.3 E1
COHORTS_C = {
    "liv_hmouse": dict(golds=LIVER, lists=["C1", "C2", "V1", "V2", "V3"], primary=True),
    "liv_hmonkey": dict(golds=LIVER, lists=["C1", "C2", "V1", "V2", "V3"], primary=False),
    "liv_hzeb": dict(golds=LIVER, lists=["C1", "C2", "V1", "V2", "V3"], primary=False),
    "liv_4sp": dict(golds=LIVER, lists=["C1", "C2", "V1", "V2", "V3"], primary=False),
    "test_hmouse": dict(golds=TESTIS, lists=["C1", "C2", "V2", "V3"], primary=True),
    "test_hmac": dict(golds=TESTIS, lists=["C1", "C2", "V2", "V3"], primary=False),
    "test_3sp": dict(golds=TESTIS, lists=["C1", "C2", "V2", "V3"], primary=False),
    "kidney": dict(golds=KIDNEY, lists=["C1", "C2", "V2", "V3"], primary=True),
    "islet": dict(golds=ISLET, lists=["C1", "C2", "V2", "V3"], primary=True),
    "bone_marrow": dict(golds=MARROW, lists=["C1", "C2", "V2", "V3", "Shay", "Shay_matched"], primary=True),
    "vz_hmouse": dict(golds=[], lists=["C1", "C2", "V2", "V3"], primary=True),
    "vz_hmmul": dict(golds=[], lists=["C1", "C2", "V2", "V3"], primary=False),
    "vz_primates": dict(golds=[], lists=["C1", "C2", "V2", "V3"], primary=False),
    "vz_full": dict(golds=[], lists=["C1", "C2", "V2", "V3"], primary=False),
}
# Why these have no pattern or level gold: bone marrow's atlas pair uses the training source on the mouse side (Y1); eye has no mouse atlas in TMS and EVEE has neither.
LIST_DIRECTION = {"C1": "above", "C2": "above", "V1": "below", "V2": "below", "V3": "below", "Shay": "below", "Shay_matched": "below"}
N_BOOT_LISTS = 2000

# Strata, on the cohort's genes. The whole set is the task; count-supported, identity bands (near-1 >= 0.98) and expression quartiles are strata of it.
# Y4's two survivors on tissues not searched (band = tertile by rank over the cohort's genes, as Y4 regions.py; pihi = top quintile of pi_raw).
Y4_REPLICATION = [
    dict(region="expr_T2__tau_T3", cohorts={"kidney": ["level", "level_atlas"], "islet": ["level_atlas"]}, readout="S_g"),
    dict(region="pihi__tau_T2", cohorts={"kidney": ["pattern", "pattern_fine", "pattern_nephron"], "islet": ["pattern", "pattern_fine"], "bone_marrow": ["pattern_coe"]}, readout="S_g"),
]
# Replicates: same sign as discovery, BH q < 0.05 one-sided over the units listed, CI excludes 0, each on all genes of the unit and reported on odd and even autosomes too.
# Called as Y4: differentiates (every Delta_j CI lower > 0 and min Delta_j >= 0.03), baseline ahead (some Delta_j CI upper < 0), matches otherwise; J = B1b, B1c, B2b, B3a, B1z.
