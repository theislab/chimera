"""Z2: is Chimera's latent competent inside a cell-type label, and what sets that? Definitions, fixed before any Z2 metric was computed.

No gene-level gold is read. TS and TMS appear only as cells with author zones (hepatocytes), never as gene scores. No Murat, EVEE, COE or disease object.
Chimera seeds: H1 runs.tsv, kept and healthy. scVI: n_latent 10, n_hidden 128, 100 epochs, the batch key of the Chimera run on the same object, seeds 42-44, saved so it can encode new cells.
Other latents (PCA, Harmony, sysVI, CAMEX, SATURN, scANVI) only where already on disk; they place the cost against mixing, they do not drive any decision.
"""
from __future__ import annotations

from revision.common import gene_ids

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "revision" / "Z2"
RUNS = REPO / "training" / "runs"
K = 30                    # neighbours for rho_self, rho_cross (Y5 KNN_CROSS), kNN classifier and kNN regression
K_MIX = 15                # mixing and label transfer: compute_integration_metrics as in Fig 3 and H1
K_GRAPH = 15              # graph connectivity
PERPLEXITY = 30           # cLISI, as scib (3 x perplexity neighbours)
N_PCS_WITHIN = 10         # within-label variance captured: top PCs of log CP10K on the top N_HVG genes of the label
N_HVG = 2000
MIN_LABEL_CELLS = 100     # a (species, label) enters within-label variance with at least this many cells
MIN_TASK_CELLS = 40       # a species enters a task with at least this many cells, as Y5
MIN_DET = 0.01            # a marker enters a species with detection of at least this

H5AD = {
    "liv_hmonkey": "data/processed/camex_human_monkey.h5ad", "liv_hmouse": "data/processed/camex_human_mouse.h5ad",
    "liv_4sp": "data/processed/camex_pancreas_unified.h5ad", "test_hmac": "data/processed/camex_testis_human_macaque.h5ad",
    "test_hmouse": "data/processed/camex_testis_human_mouse.h5ad", "test_3sp": "data/processed/camex_testis_unified.h5ad",
    "bone_marrow": "data/revision/bone_marrow/v2/bone_marrow_unified.h5ad", "kidney": "data/revision/disease_kidney/kidney_healthy_train.h5ad",
    "islet": "data/revision/disease_islet/islet_healthy_train.h5ad", "vz_hmouse": "data/revision/van_zyl/van_zyl_hmouse.h5ad",
    "vz_hmmul": "data/revision/van_zyl/van_zyl_hmmul.h5ad", "vz_primates": "data/revision/van_zyl/van_zyl_primates.h5ad",
    "vz_full": "data/revision/van_zyl/van_zyl_unified.h5ad",
}
TABLE = {"islet": "testis"}   # islet is on the testis ortholog table (B2 integration rules); others as gene_ids.TABLES
LABEL_KEY = {c: "cell_ontology_class" for c in H5AD}
LABEL_KEY.update({"kidney": "cell_type", "islet": "cell_type", "liv_hmonkey": "cell_type", "liv_hmouse": "cell_type", "liv_4sp": "cell_type"})
COHORTS_CORE = ["liv_hmouse", "liv_4sp", "liv_hmonkey", "test_hmouse", "test_3sp", "test_hmac", "kidney", "islet", "bone_marrow"]
COHORTS_LABELFREE = ["vz_hmouse", "vz_hmmul", "vz_primates", "vz_full"]

# Reference markers, literature, human symbols, mapped through the cohort's ortholog table. Fixed before any metric. up - dn, z within (species, label).
HEP_ZONE = {  # Y5 rules.ZONE["hepatocyte"]: Halpern 2017, Aizarani 2019, MacParland 2018, Ben-Moshe 2019. Pericentral up.
    "up": ["GLUL", "CYP2E1", "CYP1A2", "OAT", "AXIN2", "LECT2", "SLC1A2", "RGN", "CYP27A1", "CYP7A1", "NOTUM", "TBX3", "RNF43", "ZNRF3", "LGR5", "ADH4", "CYP3A4", "ALDH1A1"],
    "dn": ["CPS1", "HAL", "SDS", "ASS1", "ASL", "PCK1", "G6PC1", "ARG1", "ALDOB", "ALB", "PIGR", "HAMP", "APOA1", "FBP1", "GLS2", "CTSC", "TAT", "ITIH4"]}
PT_SEGMENT = {  # S1 up, S3 down: Lake 2023 Nature, Ransick 2019 Dev Cell, Park 2018 Science, Chen 2019 JASN.
    "up": ["SLC5A2", "SLC5A12", "SLC2A2", "PRODH2", "SLC6A19", "SLC4A4", "SLC22A6"],
    "dn": ["SLC7A13", "SLC5A8", "SLC22A7", "ABCC3", "SLC5A1", "AGT", "SLC3A1", "SLC16A9", "ATP11A", "SLC22A13"]}
BETA_STRESS = {  # Unfolded-protein state up, mature function down: Xin 2018 Nat Metab, Sachs 2020 Nat Metab, Camunas-Soler 2020 Cell Metab, Baron 2016 Cell Syst.
    "up": ["HSPA5", "SDF2L1", "HERPUD1", "DDIT3", "PDIA4", "PDIA6", "DNAJB9", "CRELD2", "MANF", "HYOU1", "XBP1", "DNAJC3"],
    "dn": ["MAFA", "UCN3", "G6PC2", "SLC2A2", "HOPX", "SIX3", "GLP1R", "IAPP", "NKX6-1", "PDX1"]}
ERY_MATURATION = {  # Late up, early down: Tusi 2018 Nature, Paulson 2015 Cell Rep, Popescu 2019 Nature.
    "up": ["HBB", "HBA1", "ALAS2", "SLC4A1", "BPGM", "BNIP3L", "EPB42", "SNCA"],
    "dn": ["KIT", "GATA2", "CD34", "MYB", "GFI1B", "CPA3", "MKI67", "TFRC"]}
NEU_MATURATION = {  # Late up, early down: Evrard 2018 Immunity, Xie 2020 Nat Immunol, Ballesteros 2020 Cell.
    "up": ["CXCR2", "MMP9", "CSF3R", "S100A8", "S100A9", "IL1B", "SELL", "CXCL2"],
    "dn": ["MPO", "ELANE", "AZU1", "PRTN3", "CTSG", "CEBPE", "LTF", "CAMP", "MKI67"]}
SCORE_TASKS = {
    "liv_hmouse": {"hep_zone": (["Hepatocytes"], HEP_ZONE)}, "liv_4sp": {"hep_zone": (["Hepatocytes"], HEP_ZONE)}, "liv_hmonkey": {"hep_zone": (["Hepatocytes"], HEP_ZONE)},
    "kidney": {"pt_segment": (["pt"], PT_SEGMENT)},
    "islet": {"beta_stress": (["beta"], BETA_STRESS)},
    "bone_marrow": {"ery_maturation": (["erythroid"], ERY_MATURATION), "neu_maturation": (["neutrophil"], NEU_MATURATION)},
}
# Author-label tasks inside a coarse class: (task, coarse classes in label key, column, {value: rank}). Rank is ordinal when >2 levels.
_STAGES = {"SG": ("Spermatogonia", {"Undifferentiated_Spermatogonia": 1, "Differentiated_Spermatogonia": 2}),
           "SC": ("Spermatocytes", {"Leptotene_Spermatocytes": 1, "Zygotene_Spermatocytes": 2, "Pachytene_Spermatocytes": 3}),
           "SD": ("spermatids", {"Early_Round_spermatids": 1, "Late_Round_spermatids": 2, "Elongated_spermatids": 3})}
_ROUND_ELONG = ("spermatids", {"Early_Round_spermatids": 1, "Late_Round_spermatids": 1, "Round_spermatids": 1, "Elongated_spermatids": 2})
_T = {t: {"stage_" + k: v for k, v in _STAGES.items()} for t in ("test_hmouse", "test_3sp", "test_hmac")}
for t in _T:
    _T[t]["round_vs_elongated"] = _ROUND_ELONG
LABEL_TASKS = {
    **{t: {k: (cl, "cell_type", m) for k, (cl, m) in v.items()} for t, v in _T.items()},
    "kidney": {"pt_state": ("pt", "CellStateLevel2", {"reference": 1, "degenerative": 2}), "tal_state": ("tal", "CellStateLevel2", {"reference": 1, "degenerative": 2}),
               "ec_state": ("ec", "CellStateLevel2", {"reference": 1, "degenerative": 2})},
    "bone_marrow": {"t_cd4_cd8": ("T", "cell_type", {"CD4-positive, alpha-beta T cell": 1, "naive thymus-derived CD4-positive, alpha-beta T cell": 1,
                                                     "CD8-positive, alpha-beta T cell": 2, "naive thymus-derived CD8-positive, alpha-beta T cell": 2}),
                    "mono_progression": ("monocyte", "cell_type", {"promonocyte": 1, "monocyte": 2}),
                    "neu_progression": ("neutrophil", "cell_type", {"granulocytopoietic cell": 1, "granulocyte": 2}),
                    "ery_progression": ("erythroid", "cell_type", {"proerythroblast": 1, "erythroid progenitor cell": 1, "erythroblast": 2, "erythrocyte": 2, "erythroid lineage cell": 2}),
                    "hspc_hsc": ("HSPC", "cell_type", {"hematopoietic stem cell": 1, "common myeloid progenitor": 2, "hematopoietic precursor cell": 3})},
}

# Factorial. One knob at a time from each cohort's canonical recipe; existing CPU seeds of the canonical recipe are the base (rev-<cohort>-s*).
BASE = {
    "liv_hmouse": "--adata data/processed/camex_human_mouse.h5ad --max-epochs 150 --adv-weight 200 --lambda-anchor 0.003 --free-bits 1.0",
    "test_hmouse": "--adata data/processed/camex_testis_human_mouse.h5ad --max-epochs 200 --adv-weight 200 --lambda-anchor 0.003 --free-bits 1.0 --prior-clamp-max 0.85 --celltype-conditional-mmd --celltype-key cell_ontology_class",
}
FACTORIAL_SEEDS = {"liv_hmouse": [42, 43, 44], "test_hmouse": [42, 43]}
# level -> (encoder input, what replaces or adds to the base args). Later flags win in argparse.
LEVELS = {
    "adv100": ("raw", "--adv-weight 100"), "adv50": ("raw", "--adv-weight 50"), "adv0": ("raw", "--adv-weight 0"),
    "mmd10": ("raw", "--lambda-mmd 10"), "mmd0": ("raw", "--lambda-mmd 0"), "adv0mmd0": ("raw", "--adv-weight 0 --lambda-mmd 0"),
    "fb05": ("raw", "--free-bits 0.5"), "fb2": ("raw", "--free-bits 2.0"), "novamp": ("raw", "--no-vamp-prior"),
    "nl20": ("raw", "--n-latent 20"), "nl30": ("raw", "--n-latent 30"),
    "ep300": ("raw", "--max-epochs 300"), "klw5": ("raw", "--warmup-kl-epochs 5"), "zb0": ("raw", "--lambda-z-balance 0"), "res01": ("raw", "--lambda-res 0.01"),
    "cmmd": ("raw", "--celltype-conditional-mmd --celltype-key cell_ontology_class"),   # liver only: label-conditional MMD on the eight labels
    "gmmd": ("raw", ""),                                                                 # testis only: global MMD, base without the conditional flags
    # Mechanism tests, not settings: the encoder reads raw counts (chimera/_module.py forward and _encoder_input). train_patched.py overrides that method from outside the package.
    "logx": ("log1p", ""), "logcp": ("logcp", ""), "logx_adv0mmd0": ("log1p", "--adv-weight 0 --lambda-mmd 0"),
}
# Budget amendment withdrawn by the coordinator (amendments 1, 2): the factorial as committed runs in full. The order in factorial.tsv is priority order, and rows only ever append.
COHORT_LEVELS = {
    "liv_hmouse": ["logx", "logcp", "adv50", "adv0", "mmd10", "mmd0", "cmmd", "fb05", "fb2", "nl20", "ep300", "klw5", "res01", "adv100", "adv0mmd0", "novamp", "nl30", "zb0", "logx_adv0mmd0"],
    "test_hmouse": ["adv100", "adv0", "mmd10", "mmd0", "adv0mmd0", "gmmd", "fb05", "fb2", "nl20", "klw5", "logx", "logx_adv0mmd0"],
}
# Existing runs that already differ by one knob. Single seeds, read as dose evidence only: runs with a healthy Z1 row enter.
EXISTING_GRIDS = {
    "kidney": {"base": ["rev-kid-grid3-s42", "rev-kid-s43", "rev-kid-s44", "rev-kid-s45", "rev-kid-s46"], "adv100": ["rev-kid-grid2-s42"], "adv200": ["rev-kid-grid1-s42"], "mmd100_adv200": ["rev-kid-grid4-s42"],
               "fb15_adv200": ["rev-kid-grid6-s42"], "global_adv200": ["rev-kid-grid8-s42"]},
    "islet": {"base": ["rev-isl-grid14-s42", "rev-isl-s43", "rev-isl-s44", "rev-isl-s45", "rev-isl-s46"], "adv100_mmd50": ["rev-isl-grid2-s42"], "adv200_mmd50": ["rev-isl-grid1-s42"], "adv300_mmd50": ["rev-isl-grid3-s42"],
              "mmd100_a003": ["rev-isl-grid9-s42"], "mmd150_a003": ["rev-isl-grid10-s42"], "mmd200_a003": ["rev-isl-grid11-s42"], "fb15_adv200_mmd50": ["rev-isl-grid6-s42"]},
    "test_hmouse": {"base": [f"rev-test_hmouse-s{s}" for s in range(43, 52)], "global_nolabels": ["rev-ctl-test_hmouse-nolab-s43"]},
}
RUN_TAG = {"liv_hmouse": "liv", "test_hmouse": "tes"}


def run_name(cohort: str, level: str, seed: int) -> str:
    return f"z2-{RUN_TAG[cohort]}-{level}-s{seed}"


def base_seed_runs(cohort: str) -> dict[int, str]:
    """Existing CPU seeds of the canonical recipe."""
    return {s: f"rev-{cohort}-s{s}" for s in (range(45, 52) if cohort == "liv_hmouse" else range(43, 52))}


# Decision rule, fixed before reading. A setting is "clearly better" only if ALL hold on both factorial cohorts, on seeds passing the Z1 health rule (at least 2 per level):
THRESH = dict(
    min_healthy=2,
    gain_liver_tms_auroc=0.10,    # liver TMS zone AUROC through the encoder, mean over seeds, above base by at least this
    gain_liver_rho_self=0.10,     # and liver hepatocyte rho_self, all markers
    max_loss_testis_stage=0.02,   # testis fine-stage AUROC (human in-model, macaque through encoder) not below base by more than this
    mix_ratio=0.90,               # mixing entropy at least this share of base
    max_loss_label_transfer=0.02, # label transfer not below base by more than this
    max_loss_rho_DS=0.05,         # rho(D, S_ema) on count-supported genes not below base by more than this, raw and partial
    max_loss_consensus=0.05,      # Spearman of D with canonical consensus on count-supported genes not below the canonical leave-one-out median by more than this
)
