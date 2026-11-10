"""B2 Tracks B and C of the gene-level benchmark, tools, overview: definitions, fixed before any run is scored.

Design in README.md, predictions in PREDICTIONS.md. Nothing in the chimera package changes. Chimera is retrained with the
cohort's canonical recipe; the only recipe change is the no-label form on cohorts whose recipe reads labels, the form the
repository already uses for controls (T5.4b: cell-type-conditional MMD removed, same MMD weight, then global).
"""

from __future__ import annotations

import zlib

from analysis.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "analysis" / "B2"
H5AD_DIR = REPO / "data" / "analysis" / "B2" / "h5ad"
TASK_DIR = REPO / "analysis" / "tasks" / "B2_benchmark"
T42_KEY = "T4_2"

# ---------------------------------------------------------------------------
# Track B cohorts. Recipes are the canonical ones (07_EXPERIMENT_MATRIX, T3.2, P13), identical to Track A's.
# ---------------------------------------------------------------------------
COHORTS = {
    "liv_hmouse": dict(
        h5ad="data/processed/camex_human_mouse.h5ad", gene_table="liv_hmouse", label_key="cell_type", coarse_map=False,
        library_key="batch", donor_key=None, batch_key="batch", model_key=None, cpus=8, mem="16G",
        recipe="--config A --max-epochs 150 --adv-weight 200 --free-bits 1.0",
        nolabel_recipe=None,  # the recipe reads no label: every label setting uses the unperturbed fits
        ref_runs={45: "rev-liv_hmouse-s45", 46: "rev-liv_hmouse-s46", 47: "rev-liv_hmouse-s47",
                  48: "rev-liv_hmouse-s48", 49: "rev-liv_hmouse-s49", 50: "rev-liv_hmouse-s50"}),
    "test_hmouse": dict(
        h5ad="data/processed/camex_testis_human_mouse.h5ad", gene_table="test_hmouse", label_key="cell_type", coarse_map=True,
        library_key="sample_batch", donor_key="sample_batch", batch_key="batch", model_key="cell_ontology_class", cpus=12, mem="32G",
        recipe="--config A --max-epochs 200 --adv-weight 200 --free-bits 1.0 --celltype-conditional-mmd "
               "--celltype-key cell_ontology_class --prior-clamp-max 0.85",
        nolabel_recipe="--config A --max-epochs 200 --adv-weight 200 --free-bits 1.0 --prior-clamp-max 0.85",
        ref_runs={45: "rev-test_hmouse-s45", 46: "rev-test_hmouse-s46", 47: "rev-test_hmouse-s47",
                  48: "rev-test_hmouse-s48", 49: "rev-test_hmouse-s49", 50: "rev-test_hmouse-s50"}),
    "kidney": dict(
        h5ad="data/analysis/disease_kidney/kidney_healthy_train.h5ad", gene_table="kidney", label_key="cell_ontology_class",
        coarse_map=False, library_key="sample_batch", donor_key="donor", batch_key="batch", model_key="cell_ontology_class",
        cpus=16, mem="64G",
        recipe="--config A --max-epochs 200 --batch-key batch --adv-weight 300 --lambda-anchor 0.003 --lambda-mmd 50 "
               "--free-bits 1.0 --celltype-conditional-mmd --celltype-key cell_ontology_class --prior-clamp-max 0.85",
        nolabel_recipe="--config A --max-epochs 200 --batch-key batch --adv-weight 300 --lambda-anchor 0.003 --lambda-mmd 50 "
                       "--free-bits 1.0 --prior-clamp-max 0.85",
        ref_runs={42: "rev-kid-grid3-s42", 43: "rev-kid-s43", 44: "rev-kid-s44", 45: "rev-kid-s45", 46: "rev-kid-s46"}),
}
# Second wave of Track B and the tools of Track C: objects can be built, fits are listed only if the wave is launched.
WAVE2 = {
    "islet": dict(
        h5ad="data/analysis/disease_islet/islet_healthy_train.h5ad", gene_table="testis", label_key="cell_ontology_class", coarse_map=False,
        library_key="sample_batch", donor_key="sample_batch", batch_key="batch", model_key="cell_ontology_class", cpus=12, mem="32G",
        recipe=None, nolabel_recipe=None, ref_runs={}),
}
ALL_COHORTS = {**COHORTS, **WAVE2}
SPECIES = ("Human", "Mouse")
# Cell-type maps. Testis model labels (cell_ontology_class) follow the T4.2 coarse labels through this map.
TESTIS_MODEL_PARENT = {"SG": "Spermatogonia", "SC": "Spermatocytes", "rSD": "spermatids", "eSD": "spermatids",
                       "Sertoli": "Sertoli", "Other_somatic": "somatic"}
COARSE = {  # the coarse labels of X6 on liver and testis; kidney by nephron segment group, written from names only
    "liv_hmouse": {"Hepatocytes": "epithelial", "Cholangio": "epithelial", "Endothelial": "endothelial",
                   "fibroblasts": "stromal_immune", "Kupffer cells": "stromal_immune", "Macrophages": "stromal_immune",
                   "B Plasma cells": "stromal_immune", "HsPCs": "stromal_immune"},
    "test_hmouse": {"SG": "germ_early", "SC": "germ_early", "rSD": "germ_late", "eSD": "germ_late",
                    "Sertoli": "somatic", "Other_somatic": "somatic"},
    "kidney": {"pt": "proximal", "dtl": "loop", "atl": "loop", "tal": "loop", "dct": "distal", "cnt": "distal", "pc": "distal",
               "ic": "distal", "pod": "glomerular", "pec": "glomerular", "pape": "glomerular", "ec": "stroma_vascular", "fib": "stroma_vascular",
               "vsm_p": "stroma_vascular", "lymphoid": "immune", "myeloid": "immune"},
}

# ---------------------------------------------------------------------------
# Settings. reps: three data replicates, each with its own perturbation draw and training seed.
# ---------------------------------------------------------------------------
REPS = [0, 1, 2]
FIT_SEEDS = {0: 142, 1: 143, 2: 144}
SETTINGS = {
    # family label: what changes is the cell annotation only; counts stay
    "full": dict(family="control", kind="none"),
    "relabel10": dict(family="label", kind="relabel", q=0.10),
    "relabel25": dict(family="label", kind="relabel", q=0.25),
    "relabel50": dict(family="label", kind="relabel", q=0.50),
    "relabel25_near": dict(family="label", kind="relabel_near", q=0.25),  # confusion toward the most similar type (X6 disclosure)
    "coarse": dict(family="label", kind="coarse"),
    "fine": dict(family="label", kind="fine"),
    "species_diff": dict(family="label", kind="species_diff"),
    "nolabel": dict(family="label", kind="nolabel", r_min=0.3),
    "nolabel_r02": dict(family="label", kind="nolabel", r_min=0.2, chimera_from="nolabel"),
    # composition: cells are removed
    "type_missing": dict(family="composition", kind="type_missing"),
    "rare_types": dict(family="composition", kind="rare_types"),
    "size_1to1": dict(family="composition", kind="size", ratio=1),
    "size_1to5": dict(family="composition", kind="size", ratio=5),
    "size_1to20": dict(family="composition", kind="size", ratio=20),
    # sampling
    "cells50": dict(family="sampling", kind="cells", frac=0.50),
    "cells25": dict(family="sampling", kind="cells", frac=0.25),
    "cells10": dict(family="sampling", kind="cells", frac=0.10),
    "depth50": dict(family="sampling", kind="depth", frac=0.50),
    "depth25": dict(family="sampling", kind="depth", frac=0.25),
    "few_donors": dict(family="sampling", kind="few_donors"),
    # depth law arm: the same objects as depth50 and depth25 (object_of), recipe with lambda_anchor scaled by analysis/common/depth_anchor.py; added at the coordinator's
    # instruction before any of these rows ran. Direct methods read the same counts, so only Chimera is new. Reported beside the recipe-unchanged rows.
    "depth50_law": dict(family="sampling", kind="depth", frac=0.50, object_of="depth50", depth_law=True),
    "depth25_law": dict(family="sampling", kind="depth", frac=0.25, object_of="depth25", depth_law=True),
    # technical
    "extra_batch": dict(family="technical", kind="extra_batch"),
    "chemistry": dict(family="technical", kind="chemistry"),
}
# Main settings of the tools on Track B (plus full, ref): one per kind of problem at the middle severity.
TOOL_SETTINGS = ["full", "relabel25", "coarse", "nolabel", "type_missing", "size_1to5", "cells25", "depth25", "extra_batch", "chemistry"]
TOOL_REF_SEEDS = {0: 45, 1: 46, 2: 47}

# Parameters of the perturbations, fixed now.
RARE_CELLS = 40            # rare_types: every type but the two largest shared ones is cut to this many cells per species
FEW_DONORS = {"test_hmouse": 1, "kidney": 2}  # donors kept per species; liver has one library per species: not applicable
FINE_K = 3                 # fine: each (species, type) split in 3 by k-means on 20 PCs of 2,000 HVGs, if >= FINE_MIN cells in both species
FINE_MIN = 150
XB_SIGMA = 0.5             # extra batch: half of each species' cells get gene factor exp(-|N(0, XB_SIGMA)|) times XB_DEPTH, per species
XB_DEPTH = 0.7
CHEM_CLIP = 16.0           # chemistry: empirical gene-wise capture profile, see perturb.chem_profile
MIN_LIB = 1                # a cell left with no count is dropped after thinning, and counted
LEIDEN = dict(n_top_genes=2000, n_pcs=30, n_neighbors=15, resolution=1.0, min_cells=20, n_markers=50, marker_det=0.10)  # X6
MIN_CELLS_LABEL = 30       # T4.2
MIN_CELLS_CLUSTER = 20     # T4.2

# Health (discard rule, the same as Track A): a fit is discarded if alpha, z or S_g is non-finite or fewer than 10 of 10 latent
# dims have Var(z_mean) >= 0.01. Mixing is recorded, never used to discard. Discarded fits are counted in one Methods line.
# A setting's readout needs at least 2 healthy fits of 3; otherwise the method has no value in that task and ranks last in it.
ACTIVE_VAR = 0.01
N_LATENT = 10
MIN_REPS = 2

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
COV_C = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
COV_PAT = COV_C + ["tau_shared"]
# Track B golds. 05 sections 3-4. Covariates are the unperturbed cohort's in every setting, so the estimand does not move.
B_GOLDS = {
    "liv_hmouse": {"pattern": ("G_H3", COV_PAT), "level": ("G_H1", COV_C)},
    "test_hmouse": {"pattern": ("G_H4", COV_PAT + ["H4_nsp"]), "level": ("G_H2", COV_C)},
    "kidney": {"pattern": ("G_kidney", COV_PAT), "pattern_fine": ("G_kidney_fine", COV_PAT), "level": ("G_evee_kidney", COV_C)},
}
# Method lists. Oriented so that higher = more conserved.
CHIMERA = ["S_g", "D", "B1z"]
CHIMERA_EXPL = ["S_fwd", "D_raw"]
DIRECT = ["B1", "B1b", "B1c", "B2b", "B3a", "Sdir", "Lfc"]
STATIC = ["pi", "dNdS", "LOEUF"]
TOOLS = ["SATURN", "CAMEX", "GeneSpectra", "CACIMAR", "SAMap"]
CONTROLS = ["expr", "random"]
# Free per-gene gate (Z3, Z4; added at the coordinator's instruction after Track B and C values were seen): Delta-alpha of the free-gate fits where they exist (Track A arms of B1,
# Track C cohorts with Z4 fits). No free-gate fit is trained on Track B objects, so it is absent there, not failed.
CHIMERA_FREE = ["D_free"]
OVERVIEW_METHODS = CHIMERA + CHIMERA_FREE + DIRECT + STATIC + TOOLS
J_H2H = ["B1b", "B1c", "B2b", "B3a", "Sdir"]  # direct comparators; B1z is Chimera's own latent and is compared separately
N_DRAWS_B = 2_000          # as X6
N_DRAWS_C = 10_000         # as T4.3 and Y1
BOOT_SEED = 20261102
SPLINE_MIN = 300           # below this a stratum uses intercept + linear ranks (Y4)
STRATUM_MIN = 60           # below this a stratum is not scored (Y4)
ID_BANDS = [("id_lt70", 0, 0.70), ("id_70_85", 0.70, 0.85), ("id_85_95", 0.85, 0.95), ("id_95_98", 0.95, 0.98), ("id_ge98", 0.98, 1.01)]
EXPR_BANDS = 4             # quartiles of species-balanced expression


def useed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def run_name(cohort: str, setting: str, rep: int) -> str:
    return f"b2-{cohort}-{setting}-r{rep}"


def uses_labels(cohort: str) -> bool:
    return COHORTS[cohort]["model_key"] is not None


def grid(cohorts=None) -> list[dict]:
    """One row per (cohort, setting, replicate). fit=1: Chimera is trained on this object. fit=0: the counts and the recipe's
    input are unchanged by the setting, so the setting reuses the fits named in chimera_from and only direct scores are new."""
    rows = []
    for c in (cohorts or COHORTS):
        for s, cfg in SETTINGS.items():
            if cfg["kind"] == "few_donors" and c not in FEW_DONORS:
                continue
            for r in REPS:
                fit, src = 1, ""
                if cfg["family"] == "label" and not uses_labels(c):
                    fit, src = 0, run_name(c, "full", r)
                elif "chimera_from" in cfg:
                    fit, src = 0, run_name(c, cfg["chimera_from"], r)
                rows.append(dict(cohort=c, setting=s, family=cfg["family"], rep=r, seed=FIT_SEEDS[r], run=run_name(c, s, r),
                                 fit=fit, chimera_from=src))
    return rows


def ref_rows(cohorts=None) -> list[dict]:
    """Reference fits: existing unaltered fits of the canonical recipe, scored by the same code, never changed."""
    rows = []
    for c in (cohorts or COHORTS):
        for s, run in COHORTS[c]["ref_runs"].items():
            rows.append(dict(cohort=c, setting="ref", family="reference", rep=s, seed=s, run=run, fit=0, chimera_from=run))
    return rows
