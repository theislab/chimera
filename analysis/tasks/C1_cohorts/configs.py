"""C1: one configuration per cohort. Every cohort goes through the same `build.py`, grid rule, readout and H1 schema.

A *source* is what is read from disk: per species, the files, the label column and map, the unit (donor) column and
filters. A *cohort* is a model: which species of a source enter, with which ortholog table. Pairs are either subsets of
a built cohort, as the CAMEX splits (`training/build_subset.py` unchanged: prior and features carried over from the
parent; p_null recomputed), or cohorts of their own with their own table where the parent's vocabulary would cut genes
that the pair's use needs (brain_m1_hmouse: the marmoset list has 14,409 genes, and the pair feeds the Alzheimer case).

Reasons for filters are in the same words across cohorts (checklist item 10), written to `filters.csv` by build.py.
"""

from __future__ import annotations

from pathlib import Path

from analysis.common import gene_ids

ROOT = gene_ids.REPO_ROOT
TASK = ROOT / "analysis" / "tasks" / "C1_cohorts"
MAPS = TASK / "maps"
DATA = ROOT / "data" / "analysis" / "new_usecases"
BUILD = DATA / "c1_build"
RES = ROOT / "results" / "analysis" / "C1"

# Near the size of the three-species testis cohort, the size all recipes were tuned at (T12.2b).
MAX_CELLS = 28854
MIN_GENES = 200
MIN_CELLS_MODEL = 50  # checklist item 5
SEED = 20261102

DL = DATA / "brain_dlpfc"
M1 = DATA / "brain_m1"
LH = DATA / "lung_healthy"
MU = DATA / "muscle_ageing"

SOURCES = {
    "dlpfc": dict(
        reference="Ma 2022 Science, CELLxGENE collection e1fa9900; 10x 3' v3, dorsolateral prefrontal cortex",
        map=MAPS / "brain_dlpfc.tsv", label_col="subclass", unit_col="donor_id",
        species={
            "hsapiens": dict(name="Human", files=[DL / "992791d9-4ba2-4ca6-94ef-1013f73e4696.h5ad"], source="Ma2022"),
            "ptroglodytes": dict(name="Chimpanzee", files=[DL / "bd8e92c7-d8b4-467f-8715-bc9447b441d9.h5ad"], source="Ma2022"),
            "mmulatta": dict(name="Macaque", files=[DL / "4ff87ad3-c44a-4b72-acf0-291999d6b675.h5ad"], source="Ma2022"),
            "cjacchus": dict(name="Marmoset", files=[DL / "5187050e-ae8a-4712-b056-4b4965947103.h5ad"], source="Ma2022"),
        },
    ),
    "m1": dict(
        reference="Bakken 2021 Nature, CELLxGENE collection 367d95c0; 10x 3' v3, primary motor cortex; files are the authors' subsampled glutamatergic, GABAergic and non-neuronal sets",
        map=MAPS / "brain_m1.tsv", label_col="BICCN_subclass_label", unit_col="donor_id",
        species={
            "hsapiens": dict(name="Human", files=[M1 / f"{x}.h5ad" for x in ("a4f4fd2c-56cd-467f-968f-1b929a79d97f", "f0483022-8afa-4c59-a431-b613f30b212d", "dca77ed0-1f2f-474e-bec7-2f885968c73a")], source="Bakken2021"),
            "cjacchus": dict(name="Marmoset", files=[M1 / f"{x}.h5ad" for x in ("43c65fc1-a90b-42e9-a81c-06cb273f3c2d", "92a1389d-bb88-4ad0-a09a-70c4727e2f44", "a61926b3-82be-4848-b209-b3bc5643240f")], source="Bakken2021"),
            "mmusculus": dict(name="Mouse", files=[M1 / f"{x}.h5ad" for x in ("4f4c1d87-fda8-486b-86b3-acd56b46ab05", "6522326d-cf23-4e0a-af4b-a545b9e11924", "dbeeface-ee53-4a0a-b13d-eac5428e9b6b")], source="Bakken2021"),
        },
    ),
    "lung": dict(
        reference="Travaglini 2020 (human), Tabula Microcebus 2025 (lemur), Tabula Muris Senis 2020 (mouse); 10x 3' v2 in all three",
        map=MAPS / "lung_healthy.tsv", unit_col="donor_id",
        species={
            "hsapiens": dict(name="Human", files=[LH / "f5568ea3-c249-4e4e-91f8-46abc30a5612.h5ad"], source="Travaglini2020", label_col="free_annotation",
                             filters={"tissue": ["lung"]}, filter_reason="blood cells of the same donors are a different tissue"),
            "mmurinus": dict(name="Lemur", files=[LH / "af24ba8a-0c53-4aa6-b7c8-cf1bfec15b41.h5ad"], source="TabulaMicrocebus", label_col="free_annotation_v1",
                             filters={"assay": ["10x 3' v2"]}, filter_reason="Smart-seq2 read counts are not on the 3' UMI count scale"),
            "mmusculus": dict(name="Mouse", files=[LH / "e67a08e2-96fc-4244-b8a3-fca092b22f77.h5ad"], source="TabulaMurisSenis", label_col="free_annotation"),
        },
    ),
}

COHORTS = {
    # K = 4: the question of near-identical proteins. Pairs are CAMEX-style subsets below.
    # esm_species: gene features from the species with SATURN embeddings; chimpanzee and marmoset embeddings cost 37.6 M residues on CPU,
    # so they enter as a sensitivity when ready (`*_esmfull`), plan amendment.
    "brain_dlpfc": dict(source="dlpfc", species=["hsapiens", "ptroglodytes", "mmulatta", "cjacchus"], esm_species=["hsapiens", "mmulatta"], tissue="cortex", label="DLPFC 4 primates"),
    # K = 3: human, marmoset, mouse in one region.
    "brain_m1": dict(source="m1", species=["hsapiens", "cjacchus", "mmusculus"], esm_species=["hsapiens", "mmusculus"], tissue="cortex", label="M1 human, marmoset, mouse"),
    # K = 2 with its own table: human-mouse identity as the prior, the vocabulary of the Alzheimer case.
    "brain_m1_hmouse": dict(source="m1", species=["hsapiens", "mmusculus"], tissue="cortex", label="M1 human, mouse"),
    "lung_healthy": dict(source="lung", species=["hsapiens", "mmurinus", "mmusculus"], tissue="lung", label="Lung human, lemur, mouse"),
}

# tag -> parent, species kept (obs names).
SUBSETS = {
    "brain_dlpfc_hchimp": dict(parent="brain_dlpfc", species=["Human", "Chimpanzee"], tissue="cortex", label="DLPFC human, chimpanzee"),
    "brain_dlpfc_hmac": dict(parent="brain_dlpfc", species=["Human", "Macaque"], tissue="cortex", label="DLPFC human, macaque"),
    "brain_dlpfc_hmarm": dict(parent="brain_dlpfc", species=["Human", "Marmoset"], tissue="cortex", label="DLPFC human, marmoset"),
}


def h5ad(cohort: str) -> Path:
    return BUILD / "objects" / f"{cohort}.h5ad"


def table_path(cohort: str) -> Path:
    if cohort in SUBSETS:
        cohort = SUBSETS[cohort]["parent"]
    return BUILD / "objects" / f"ortholog_table_{cohort}.csv"


def scaler_path(cohort: str) -> Path:
    return BUILD / "objects" / f"gin_scaler_{cohort}.pkl"


# GEO cohort prepared by prepare_tumour.py; symbols mapped to Ensembl there.
SOURCES["zilionis"] = dict(
    reference="Zilionis 2019 Immunity, GSE127465, raw counts per library, inDrop; tumour-infiltrating immune cells, human NSCLC and KP mouse tumours",
    map=MAPS / "tumour_zilionis.tsv", label_col="label", unit_col="donor_id",
    species={"hsapiens": dict(name="Human", files=[DATA / "tumour_zilionis" / "prepared_human.h5ad"], source="Zilionis2019"),
             "mmusculus": dict(name="Mouse", files=[DATA / "tumour_zilionis" / "prepared_mouse.h5ad"], source="Zilionis2019")})
COHORTS["tumour_zilionis"] = dict(source="zilionis", species=["hsapiens", "mmusculus"], tissue="tumour myeloid", label="Tumour human, mouse")


# Reference only, never in training: Jorstad 2023 middle temporal gyrus; 10x 3' v3 cells. Chimpanzee and gorilla are not on CELLxGENE.
MT = DATA / "brain_mtg"
SOURCES["mtg"] = dict(
    reference="Jorstad 2023 Science, CELLxGENE collection 4dca242c, reference only",
    map=MAPS / "brain_mtg.tsv", label_col="Subclass", unit_col="donor_id",
    species={"hsapiens": dict(name="Human", files=[MT / "224db6b4-2895-4c4e-8503-bb647fae9b29.h5ad"], source="Jorstad2023", filters={"assay": ["10x 3' v3"]}, filter_reason="Smart-seq reads are not on the UMI count scale"),
             "mmulatta": dict(name="Macaque", files=[MT / "261dd26f-1507-41a5-94e5-b96903de74a9.h5ad"], source="Jorstad2023"),
             "cjacchus": dict(name="Marmoset", files=[MT / "c556b8e8-e348-49b9-b4da-08b05cd43350.h5ad"], source="Jorstad2023")})


# Model on young donors of the cell fraction, one chemistry per species; DE (prepare_de.py) on other donors, cell fraction, plan section 13f.
SOURCES["muscle"] = dict(
    reference="Kedlian 2024 Nature Aging, CELLxGENE collection 2d40e6a7; cell suspensions, human intercostal 10x 3' v3, mouse hindlimb 10x 3' v2",
    map=MAPS / "muscle_ageing.tsv", unit_col="donor_id",
    species={"hsapiens": dict(name="Human", files=[MU / "5fd93c55-8047-48fb-905c-6ef2f7767483.h5ad"], source="Kedlian2024", label_col="annotation_level0",
                              filters={"donor_id": ["583B", "591C"], "batch": ["cells"], "assay": ["10x 3' v3"]}, filter_reason="young donors kept out of the DE; one chemistry; cell fraction in both species, since fraction follows class in the nucleus libraries"),
             "mmusculus": dict(name="Mouse", files=[MU / "7085b988-463e-4183-b33d-838e1f6ad739.h5ad"], source="Kedlian2024", label_col="annotation",
                               filters={"donor_id": ["Donor1", "Donor2"], "suspension_type": ["cell"]}, filter_reason="3-month donors kept out of the DE; cell fraction in both species")})
COHORTS["muscle_ageing"] = dict(source="muscle", species=["hsapiens", "mmusculus"], tissue="skeletal muscle", label="Muscle human, mouse")


# Check of the feature choice (coordinator): human + macaque with its own table, prior on human-macaque identity and ESM of both species, as the canonical pairs;
# compared with brain_dlpfc_hmac (subset of the 4-species object) on shared genes.
COHORTS["brain_dlpfc_hmac_own"] = dict(source="dlpfc", species=["hsapiens", "mmulatta"], tissue="cortex", label="DLPFC human, macaque, own table")
