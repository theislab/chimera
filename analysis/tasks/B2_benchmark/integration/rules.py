"""Integration benchmark on the new cohorts, a track of its own (Track I). Integration is the substrate of Chimera's readouts, not its aim; this track says whether the latent is usable.

Same benchmark as Fig 3: the 10-model panel, compute_integration_metrics unmodified (k 15, seed 42), direction correction, min-max across the models of each cohort, mixing and biology means, composite 0.4 mixing + 0.6 biology
(T5.5 score_external.score_table). Scaling is PER COHORT, so adding a cohort cannot change any value of the original seven; `score.py` checks this, and the seven-cohort aggregate is kept as reported, with the extended one beside it.

Objects and Chimera fit: H1 table, kept s42 fit. Batch keys: `config_key` is the run's config.json batch key (coordinator's instruction), and fills the "scVI-donor", Harmony, scANVI, sysVI columns. On kidney and bone marrow that key is one-to-one with species, as `batch` is on
liver H+Mouse, so scVI-donor equals scVI-sp there in all but name; a sensitivity table therefore replaces those four columns by the sample-level key `sample_key` (what the testis rows use), reported beside and not in the main table.
Van Zyl H+Mouse: the seven-method row of T11.4 is reused as it stands (labels cell_type_harmonised, Harmony/scVI-donor/scANVI/sysVI on `batch`); only SATURN, SATURN-Leiden and CAMEX are new.
SATURN, SATURN-Leiden, CAMEX on new cohorts go through the 1:1 orthologs (T5.6 Adapter B; no raw per-species gene sets or homology tables exist for them); the seven original rows use the native adapters. Stated in the table's `adapter` column.
"""

from __future__ import annotations

from analysis.common import gene_ids

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "analysis" / "B2" / "integration"
INPUTS = REPO / "data" / "analysis" / "B2" / "integration_inputs"
COHORTS = {
    "bone_marrow": dict(h5ad="data/analysis/bone_marrow/v2/bone_marrow_unified.h5ad", stem="bone_marrow_unified", label="cell_ontology_class", table="bone_marrow", run="rev-bm2-grid5-s42",
                        config_key="species", sample_key="batch", tissue="bone marrow", dataset_label="bone marrow 3-sp",
                        species={"human": ("human", "hsapiens_gene_name"), "mouse": ("mouse", "mmusculus_gene_name"), "mouse_lemur": ("mouse_lemur", "mmurinus_gene_name")}),
    "kidney": dict(h5ad="data/analysis/disease_kidney/kidney_healthy_train.h5ad", stem="kidney_healthy_train", label="cell_type", table="kidney", run="rev-kid-grid3-s42",
                   config_key="batch", sample_key="sample_batch", tissue="kidney", dataset_label="kidney H+Mouse",
                   species={"Human": ("human", "hsapiens_gene_name"), "Mouse": ("mouse", "mmusculus_gene_name")}),
    "islet": dict(h5ad="data/analysis/disease_islet/islet_healthy_train.h5ad", stem="islet_healthy_train", label="cell_type", table="testis", run="rev-isl-grid14-s42",
                  config_key="batch", sample_key="sample_batch", tissue="islet", dataset_label="islet H+Mouse",
                  species={"Human": ("human", "hsapiens_gene_name"), "Mouse": ("mouse", "mmusculus_gene_name")}),
    "vz_hmouse": dict(h5ad="data/analysis/van_zyl/van_zyl_hmouse.h5ad", stem="van_zyl_hmouse", label="cell_type_harmonised", table="van_zyl", run="rev-vz-hmouse-grid1-s42",
                      config_key="batch", sample_key=None, tissue="eye", dataset_label="eye H+Mouse", reuse_t11=True,
                      species={"human": ("human", "hsapiens_gene_name"), "mouse": ("mouse", "mmusculus_gene_name")}),
}
MODELS_ORDER = ["PCA", "scVI-sp", "scVI-donor", "scANVI", "Harmony", "sysVI", "Chimera", "SATURN", "SATURN-Leiden", "CAMEX"]


def baselines_tsv() -> list[tuple[str, str]]:
    rows = []
    for c, d in COHORTS.items():
        if d.get("reuse_t11"):
            continue
        ad = f"--adata {d['h5ad']}"
        lab = d["label"]
        rows.append((f"{c}:PCA", f"{ad} --method pca --seed 42"))
        rows.append((f"{c}:scVI-sp", f"{ad} --method scvi --batch-key species --seed 42"))
        for key, tag in ((d["config_key"], "cfg"), (d["sample_key"], "sample")):
            if tag == "cfg" and key == "species":  # scVI on species is scVI-sp already
                pass
            else:
                rows.append((f"{c}:{tag}:scVI", f"{ad} --method scvi --batch-key {key} --seed 42"))
            rows.append((f"{c}:{tag}:Harmony", f"{ad} --method harmony --batch-key {key} --seed 42"))
            rows.append((f"{c}:{tag}:scANVI", f"{ad} --method scanvi --batch-key {key} --labels-key {lab} --seed 42"))
            rows.append((f"{c}:{tag}:sysVI", f"{ad} --method sysvi --batch-key {key} --species-key species --seed 42"))
    return rows


def latent_paths(c: str, tag: str) -> dict:
    """model -> latent folder, per the runner's cache names. tag: cfg or sample."""
    d = COHORTS[c]
    stem = d["stem"]
    R = REPO / "training" / "runs"
    key = d["config_key"] if tag == "cfg" else d["sample_key"]
    bk = "" if key == "species" else f"-bk_{key}"
    return {"PCA": R / "baselines_pca" / stem, "scVI-sp": R / "baselines_scvi" / f"{stem}-s42-e100-nl10",
            "scVI-donor": R / "baselines_scvi" / f"{stem}-s42-e100-nl10{bk}", "scANVI": R / "baselines_scanvi" / f"{stem}-s42{bk}",
            "Harmony": R / "baselines_harmony" / f"{stem}-s42{bk}", "sysVI": R / "baselines_sysvi" / f"{stem}-s42{bk}"}
