"""T7.9: islet T2D objects (plan section 8b), the T7.2 pipeline on islet data.

Human: 48-donor islet atlas (CELLxGENE 464a2c70), integer counts from ``raw.X``.
Mouse: Hrovatin 2023 atlas (CELLxGENE 2cf0b367), integer counts from ``raw.X``, only the
libraries listed in ``donor_roles.tsv`` (db/db and mSTZ studies; 4m and aged healthy adults).
Roles come from ``donor_roles.tsv`` (``split_donors.py``, seed 20261101). No logFC, no
disease concordance and no Chimera readout is computed here.

Steps (section 8b):

1. ``_validate_raw_counts``, ``_standardize_obs_columns`` and ``_qc_filter``
   (>= 200 genes, mitochondrial fraction < 20%) from ``chimera._data_utils``. ``obs['batch']`` of
   the healthy object is the technical source dataset (``technical_batch``); donors and libraries
   are in ``obs['sample_batch']``.
2. ``cell_ontology_class`` from the author labels through ``celltype_map.tsv`` (human
   ``Annotated_Clusters``, mouse ``cell_type_reannotatedIntegrated``; Hrovatin ``low_q`` cells
   are unresolved). Coarse marker check per author label, reported only.
3. Per-library pseudobulks per class for every unit in ``donor_roles.tsv`` (all QC-passed
   annotated cells, no caps, native gene space; ``layers['n_detected']`` for the 5% rule).
4. Gate G-D1b and the owner filter rule: a unit enters a type's DE only with >= 20 cells; a
   type is DE-able only if >= 6 ND (DE half), >= 12 T2D, >= 2 control and >= 2 diabetic mouse
   units remain. Every dropped unit-type pair and type is written to ``filters.csv``.
5. Healthy islet object (training-half donors and training mice only): classes with >= 50
   training cells in each species (others filtered and logged), caps of 2,000 cells per
   species x class shared by donors (T7.2 water-filling, seed 20261101), testis table
   vocabulary, ``compute_conservation_prior``, ESM features as T7.2,
   ``construct_unified_adata``, ``compute_expression_stats`` and ``assemble_gin_features`` on
   these cells, ``precompute_p_null``, ``assert_table_matches``. Genes are dropped if absent from
   either species' matrix, not measured in every mouse training study, or in < 3 cells (list and
   reasons in ``filters_genes.csv``). The Hrovatin matrix is the union of its studies' references:
   a gene missing from a study's reference is padded with zeros there (``var['present_*']``), so a
   false zero that a check by gene ID cannot see. Mouse pseudobulks carry the same flags for the
   DE studies (``var['measured_dbdb']``, ``var['measured_mstz']``).

    PYTHONPATH=<chimera repo> python revision/tasks/T7_9_islet/build.py [--force] [--reannotate]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem

from revision.common import gene_ids
from revision.common.outputs import TaskOutput, add_common_args, git_sha
from revision.tasks.T7_2_disease_objects import build as t72

from chimera._data_utils import (
    assemble_gin_features,
    compute_conservation_prior,
    compute_esm2_interspecies_variance,
    compute_expression_stats,
    construct_unified_adata,
    download_and_process_esm2,
    pca_esm2,
    precompute_p_null,
)

log = logging.getLogger("T7_9")

ROOT = gene_ids.REPO_ROOT
TASK = ROOT / "revision" / "tasks" / "T7_9_islet"
RAW = ROOT / "data" / "revision" / "disease_islet" / "raw"
HUMAN_H5AD = RAW / "human48" / "464a2c70-ebab-48c2-818d-be44c0a9c29e.h5ad"
MOUSE_H5AD = RAW / "hrovatin" / "2cf0b367-e31b-4ab3-8557-8f2c4ac2a31a.h5ad"
ROLES = TASK / "donor_roles.tsv"
CTMAP = TASK / "celltype_map.tsv"
TABLE = t72.TABLE
SATURN = t72.SATURN
OUT = ROOT / "data" / "revision" / "disease_islet"

SEED = 20261101
MIN_CELLS_UNIT = 20
MIN_CELLS_MODEL = 50
DE_TYPES = ["beta", "alpha", "delta", "ductal", "endothelial", "stellate"]
MIN_ND, MIN_T2D, MIN_MOUSE_CTRL, MIN_MOUSE_DIAB, MIN_DEABLE = 6, 12, 2, 2, 3
# Study -> Hrovatin var flag: was the gene in that study's reference?
MEASURED = {"4m": "present_Fltp_adult", "aged": "present_Fltp_2y", "db/db": "present_VSG", "mSTZ": "present_STZ"}

HUMAN_OBS = ["donor_id", "disease", "sex", "assay", "Annotated_Clusters", "LibraryID", "BMI", "HbA1c",
             "development_stage", "Center"]
MOUSE_OBS = ["dataset__design__sample", "dataset", "design", "sex", "assay", "age", "strain",
             "cell_type_reannotatedIntegrated", "low_q", "GEO_accession"]

# Coarse marker sets (human symbols); an annotation check only.
MARKERS = {
    "beta": ["INS", "IAPP", "MAFA", "NKX6-1", "PDX1", "G6PC2"],
    "alpha": ["GCG", "ARX", "IRX2", "LOXL4"],
    "delta": ["SST", "HHEX", "RBP4"],
    "pp": ["PPY"],
    "ductal": ["KRT19", "SOX9", "CFTR", "SPP1", "KRT7"],
    "acinar": ["PRSS1", "CPA1", "CELA3A", "CTRB2", "PNLIP"],
    "endothelial": ["PECAM1", "PLVAP", "FLT1", "KDR", "ESAM"],
    "stellate": ["PDGFRB", "COL1A1", "COL1A2", "RGS5", "SPARC"],
    "immune": ["PTPRC", "CD74", "LAPTM5", "TYROBP"],
    "schwann": ["SOX10", "PLP1", "CDH19"],
}


# --------------------------------------------------------------------------- loading


def _read(path, obs_cols, rows=None) -> ad.AnnData:
    with h5py.File(path, "r") as f:
        obs = pd.DataFrame({c: read_elem(f["obs"][c]) for c in obs_cols})
        obs.index = pd.Index(read_elem(f["obs"][f["obs"].attrs["_index"]])).astype(str)
        var = read_elem(f["raw"]["var"])
        X = read_elem(f["raw"]["X"])
    X = sp.csr_matrix(X)
    if rows is not None:
        X, obs = X[rows], obs.iloc[rows]
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame({"gene_ids": var.index.astype(str)},
                                                   index=var["feature_name"].astype(str).to_numpy()))
    a.var_names_make_unique()
    return a


def mouse_measured() -> pd.DataFrame:
    """Per mouse study: was the gene in that study's reference (ENSMUSG index, one bool per study)."""
    with h5py.File(MOUSE_H5AD, "r") as f:
        var = read_elem(f["raw"]["var"])
    return pd.DataFrame({k: var[c].astype(bool).to_numpy() for k, c in MEASURED.items()},
                        index=var.index.astype(str))


def load_human(roles: pd.DataFrame) -> ad.AnnData:
    h = _read(HUMAN_H5AD, HUMAN_OBS)
    r = roles[roles.species == "Human"].set_index("unit")
    o = h.obs
    h.obs = pd.DataFrame({
        "sample": o["donor_id"].astype(str), "donor": o["donor_id"].astype(str), "species": "Human",
        "study": "human48", "disease": o["disease"].astype(str), "sex": o["sex"].astype(str),
        "assay": o["assay"].astype(str), "library": o["LibraryID"].astype(str),
        "author_label": o["Annotated_Clusters"].astype(str), "bmi": pd.to_numeric(o["BMI"], errors="coerce"),
        "hba1c": pd.to_numeric(o["HbA1c"], errors="coerce"), "age": o["development_stage"].astype(str),
        "center": o["Center"].astype(str),
    }, index=h.obs_names)
    h.obs["role"] = h.obs["donor"].map(r["role"]).to_numpy()
    h.obs["condition"] = h.obs["donor"].map(r["condition"]).to_numpy()
    if h.obs["role"].isna().any():
        raise ValueError("human donor without role")
    log.info("human: %d cells x %d genes", h.n_obs, h.n_vars)
    return h


def load_mouse(roles: pd.DataFrame) -> ad.AnnData:
    r = roles[roles.species == "Mouse"].set_index("unit")
    with h5py.File(MOUSE_H5AD, "r") as f:
        units = read_elem(f["obs"]["dataset__design__sample"]).astype(str)
    rows = np.flatnonzero(np.isin(np.asarray(units), r.index.to_numpy()))
    m = _read(MOUSE_H5AD, MOUSE_OBS, rows=rows)
    o = m.obs
    m.obs = pd.DataFrame({
        "sample": o["dataset__design__sample"].astype(str), "donor": o["dataset__design__sample"].astype(str),
        "species": "Mouse", "study": o["dataset"].astype(str), "design": o["design"].astype(str),
        "disease": "NA", "sex": o["sex"].astype(str), "assay": o["assay"].astype(str),
        "age": o["age"].astype(str), "strain": o["strain"].astype(str),
        "author_label": o["cell_type_reannotatedIntegrated"].astype(str),
        "low_q": o["low_q"].astype(bool), "geo": o["GEO_accession"].astype(str),
    }, index=m.obs_names)
    m.obs["role"] = m.obs["sample"].map(r["role"]).to_numpy()
    m.obs["condition"] = m.obs["sample"].map(r["condition"]).to_numpy()
    m.obs["disease"] = m.obs["sample"].map(r["disease"]).astype(str).to_numpy()
    log.info("mouse: %d cells x %d genes, %d libraries", m.n_obs, m.n_vars, m.obs["sample"].nunique())
    return m


def annotate(a: ad.AnnData, species: str, ctmap: pd.DataFrame) -> None:
    mp = ctmap[ctmap.species == species].set_index("author_label")["cell_ontology_class"]
    unknown = sorted(set(a.obs["author_label"]) - set(mp.index))
    if unknown:
        raise ValueError(f"{species}: labels not in celltype_map.tsv: {unknown}")
    a.obs["cell_ontology_class"] = a.obs["author_label"].map(mp).to_numpy()
    if "low_q" in a.obs:
        a.obs.loc[a.obs["low_q"].to_numpy(), "cell_ontology_class"] = "unresolved"


def marker_check(a: ad.AnnData, species: str, table: pd.DataFrame, res: TaskOutput) -> None:
    if species == "Human":
        mk = MARKERS
    else:
        hm = t72.human_to_mouse(sorted({g for v in MARKERS.values() for g in v}), a.var, table)
        id2sym = pd.Series(a.var_names, index=a.var["gene_ids"].to_numpy())
        mk = {c: [id2sym[hm[g][0]] for g in v if g in hm and hm[g][0] in id2sym.index] for c, v in MARKERS.items()}
        res.json({c: {"human": MARKERS[c], "mouse": mk[c]} for c in mk}, "mouse_coarse_markers.json")
    sc_ = t72.marker_scores(t72.lognorm(a), mk)
    chk = t72.marker_class(sc_, a.obs["author_label"].astype(str))
    chk["n"] = a.obs["author_label"].value_counts()
    res.csv(chk.reset_index().rename(columns={"index": "author_label"}), f"marker_check_{species.lower()}.csv")


# --------------------------------------------------------------------------- gate, filters


def gate_gd1b(cells: pd.DataFrame, roles: pd.DataFrame):
    de = roles[roles.role.str.startswith("de_")].copy()
    tab = cells.groupby(["sample", "cell_ontology_class"]).size().unstack(fill_value=0)
    tab = tab.reindex(index=de["unit"], columns=DE_TYPES, fill_value=0)
    de = de.set_index("unit")
    group = (de["species"] + "_" + de["role"]).rename("group")
    filt, rows = [], []
    for t in DE_TYPES:
        ok = tab[t] >= MIN_CELLS_UNIT
        for u in tab.index[~ok]:
            filt.append(dict(stage="DE", species=de.loc[u, "species"], cell_type=t, unit=u,
                             n_cells=int(tab.loc[u, t]), rule=f"unit has < {MIN_CELLS_UNIT} cells of type",
                             action="unit dropped from this type's DE"))
        n = ok.groupby(group).sum()
        r = dict(cell_type=t, nd_ok=int(n.get("Human_de_control", 0)), nd_n=int((group == "Human_de_control").sum()),
                 t2d_ok=int(n.get("Human_de_disease", 0)), t2d_n=int((group == "Human_de_disease").sum()),
                 mouse_ctrl_ok=int(n.get("Mouse_de_control", 0)), mouse_ctrl_n=int((group == "Mouse_de_control").sum()),
                 mouse_diab_ok=int(n.get("Mouse_de_disease", 0)), mouse_diab_n=int((group == "Mouse_de_disease").sum()),
                 min_cells_human=int(tab.loc[de.index[de.species == "Human"], t].min()),
                 min_cells_mouse=int(tab.loc[de.index[de.species == "Mouse"], t].min()),
                 mouse_units_ok=",".join(u for u in tab.index[ok] if de.loc[u, "species"] == "Mouse"))
        r["de_able"] = bool(r["nd_ok"] >= MIN_ND and r["t2d_ok"] >= MIN_T2D and r["mouse_ctrl_ok"] >= MIN_MOUSE_CTRL
                            and r["mouse_diab_ok"] >= MIN_MOUSE_DIAB)
        r["de_able_strict_all_units"] = bool(ok.all())
        if not r["de_able"]:
            filt.append(dict(stage="DE", species="both", cell_type=t, unit="all", n_cells=np.nan,
                             rule="type not DE-able under G-D1b unit minima", action="type filtered from DE"))
        rows.append(r)
    types = pd.DataFrame(rows)
    n_able = int(types["de_able"].sum())
    nunit = group.value_counts()
    crit = pd.DataFrame([
        dict(criterion="ND DE-half donors >= 6", value=int(nunit.get("Human_de_control", 0)), threshold=MIN_ND),
        dict(criterion="T2D donors >= 12", value=int(nunit.get("Human_de_disease", 0)), threshold=MIN_T2D),
        dict(criterion="mouse control DE libraries >= 2", value=int(nunit.get("Mouse_de_control", 0)),
             threshold=MIN_MOUSE_CTRL),
        dict(criterion="mouse diabetic DE libraries >= 2", value=int(nunit.get("Mouse_de_disease", 0)),
             threshold=MIN_MOUSE_DIAB),
        dict(criterion="beta DE-able", value=int(types.loc[types.cell_type == "beta", "de_able"].iloc[0]), threshold=1),
        dict(criterion=">= 3 of 6 DE types DE-able", value=n_able, threshold=MIN_DEABLE),
    ])
    crit["passed"] = crit["value"] >= crit["threshold"]
    return crit, types, tab, filt


# --------------------------------------------------------------------------- healthy object


def technical_batch(obs: pd.DataFrame) -> pd.Series:
    """``obs['batch']``: the technical source dataset within a species, as in the canonical cohorts.

    Species + source study, plus chemistry where it differs within a study. Donors and libraries
    never enter it (iter94); they stay in ``obs['sample_batch']`` for DE and pseudobulks.
    """
    chem = obs["assay"].astype(str).str.extract(r"(v\d)$")[0]
    study = obs["study"].astype(str)
    multi = chem.groupby(study).transform("nunique") > 1
    out = obs["species"].astype(str) + "_" + study
    return out.where(~multi, out + "_" + chem)


def build_healthy(h: ad.AnnData, m: ad.AnnData, table: pd.DataFrame, filt: list,
                  res: TaskOutput) -> tuple[ad.AnnData, dict]:
    obs_cols = ["sample", "donor", "species", "study", "condition", "cell_ontology_class", "role", "sex", "assay",
                "age", "author_label"]
    hs = h[(h.obs["role"] == "train_healthy").to_numpy()].copy()
    ms = m[(m.obs["role"] == "train_healthy").to_numpy()].copy()
    ch, cm = hs.obs["cell_ontology_class"].value_counts(), ms.obs["cell_ontology_class"].value_counts()
    allc = sorted((set(ch.index) | set(cm.index)) - {"unresolved"})
    classes = [c for c in allc if ch.get(c, 0) >= MIN_CELLS_MODEL and cm.get(c, 0) >= MIN_CELLS_MODEL]
    for c in allc:
        if c not in classes:
            filt.append(dict(stage="healthy_model", species="both", cell_type=c, unit="training cells",
                             n_cells=f"Human {int(ch.get(c, 0))}, Mouse {int(cm.get(c, 0))}",
                             rule=f"< {MIN_CELLS_MODEL} training cells in a species", action="class filtered from model"))
    hs = hs[hs.obs["cell_ontology_class"].isin(classes).to_numpy()].copy()
    ms = ms[ms.obs["cell_ontology_class"].isin(classes).to_numpy()].copy()
    rng = np.random.default_rng(SEED)
    hs = hs[t72.cap_nuclei(hs.obs, rng)].copy()
    ms = ms[t72.cap_nuclei(ms.obs, rng)].copy()
    human_present, mouse_present = set(h.var["gene_ids"]), set(m.var["gene_ids"])
    def _by_id(a: ad.AnnData) -> ad.AnnData:
        a.var_names = a.var["gene_ids"].astype(str).to_numpy()
        a = a[:, ~a.var_names.duplicated()].copy()
        a.obs = a.obs[obs_cols].copy()
        a.var = a.var[[]]
        return a

    hs, ms = _by_id(hs), _by_id(ms)

    prior = compute_conservation_prior(table)
    consensus, per_sp = download_and_process_esm2(table, species_list=["hsapiens", "mmulatta", "mmusculus"],
                                                  saturn_data_dir=SATURN)
    esm_pca = pca_esm2(consensus, n_components=64)
    esm_var = compute_esm2_interspecies_variance(per_sp)
    G = len(table)
    unified = construct_unified_adata({"hsapiens": hs, "mmusculus": ms}, table,
                                      np.zeros((G, 72), dtype=np.float32), prior)
    unified.X = sp.csr_matrix(unified.X, dtype=np.float32)
    stats = compute_expression_stats(unified)
    feats, scaler = assemble_gin_features(esm_pca, esm_var, table, stats)
    unified.varm["gene_features"] = feats
    unified.uns["p_null"] = precompute_p_null(unified)
    unified.obs["sample_batch"] = unified.obs["sample"].astype(str)
    unified.obs["batch"] = technical_batch(unified.obs).to_numpy()
    unified.obs["cell_type"] = unified.obs["cell_ontology_class"].astype(str)  # same labels, standard name
    for c in ("species", "batch", "sample_batch", "cell_type", "cell_ontology_class", "condition", "donor", "role", "sex", "assay", "study",
              "sample", "age", "author_label"):
        unified.obs[c] = unified.obs[c].astype(str).astype("category")

    canon = ad.read_h5ad(t72.CANON_TESTIS, backed="r")
    cf = np.asarray(canon.varm["gene_features"])
    struct = [i for i in range(72) if i not in (66, 67, 68)]
    max_diff = float(np.abs(cf[:, struct] - feats[:, struct]).max())
    canon.file.close()

    present = np.array([(hg in human_present) and (mg in mouse_present)
                        for hg, mg in zip(table["hsapiens_gene_id"], table["mmusculus_gene_id"])])
    nnz = np.asarray((unified.X != 0).sum(axis=0)).ravel()
    flags = mouse_measured()
    trained = set(flags.index[flags["4m"] & flags["aged"]])
    measured = table["mmusculus_gene_id"].isin(trained).to_numpy()
    gene_mask = present & measured & (nnz >= 3)
    reason = np.select([~present, ~measured, nnz < 3],
                       ["absent from a species matrix", "not measured in every mouse training study (padded zeros)",
                        "in < 3 cells"], default="kept")
    res.csv(table[["ortholog_group_id", "hsapiens_gene_id", "hsapiens_gene_name", "mmusculus_gene_id"]]
            .assign(reason=reason).query("reason != 'kept'"), "filters_genes.csv")
    t72._write_h5ad(unified, OUT / "islet_healthy_train_full_vocab.h5ad")
    obj = unified[:, gene_mask].copy()
    obj.uns["p_null"] = precompute_p_null(obj, species_key="species")
    info = dict(classes=classes, n_genes_table=G, n_genes_absent_in_a_matrix=int((~present).sum()),
                n_genes_unmeasured_in_a_mouse_training_study=int((present & ~measured).sum()),
                n_genes_lt3_cells_among_measured=int((present & measured & (nnz < 3)).sum()),
                n_genes=int(gene_mask.sum()),
                max_abs_diff_struct_features_vs_canonical_testis=max_diff,
                frac_prior_ge_095=float((obj.var["conservation_prior"] >= 0.95).mean()),
                frac_table_identity_ge_95=float((table["mean_identity"] >= 95).mean()))
    with open(OUT / "islet_healthy_train_gin_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return obj, info


# --------------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--reannotate", action="store_true")
    a = p.parse_args()
    res = TaskOutput("T7_9", a.outdir, prerun=a.prerun, force=a.force)
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TABLE)
    if len(table) != 15732:
        raise ValueError(f"testis table has {len(table)} rows")
    roles = pd.read_csv(ROLES, sep="\t", dtype=str)
    ctmap = pd.read_csv(CTMAP, sep="\t", dtype=str)

    ann_h, ann_m = OUT / "annotated_human.h5ad", OUT / "annotated_mouse.h5ad"
    if a.reannotate or not (ann_h.exists() and ann_m.exists()):
        h, qh = t72.qc(load_human(roles), "human48")
        m, qm = t72.qc(load_mouse(roles), "hrovatin")
        res.csv(qh.pop("per_sample"), "qc_human_per_sample.csv", index=True)
        res.csv(qm.pop("per_sample"), "qc_mouse_per_sample.csv", index=True)
        res.json(dict(human=qh, mouse=qm, min_genes=t72.MIN_GENES, max_mt_pct=t72.MAX_MT), "qc_summary.json")
        annotate(h, "Human", ctmap)
        annotate(m, "Mouse", ctmap)
        marker_check(h, "Human", table, res)
        marker_check(m, "Mouse", table, res)
        for x, path in ((h, ann_h), (m, ann_m)):
            for c in x.obs.columns:
                if x.obs[c].dtype == object:
                    x.obs[c] = x.obs[c].astype(str)
            t72._write_h5ad(x, path)
    else:
        h, m = ad.read_h5ad(ann_h), ad.read_h5ad(ann_m)

    cells = pd.concat([x.obs[["species", "sample", "role", "condition", "cell_ontology_class"]] for x in (h, m)])
    res.csv(cells.astype(str).value_counts().rename("n_cells").reset_index()
            .sort_values(["species", "role", "sample", "cell_ontology_class"]), "counts_per_unit_type.csv")
    res.csv(cells.astype(str).groupby(["species", "role", "condition", "cell_ontology_class"]).size()
            .unstack(fill_value=0).reset_index(), "counts_role_type.csv")

    keys_h = ["sample", "donor", "species", "study", "condition", "role", "disease", "sex", "assay", "bmi", "hba1c",
              "age", "center"]
    keys_m = ["sample", "donor", "species", "study", "condition", "role", "disease", "design", "sex", "assay", "age",
              "strain", "geo"]
    paths = {}
    for x, keys, name in ((h, keys_h, "human"), (m, keys_m, "mouse")):
        pb = t72.pseudobulk(x, keys)
        pb.var["gene_symbol"] = pb.var_names
        pb.var["gene_id"] = x.var["gene_ids"].reindex(pb.var_names).to_numpy()
        if name == "mouse":
            flags = mouse_measured().reindex(pb.var["gene_id"].astype(str).to_numpy())
            pb.var["measured_dbdb"] = flags["db/db"].to_numpy(dtype=bool)
            pb.var["measured_mstz"] = flags["mSTZ"].to_numpy(dtype=bool)
        for c in pb.obs.columns:
            if pb.obs[c].dtype == object:
                pb.obs[c] = pb.obs[c].astype(str)
        paths[f"pseudobulk_{name}"] = OUT / f"islet_pseudobulk_{name}.h5ad"
        t72._write_h5ad(pb, paths[f"pseudobulk_{name}"])

    crit, types, unit_tab, filt = gate_gd1b(cells.astype(str), roles)
    res.csv(crit, "gd1b_criteria.csv")
    res.csv(types, "gd1b_types.csv")
    res.csv(unit_tab, "gd1b_units.csv", index=True)
    gd1b = bool(crit["passed"].all())
    log.info("G-D1b %s\n%s\n%s", "PASS" if gd1b else "FAIL", crit.to_string(), types.to_string())

    obj, info = build_healthy(h, m, table, filt, res)
    res.csv(pd.DataFrame(filt), "filters.csv")
    rho = gene_ids.assert_table_matches(obj, gene_ids.load_table("testis"))
    paths["healthy"] = OUT / "islet_healthy_train.h5ad"
    paths["healthy_full_vocab"] = OUT / "islet_healthy_train_full_vocab.h5ad"
    paths["gin_scaler"] = OUT / "islet_healthy_train_gin_scaler.pkl"
    t72._write_h5ad(obj, paths["healthy"])
    res.csv(obj.obs.groupby(["species", "cell_ontology_class"], observed=True).size().rename("n_cells").reset_index(),
            "counts_healthy_object.csv")
    res.csv(obj.obs.groupby(["species", "sample"], observed=True).size().rename("n_cells").reset_index(),
            "counts_healthy_object_per_unit.csv")
    info.update(assert_table_matches_spearman=rho, n_obs=int(obj.n_obs), n_vars=int(obj.n_vars),
                p_null_shape=list(obj.uns["p_null"].shape), gd1b_pass=gd1b, cap=t72.CAP, seed=SEED)
    res.json(info, "healthy_summary.json")
    paths["annotated_human"], paths["annotated_mouse"] = ann_h, ann_m
    out_hash = {k: dict(path=str(v.relative_to(ROOT)), bytes=v.stat().st_size, sha256=t72.sha256(v))
                for k, v in paths.items()}
    in_hash = {str(x.relative_to(ROOT)): t72.sha256(x) for x in (HUMAN_H5AD, MOUSE_H5AD, ROLES, CTMAP, TABLE)}
    res.json(dict(outputs=out_hash, inputs=in_hash, code_sha=git_sha()), "hashes.json")
    res.finish(dict(gd1b_pass=gd1b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
