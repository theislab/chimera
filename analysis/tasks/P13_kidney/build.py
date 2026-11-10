"""P13: kidney AKI objects (plan section 8c), the T7.2 / T7.9 pipeline on the KPMP collection.

Human: CELLxGENE snRNA v2.0 (``human_kidney_v2.h5ad``); mouse: snRNA v2.0
(``mouse_kidney_v2.h5ad``); integer counts from ``raw/X``. Only nuclei of the units listed in
``unit_roles.tsv`` (``split_units.py``, seed 20261101) are read. No logFC, no disease
concordance and no Chimera readout is computed here.

The human file (1.39 M nuclei) is streamed in row blocks of ``raw/X``; per-cell steps run per
block, so the result equals a whole-file run:

1. ``_validate_raw_counts``, ``_standardize_obs_columns`` (batch = unit) and ``_qc_filter``
   (>= 200 genes) from ``chimera._data_utils``. No mitochondrial filter in any species (plan
   amendment after section 8c): it cannot act evenly. The human matrix has no MT- genes, mouse
   has 11, the authors' ``percent.mt`` is missing for all Kirita nuclei and disagrees with the
   matrix-based fraction in mouse, and in human it would remove AKI nuclei preferentially. What
   the 20% rule would have removed is written to ``mt_filter_not_applied.csv``.
2. ``cell_ontology_class`` from ``SubclassLevel1`` through ``celltype_map.tsv``.
3. Per-unit pseudobulks per class for every unit (summed counts, ``layers['n_detected']``),
   all QC-passed nuclei, no caps, native gene space; per unit x class x ``CellStateLevel2``
   nucleus counts (descriptive state fractions).
4. Gate G-D1k and the owner filter rule (``filters.csv``).
5. Healthy kidney object from ``train_healthy`` units (human multiome nuclei only; mouse Ref):
   classes with >= 50 training nuclei in each species, caps of 2,000 nuclei per species x class
   shared by units (T7.2 water-filling, seed 20261101), kidney ortholog table,
   ``compute_conservation_prior``, ESM-2 (human, mouse), ``pca_esm2``,
   ``compute_esm2_interspecies_variance``, ``construct_unified_adata``,
   ``compute_expression_stats`` and ``assemble_gin_features`` on these nuclei, genes in < 3
   training nuclei dropped, ``precompute_p_null``, ``assert_table_matches``.

    PYTHONPATH=<chimera repo> python analysis/tasks/P13_kidney/build.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem

from chimera._data_utils import (
    _qc_filter,
    _standardize_obs_columns,
    _validate_raw_counts,
    assemble_gin_features,
    compute_conservation_prior,
    compute_esm2_interspecies_variance,
    compute_expression_stats,
    construct_unified_adata,
    download_and_process_esm2,
    pca_esm2,
    precompute_p_null,
)
from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args, git_sha
from analysis.tasks.P13_kidney.split_units import _animal
from analysis.tasks.T7_2_disease_objects import build as t72

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("P13_build")

ROOT = gene_ids.REPO_ROOT
TASK = ROOT / "analysis" / "tasks" / "P13_kidney"
OUT = ROOT / "data" / "analysis" / "disease_kidney"
RAW = OUT / "raw"
FILES = {"Human": RAW / "human_kidney_v2.h5ad", "Mouse": RAW / "mouse_kidney_v2.h5ad"}
ROLES = TASK / "unit_roles.tsv"
CTMAP = TASK / "celltype_map.tsv"
TABLE = OUT / "ortholog_table_kidney.csv"
ESM2 = ROOT / "data" / "raw" / "saturn_esm2" / "protein_embeddings_export" / "ESM2"
ESM2_LINKS = {"human_embedding.torch": "human_embedding.torch", "mouse_embedding.torch": "mouse_embedding.torch"}

SEED = 20261101
MIN_GENES, MT_RULE = 200, 20.0  # MT_RULE: reported only, not applied
MIN_CELLS_UNIT, MIN_CELLS_MODEL = 20, 50
DE_TYPES = ["pt", "tal", "dct", "pc", "ic", "ec", "fib", "myeloid", "lymphoid"]
MIN_H_CTRL, MIN_H_AKI, MIN_M_SHAM, MIN_M_IRI, MIN_DEABLE = 6, 10, 3, 4, 4
BLOCK = 100_000  # physical rows of raw/X per read
MULTIOME = "10x multiome"
OBS_COLS = {"Human": ["donor_id", "assay", "SubclassLevel1", "CellStateLevel2", "region", "percent.mt"],
            "Mouse": ["donor_id", "experiment", "source", "assay", "SubclassLevel1", "CellStateLevel2", "percent.mt"]}
UNIT_KEYS = ["unit", "donor", "species", "role", "condition", "assay", "source", "collection", "sex", "age",
             "stage", "time", "subtype"]


# --------------------------------------------------------------------------- streaming


def read_obs_units(path: Path, species: str, roles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    with h5py.File(path, "r") as f:
        obs = pd.DataFrame({c: read_elem(f["obs"][c]) for c in OBS_COLS[species]}).astype(str)
        obs.index = pd.Index(read_elem(f["obs"][f["obs"].attrs["_index"]])).astype(str)
        var = read_elem(f["raw"]["var"])
    if species == "Human":
        obs["unit"] = obs["donor_id"] + "|" + obs["assay"]
    else:
        obs["unit"] = [_animal(s, e, d) for s, e, d in zip(obs["source"], obs["experiment"], obs["donor_id"])]
    r = roles[roles.species == species].set_index("unit")
    obs = obs[obs["unit"].isin(r.index)].copy()
    var = pd.DataFrame({"gene_ids": var.index.astype(str)}, index=var["feature_name"].astype(str).to_numpy())
    return obs, var


def stream(path: Path, species: str, obs: pd.DataFrame, var: pd.DataFrame, roles: pd.DataFrame,
           ctmap: pd.Series, keep_train) -> dict:
    """One pass over raw/X in physical row blocks: QC, labels, pseudobulk sums, training nuclei kept."""
    r = roles[roles.species == species].set_index("unit")
    with h5py.File(path, "r") as f:
        all_index = pd.Index(read_elem(f["obs"][f["obs"].attrs["_index"]])).astype(str)
    if not all_index.is_unique:
        raise ValueError(f"{species}: obs names not unique")
    rows = np.flatnonzero(all_index.isin(obs.index))
    obs = obs.loc[all_index[rows]]
    var_u = var.copy()
    var_u.index = ad.utils.make_index_unique(pd.Index(var_u.index).astype(str))
    n_genes = len(var_u)
    groups: dict[str, int] = {}
    sums: list[np.ndarray] = []
    dets: list[np.ndarray] = []
    qc_parts, state_parts, train_parts, mt_rec, max_val = [], [], [], [], 0.0
    with h5py.File(path, "r") as f:
        g = f["raw"]["X"]
        indptr = g["indptr"][()]
        n_all = len(indptr) - 1
        for p0 in range(0, n_all, BLOCK):
            pos = np.flatnonzero((rows >= p0) & (rows < p0 + BLOCK))
            if len(pos) == 0:
                continue
            rb = rows[pos]
            lo, hi = int(indptr[rb[0]]), int(indptr[rb[-1] + 1])
            dat, ind = g["data"][lo:hi], g["indices"][lo:hi]
            starts, lens = indptr[rb] - lo, indptr[rb + 1] - indptr[rb]
            offs = np.concatenate([[0], np.cumsum(lens)])
            sel = np.repeat(starts - offs[:-1], lens) + np.arange(offs[-1])
            X = sp.csr_matrix((dat[sel].astype(np.float32), ind[sel], offs), shape=(len(rb), n_genes))
            a = ad.AnnData(X=X, obs=obs.iloc[pos].copy(), var=var_u.copy())
            _validate_raw_counts(a, f"{species} rows {p0}+")
            max_val = max(max_val, float(X.data.max()) if X.nnz else 0.0)
            a.obs["batch"] = a.obs["unit"]
            a = _standardize_obs_columns(a)
            n_before = a.obs["unit"].value_counts()
            mt_rec.append(mt_would_remove(a, species))
            a = _qc_filter(a, min_genes=MIN_GENES, max_mt_pct=np.inf)  # no mitochondrial filter, see docstring
            qc_parts.append(pd.DataFrame({"before": n_before, "after": a.obs["unit"].value_counts()}).fillna(0))
            cls = a.obs["SubclassLevel1"].map(ctmap)
            if cls.isna().any():
                raise ValueError(f"unmapped labels: {sorted(a.obs.loc[cls.isna(), 'SubclassLevel1'].unique())}")
            a.obs["cell_ontology_class"] = cls.to_numpy()
            key = (a.obs["unit"] + "||" + a.obs["cell_ontology_class"]).to_numpy()
            uniq, codes = np.unique(key, return_inverse=True)
            for k in uniq:
                if k not in groups:
                    groups[k] = len(groups)
                    sums.append(np.zeros(n_genes, dtype=np.int64))
                    dets.append(np.zeros(n_genes, dtype=np.int64))
            Gm = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
            Xc = a.X.tocsr().astype(np.float64)
            s_ = (Gm @ Xc).toarray()
            d_ = (Gm @ (Xc > 0).astype(np.float64)).toarray()
            for j, k in enumerate(uniq):
                sums[groups[k]] += np.rint(s_[j]).astype(np.int64)
                dets[groups[k]] += np.rint(d_[j]).astype(np.int64)
            state_parts.append(a.obs.groupby(["unit", "cell_ontology_class", "CellStateLevel2"]).size())
            tr = keep_train(a.obs, r)
            if tr.any():
                t = a[tr].copy()
                t.obs = t.obs[["unit", "cell_ontology_class", "CellStateLevel2"]].copy()
                train_parts.append(t)
            log.info("%s physical rows %d-%d: %d listed, %d after QC", species, p0, p0 + BLOCK, len(rb), a.n_obs)
    qc = pd.concat(qc_parts).groupby(level=0).sum().astype(int)
    states = pd.concat(state_parts).groupby(level=[0, 1, 2]).sum()
    keys = sorted(groups, key=groups.get)
    pb_obs = pd.DataFrame([dict(unit=k.split("||")[0], cell_ontology_class=k.split("||")[1]) for k in keys])
    meta = r.reindex(pb_obs["unit"])
    for c in UNIT_KEYS[1:]:
        pb_obs[c] = meta[c].astype(str).to_numpy()
    pb_obs.index = [k.replace("||", "|") for k in keys]
    nn = states.groupby(level=[0, 1]).sum()
    pb_obs["n_nuclei"] = [int(nn.get((u, c), 0)) for u, c in zip(pb_obs["unit"], pb_obs["cell_ontology_class"])]
    pb = ad.AnnData(X=sp.csr_matrix(np.vstack(sums)), obs=pb_obs, var=var_u.copy())
    pb.layers["n_detected"] = sp.csr_matrix(np.vstack(dets).astype(np.int32))
    order = np.lexsort((pb.obs["cell_ontology_class"].to_numpy(), pb.obs["unit"].to_numpy()))
    pb = pb[order].copy()
    train = ad.concat(train_parts, merge="same") if train_parts else None
    mt = pd.concat(mt_rec).groupby(level=0).sum().astype(int)
    return dict(pb=pb, qc=qc, states=states, train=train, max_val=max_val, n_rows=int(len(rows)), mt=mt)


def mt_would_remove(a: ad.AnnData, species: str) -> pd.DataFrame:
    """Per unit: nuclei the 20% rule would drop, by the authors' percent.mt and by the matrix."""
    x = pd.to_numeric(a.obs["percent.mt"], errors="coerce")
    mt = a.var_names.str.startswith(("MT-", "mt-"))
    tot = np.asarray(a.X.sum(axis=1)).ravel()
    pm = np.asarray(a.X[:, mt].sum(axis=1)).ravel() / np.maximum(tot, 1) * 100 if mt.any() else np.zeros(a.n_obs)
    return pd.DataFrame({"n": 1, "author_nan": x.isna().astype(int).to_numpy(),
                         "author_ge20": (x >= MT_RULE).astype(int).to_numpy(),
                         "matrix_ge20": (pm >= MT_RULE).astype(int)}, index=a.obs["unit"].to_numpy()).groupby(level=0).sum()


def keep_train_human(o: pd.DataFrame, r: pd.DataFrame) -> np.ndarray:
    role = o["unit"].map(r["role"])
    return ((role == "train_healthy") & (o["assay"] == MULTIOME)).to_numpy()


def keep_train_mouse(o: pd.DataFrame, r: pd.DataFrame) -> np.ndarray:
    return (o["unit"].map(r["role"]) == "train_healthy").to_numpy()


# --------------------------------------------------------------------------- gate


def gate_gd1k(pb_h: ad.AnnData, pb_m: ad.AnnData, roles: pd.DataFrame):
    rows, filt = [], []
    groups = {
        "human_control": (pb_h, (roles.species == "Human") & (roles.role == "de_control") & (roles.assay == MULTIOME)),
        "human_aki": (pb_h, (roles.species == "Human") & (roles.role == "de_disease") & (roles.assay == MULTIOME)),
        "mouse_sham": (pb_m, (roles.species == "Mouse") & (roles.role == "de_control")),
        "mouse_iri": (pb_m, (roles.species == "Mouse") & (roles.role == "de_disease")),
    }
    mins = dict(human_control=MIN_H_CTRL, human_aki=MIN_H_AKI, mouse_sham=MIN_M_SHAM, mouse_iri=MIN_M_IRI)
    sens = {
        "human_control_v3": (pb_h, (roles.species == "Human") & (roles.role == "de_control") & (roles.assay != MULTIOME)),
        "human_aki_v3": (pb_h, (roles.species == "Human") & (roles.role == "de_disease") & (roles.assay != MULTIOME)),
        "kirita_sham": (pb_m, roles.role == "sens_control"),
        "kirita_iri": (pb_m, roles.role == "sens_disease"),
    }
    tabs = {}
    for name, (pb, mask) in {**groups, **sens}.items():
        units = roles.loc[mask, "unit"].tolist()
        o = pb.obs[pb.obs["unit"].isin(units)]
        t = o.pivot_table(index="unit", columns="cell_ontology_class", values="n_nuclei", aggfunc="sum",
                          fill_value=0).reindex(index=units, columns=DE_TYPES, fill_value=0)
        tabs[name] = t
    for t in DE_TYPES:
        rec = dict(cell_type=t)
        for name in groups:
            ok = tabs[name][t] >= MIN_CELLS_UNIT
            rec[f"{name}_ok"], rec[f"{name}_n"] = int(ok.sum()), int(len(ok))
            for u in tabs[name].index[~ok]:
                filt.append(dict(stage="DE", group=name, cell_type=t, unit=u, n_nuclei=int(tabs[name].loc[u, t]),
                                 rule=f"unit has < {MIN_CELLS_UNIT} nuclei of type",
                                 action="unit dropped from this type's DE"))
        for name in sens:
            rec[f"{name}_ok"] = int((tabs[name][t] >= MIN_CELLS_UNIT).sum())
            rec[f"{name}_n"] = int(len(tabs[name]))
        rec["de_able"] = bool(all(rec[f"{n}_ok"] >= mins[n] for n in groups))
        rec["de_able_strict_all_units"] = bool(all(rec[f"{n}_ok"] == rec[f"{n}_n"] for n in groups))
        if not rec["de_able"]:
            filt.append(dict(stage="DE", group="all", cell_type=t, unit="all", n_nuclei=np.nan,
                             rule="type not DE-able under G-D1k unit minima", action="type filtered from DE"))
        rows.append(rec)
    types = pd.DataFrame(rows)
    n = {k: int(v[1].sum()) for k, v in groups.items()}
    crit = pd.DataFrame([
        dict(criterion="healthy DE-half human multiome units >= 6", value=n["human_control"], threshold=MIN_H_CTRL),
        dict(criterion="AKI human multiome units >= 10", value=n["human_aki"], threshold=MIN_H_AKI),
        dict(criterion="Sham mouse multiome units >= 3", value=n["mouse_sham"], threshold=MIN_M_SHAM),
        dict(criterion="IRI mouse multiome units >= 4", value=n["mouse_iri"], threshold=MIN_M_IRI),
        dict(criterion="PT DE-able", value=int(types.loc[types.cell_type == "pt", "de_able"].iloc[0]), threshold=1),
        dict(criterion=">= 4 of 9 DE types DE-able", value=int(types["de_able"].sum()), threshold=MIN_DEABLE),
    ])
    crit["passed"] = crit["value"] >= crit["threshold"]
    units = pd.concat({k: v for k, v in tabs.items()}, names=["group", "unit"])
    return crit, types, units, filt


# --------------------------------------------------------------------------- healthy object


def link_esm() -> Path:
    d = OUT / "inputs" / "esm2" / "ESM2"
    d.mkdir(parents=True, exist_ok=True)
    for link, target in ESM2_LINKS.items():
        if not (d / link).exists():
            (d / link).symlink_to(ESM2 / target)
    return d.parent


def build_healthy(th: ad.AnnData, tm: ad.AnnData, roles: pd.DataFrame, table: pd.DataFrame, filt: list,
                  observed: dict[str, set]):
    for a, s in ((th, "Human"), (tm, "Mouse")):
        r = roles[roles.species == s].set_index("unit")
        a.obs["species"] = s
        a.obs["donor"] = a.obs["unit"].astype(str)
        a.obs["sample"] = a.obs["unit"].astype(str)
        a.obs["condition"] = "control"
        for c in ("sex", "age", "assay", "source", "collection"):
            a.obs[c] = a.obs["unit"].map(r[c]).astype(str).to_numpy()
    ch, cm = th.obs["cell_ontology_class"].value_counts(), tm.obs["cell_ontology_class"].value_counts()
    allc = sorted(set(ch.index) | set(cm.index))
    classes = [c for c in allc if ch.get(c, 0) >= MIN_CELLS_MODEL and cm.get(c, 0) >= MIN_CELLS_MODEL]
    for c in allc:
        if c not in classes:
            filt.append(dict(stage="healthy_model", group="train_healthy", cell_type=c, unit="training nuclei",
                             n_nuclei=f"Human {int(ch.get(c, 0))}, Mouse {int(cm.get(c, 0))}",
                             rule=f"< {MIN_CELLS_MODEL} training nuclei in a species", action="class filtered from model"))
    pre = pd.DataFrame({"Human": ch, "Mouse": cm}).fillna(0).astype(int)
    th = th[th.obs["cell_ontology_class"].isin(classes).to_numpy()].copy()
    tm = tm[tm.obs["cell_ontology_class"].isin(classes).to_numpy()].copy()
    rng = np.random.default_rng(SEED)
    th = th[t72.cap_nuclei(th.obs, rng)].copy()
    tm = tm[t72.cap_nuclei(tm.obs, rng)].copy()
    obs_cols = ["sample", "donor", "species", "condition", "cell_ontology_class", "CellStateLevel2", "sex", "age",
                "assay", "source", "collection"]

    def _by_id(a: ad.AnnData) -> ad.AnnData:
        a.var_names = a.var["gene_ids"].astype(str).to_numpy()
        a = a[:, ~a.var_names.duplicated()].copy()
        a.obs = a.obs[obs_cols].copy()
        a.var = a.var[[]]
        a.X = sp.csc_matrix(a.X)  # column reads in construct_unified_adata (T12.1 note)
        return a

    th, tm = _by_id(th), _by_id(tm)
    prior = compute_conservation_prior(table)
    consensus, per_sp = download_and_process_esm2(table, species_list=["hsapiens", "mmusculus"],
                                                  saturn_data_dir=link_esm())
    esm_matched = {k: int(np.any(v != 0, axis=1).sum()) for k, v in per_sp.items()}
    esm_pca = pca_esm2(consensus, n_components=64)
    esm_var = compute_esm2_interspecies_variance(per_sp)
    G = len(table)
    unified = construct_unified_adata({"hsapiens": th, "mmusculus": tm}, table,
                                      np.zeros((G, 72), dtype=np.float32), prior)
    unified.X = sp.csr_matrix(unified.X, dtype=np.float32)
    stats = compute_expression_stats(unified)
    feats, scaler = assemble_gin_features(esm_pca, esm_var, table, stats)
    unified.varm["gene_features"] = feats
    unified.uns["p_null"] = precompute_p_null(unified)
    # batch = technical source within a species (canonical builder: source dataset; donors
    # live in sample_batch, never in batch; donor-level batch is the iter94 failure). The
    # training object holds one source per species (human KPMP multiome, mouse KPMP multiome
    # Ref), so batch is one value per species, the same partition as species.
    unified.obs["sample_batch"] = unified.obs["sample"].astype(str)
    unified.obs["batch"] = (unified.obs["species"].astype(str) + "|" + unified.obs["source"].astype(str)
                            + "|" + unified.obs["assay"].astype(str))
    unified.obs["cell_type"] = unified.obs["cell_ontology_class"].astype(str)
    for c in ("species", "batch", "sample_batch", "cell_type", *obs_cols):
        unified.obs[c] = unified.obs[c].astype(str).astype("category")
    # A group stays only if both genes were observed (count > 0) in their species' listed units:
    # a feature-listed gene never seen in a species is a structural zero, not an expression value.
    seen = (table["hsapiens_gene_id"].isin(observed["Human"]) & table["mmusculus_gene_id"].isin(observed["Mouse"])).to_numpy()
    nnz = np.asarray((unified.X != 0).sum(axis=0)).ravel()
    gene_mask = seen & (nnz >= 3)
    t72._write_h5ad(unified, OUT / "kidney_healthy_train_full_vocab.h5ad")
    obj = unified[:, gene_mask].copy()
    obj.uns["p_null"] = precompute_p_null(obj, species_key="species")
    with open(OUT / "kidney_healthy_train_gin_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    Xc = obj.X.tocsc()
    zero_nuclei, in_shared = {}, {}
    ct = pd.crosstab(obj.obs["cell_ontology_class"].astype(str), obj.obs["species"].astype(str))
    shared = (ct > 0).all(axis=1)
    for s_ in ("Human", "Mouse"):
        m_ = (obj.obs["species"].astype(str) == s_).to_numpy()
        zero_nuclei[s_] = int((np.asarray((Xc[m_] > 0).sum(axis=0)).ravel() == 0).sum())
        in_shared[s_] = float(ct.loc[shared, s_].sum() / ct[s_].sum())
    info = dict(classes=classes, n_genes_table=G, n_genes_never_observed_in_a_species=int((~seen).sum()),
                n_genes_lt3_nuclei_among_observed=int((seen & (nnz < 3)).sum()),
                n_object_genes_zero_training_nuclei=zero_nuclei, frac_cells_in_shared_class=in_shared,
                n_genes=int(gene_mask.sum()), esm2_matched=esm_matched,
                frac_prior_ge_095=float((obj.var["conservation_prior"] >= 0.95).mean()),
                frac_table_identity_ge_95=float((table["mean_identity"] >= 95).mean()),
                n_constant_feature_columns=int((feats.std(axis=0) == 0).sum()))
    return obj, info, pre


def object_checks(obj: ad.AnnData) -> dict:
    """Training-object contract: raw integer counts in X, no layers, required obs/var/uns/varm."""
    X = obj.X if sp.issparse(obj.X) else sp.csr_matrix(obj.X)
    d = X.data
    n_sp = obj.obs["species"].nunique()
    c = dict(
        x_integer=bool(len(d) == 0 or np.allclose(d, np.round(d))), x_nonnegative=bool(len(d) == 0 or d.min() >= 0),
        x_max=float(d.max()) if len(d) else 0.0, layers_empty=len(obj.layers) == 0,
        obs_columns_present=all(k in obj.obs for k in ("species", "cell_type", "cell_ontology_class", "batch")),
        one_batch_per_species=bool((obj.obs.groupby("species", observed=True)["batch"].nunique() == 1).all()
                                   and (obj.obs.groupby("batch", observed=True)["species"].nunique() == 1).all()),
        n_batches=int(obj.obs["batch"].nunique()), n_sample_batches=int(obj.obs["sample_batch"].nunique()),
        p_null_shape=list(np.asarray(obj.uns["p_null"]).shape),
        p_null_ok=list(np.asarray(obj.uns["p_null"]).shape) == [n_sp, obj.n_vars],
        p_null_rows_sum_1=bool(np.allclose(np.asarray(obj.uns["p_null"]).sum(axis=1), 1.0)),
        gene_features_shape=list(obj.varm["gene_features"].shape),
        gene_features_ok=obj.varm["gene_features"].shape == (obj.n_vars, 72),
        gene_features_finite=bool(np.isfinite(obj.varm["gene_features"]).all()),
        conservation_prior_present="conservation_prior" in obj.var,
    )
    c["all_ok"] = all(v for k, v in c.items() if isinstance(v, bool))
    return c


# --------------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    a = p.parse_args()
    res = TaskOutput("P13", a.outdir, prerun=a.prerun, force=a.force)
    roles = pd.read_csv(ROLES, sep="\t", dtype=str)
    ctmap = pd.read_csv(CTMAP, sep="\t", dtype=str).set_index("author_label")["cell_ontology_class"]
    table = pd.read_csv(TABLE)
    out = {}
    for s, keep in (("Mouse", keep_train_mouse), ("Human", keep_train_human)):
        obs, var = read_obs_units(FILES[s], s, roles)
        log.info("%s: %d nuclei in listed units", s, len(obs))
        out[s] = stream(FILES[s], s, obs, var, roles, ctmap, keep)
        pb = out[s]["pb"]
        pb.var["gene_symbol"] = pb.var_names
        for c in pb.obs.columns:
            if pb.obs[c].dtype == object:
                pb.obs[c] = pb.obs[c].astype(str)
        t72._write_h5ad(pb, OUT / f"kidney_pseudobulk_{s.lower()}.h5ad")
        res.csv(out[s]["qc"].rename_axis("unit").reset_index(), f"qc_{s.lower()}_per_unit.csv")
        res.csv(out[s]["mt"].rename_axis("unit").reset_index(), f"mt_filter_not_applied_{s.lower()}.csv")
        res.csv(out[s]["states"].rename("n_nuclei").reset_index(), f"states_{s.lower()}_unit_type.csv")
        res.csv(pb.obs[["unit", "role", "condition", "assay", "cell_ontology_class", "n_nuclei"]],
                f"counts_{s.lower()}_unit_type.csv")
    res.json({s: dict(n_rows_read=o["n_rows"], max_count=o["max_val"],
                      n_after_qc=int(o["qc"]["after"].sum()), n_before_qc=int(o["qc"]["before"].sum()))
              for s, o in out.items()} | dict(min_genes=MIN_GENES, mitochondrial_filter="none"), "qc_summary.json")

    crit, types, units, filt = gate_gd1k(out["Human"]["pb"], out["Mouse"]["pb"], roles)
    res.csv(crit, "gd1k_criteria.csv")
    res.csv(types, "gd1k_types.csv")
    res.csv(units.reset_index(), "gd1k_units.csv")
    gd1k = bool(crit["passed"].all())
    log.info("G-D1k %s\n%s\n%s", "PASS" if gd1k else "FAIL", crit.to_string(), types.to_string())

    observed = {}
    for s_, o_ in out.items():
        tot = np.asarray(o_["pb"].X.sum(axis=0)).ravel()
        observed[s_] = set(o_["pb"].var["gene_ids"].astype(str).to_numpy()[tot > 0])
    obj, info, pre = build_healthy(out["Human"]["train"], out["Mouse"]["train"], roles, table, filt, observed)
    res.csv(pd.DataFrame(filt), "filters.csv")
    res.csv(pre.rename_axis("cell_ontology_class").reset_index(), "counts_training_before_caps.csv")
    rho = gene_ids.assert_table_matches(obj, gene_ids.load_table("kidney"))
    checks = object_checks(obj)
    res.json(checks, "object_checks.json")
    if not checks["all_ok"]:
        raise AssertionError(f"healthy object checks failed: {checks}")
    path = gene_ids.h5ad_path("kidney")
    t72._write_h5ad(obj, path)
    from analysis.common.check_cohort import check as check_cohort

    checks["check_cohort_batch"] = check_cohort(str(path), batch_key="batch", celltype_key="cell_ontology_class")
    checks["check_cohort_species"] = check_cohort(str(path), batch_key="species", celltype_key="cell_ontology_class")
    res.json(checks, "object_checks.json")
    if checks["check_cohort_batch"] or checks["check_cohort_species"]:
        raise AssertionError(f"check_cohort failed: {checks}")
    res.csv(obj.obs.groupby(["species", "cell_ontology_class"], observed=True).size().rename("n_nuclei").reset_index(),
            "counts_healthy_object.csv")
    res.csv(obj.obs.groupby(["species", "sample"], observed=True).size().rename("n_nuclei").reset_index(),
            "counts_healthy_object_per_unit.csv")
    info.update(assert_table_matches_spearman=rho, n_obs=int(obj.n_obs), n_vars=int(obj.n_vars),
                p_null_shape=list(obj.uns["p_null"].shape), gd1k_pass=gd1k, cap=t72.CAP, seed=SEED,
                n_per_species=obj.obs["species"].value_counts().astype(int).to_dict(),
                n_batches=int(obj.obs["batch"].nunique()))
    res.json(info, "healthy_summary.json")
    paths = {"healthy": path, "healthy_full_vocab": OUT / "kidney_healthy_train_full_vocab.h5ad",
             "gin_scaler": OUT / "kidney_healthy_train_gin_scaler.pkl",
             "pseudobulk_human": OUT / "kidney_pseudobulk_human.h5ad",
             "pseudobulk_mouse": OUT / "kidney_pseudobulk_mouse.h5ad"}
    out_hash = {k: dict(path=str(v.relative_to(ROOT)), bytes=v.stat().st_size, sha256=t72.sha256(v))
                for k, v in paths.items()}
    in_hash = {str(x.relative_to(ROOT)): t72.sha256(x) for x in (*FILES.values(), ROLES, CTMAP, TABLE)}
    res.json(dict(outputs=out_hash, inputs=in_hash, code_sha=git_sha()), "hashes.json")
    res.finish(dict(gd1k_pass=gd1k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
