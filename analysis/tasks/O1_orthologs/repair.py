"""O2: canonical objects rebuilt with counts joined on Ensembl ID. Rules in ledgers/O2_repair.md.

The steps of notebooks/01_data_preparation.ipynb, in order and with the notebook's arguments; the repair acts on the unified object between
``construct_unified_adata`` and ``compute_expression_stats``. ``--mode control`` applies no repair and must return the canonical object.
Writes data/analysis/O1/repair_<tissue>/ (or control_<tissue>/) only.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.repair --tissue liver --mode control
    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.repair --tissue liver --mode repair
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from chimera._data_utils import (assemble_gin_features, compute_conservation_prior, compute_esm2_interspecies_variance, compute_expression_stats, construct_unified_adata,
                                 download_and_process_esm2, pca_esm2, precompute_p_null)
from analysis.tasks.O1_orthologs import rules as RU

warnings.filterwarnings("ignore", message=".*not unique.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, str(RU.REPO / "training"))
from build_subset import build_subset  # noqa: E402

ENS = {"Human": "hsapiens", "Monkey": "mmulatta", "Macaque": "mmulatta", "Mouse": "mmusculus", "Zebrafish": "drerio"}
PROC = RU.REPO / "data/processed"
CFG = {"liver": dict(raw="camex_pancreas.h5ad", uni="camex_pancreas_unified.h5ad", subsets={("Human", "Monkey"): "camex_human_monkey.h5ad", ("Human", "Mouse"): "camex_human_mouse.h5ad"}),
       "testis": dict(raw="camex_testis.h5ad", uni="camex_testis_unified.h5ad", subsets={("Human", "Mouse"): "camex_testis_human_mouse.h5ad"})}
TESTIS_LABELS = {"human": "Human", "homo sapiens": "Human", "mouse": "Mouse", "mus musculus": "Mouse", "macaque": "Macaque", "rhesus": "Macaque", "rhesus macaque": "Macaque", "monkey": "Macaque"}


def load_raw(tissue: str) -> ad.AnnData:
    a = ad.read_h5ad(RU.REPO / "data/raw" / CFG[tissue]["raw"])
    if tissue == "testis":  # notebook 8.2
        s = a.obs["species"].astype(str)
        c = s.str.lower().str.strip().map(TESTIS_LABELS)
        k = c.notna().to_numpy()
        a = a[k].copy()
        a.obs["species"] = pd.Categorical(c[k].astype(str))
        a.obs["batch"] = pd.Categorical(a.obs["batch"].astype(str))
    return a


def raw_for(tissue: str, species: str, cells: pd.Index):
    r = ad.read_h5ad(RU.RAW_DIR[tissue] / RU.RAW[tissue][species])
    r.var_names_make_unique()
    pos = r.obs_names.get_indexer(cells)
    if (pos < 0).any():
        alt = cells.str.replace(r"-[^-]*$", "", regex=True)
        pos = r.obs_names.get_indexer(alt)
    if (pos < 0).any() or len(set(pos)) != len(pos):
        raise AssertionError(f"{tissue} {species}: cells do not map one to one")
    ids = None
    for c in ("gene_ids", "gene"):
        if c in r.var:
            ids = r.var[c].astype(str).to_numpy()
    return sp.csc_matrix(sp.csr_matrix(r.X)[pos]).astype(np.int64), pd.Index(r.var_names.astype(str)), ids


def set_columns(X: sp.csc_matrix, cells: np.ndarray, changes: dict, R: sp.csc_matrix) -> sp.csc_matrix:
    """New matrix with, on ``cells`` (positions in X, in R's row order), column g = R[:, row]; row -1 gives zeros. Other entries untouched."""
    gs = np.array(sorted(changes))
    C = X.tocoo()
    in_cols, in_cells = np.zeros(X.shape[1], bool), np.zeros(X.shape[0], bool)
    in_cols[gs], in_cells[cells] = True, True
    keep = ~(in_cols[C.col] & in_cells[C.row])
    rows, cols, vals = [C.row[keep]], [C.col[keep]], [C.data[keep]]
    for g in gs:
        r = changes[g]
        if r >= 0:
            c = R[:, int(r)].tocoo()
            rows.append(cells[c.row])
            cols.append(np.full(c.nnz, g))
            vals.append(c.data)
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=X.shape, dtype=np.int64)


def repair(uni: ad.AnnData, tissue: str, table: pd.DataFrame) -> pd.DataFrame:
    X = sp.csc_matrix(uni.X).astype(np.int64)
    gi = np.array([int(v.split("_")[1]) for v in uni.var_names])
    t = table.set_index("ortholog_group_id").reindex(gi)
    log = []
    obs_names = pd.Index(uni.obs_names.astype(str))
    labels = uni.obs["species"].astype(str).to_numpy()
    for species in RU.RAW[tissue]:
        if species == "Zebrafish":
            continue
        cells = np.flatnonzero(labels == species)
        R, names, ids = raw_for(tissue, species, obs_names[cells])
        key = ENS[species]
        tid = t[f"{key}_gene_id"].astype(str).to_numpy()
        tname = t[f"{key}_gene_name"]
        before = np.asarray(X[cells][:, :].sum(0)).ravel()
        tot = np.asarray(R.sum(0)).ravel()
        ch = {}
        if ids is not None:  # ID route
            first = pd.Series(np.arange(len(ids)), index=ids)
            first = first[~first.index.duplicated()]
            id_row = first.reindex(tid).fillna(-1).astype(int).to_numpy()
            name_row = names.get_indexer(pd.Index(tname.fillna("__none__").astype(str)))
            others = pd.Series(np.arange(len(tid)), index=tid)
            for g in range(len(tid)):
                ir, nr = id_row[g], name_row[g]
                if ir >= 0:
                    if ir != nr or before[g] != tot[ir]:  # also tables with a repeated symbol: the name join fed only the last group
                        ch[g] = ir
                elif nr >= 0 and before[g] > 0:
                    rid = ids[nr]
                    if rid in others.index and others[rid] != g and rid != tid[g]:
                        ch[g] = -1
            ch = {g: r for g, r in ch.items() if not (r >= 0 and before[g] == 0 and tot[r] == 0)}  # zero on both sides: no change
            why = {g: ("no_row_counts_elsewhere" if r < 0 else ("false_zero" if before[g] == 0 else "wrong_row")) for g, r in ch.items()}
            name_row_v = name_row
        else:  # liver macaque: ID rows only, zero columns only
            c = pd.read_parquet(RU.DATA / "compara110_human_rows.parquet")
            mf = c[(c.species_ens == "macaca_fascicularis") & (c.source == "protein") & (c.homology_type == "ortholog_one2one")].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id").gene_id
            gid = t.hsapiens_gene_id.map(mf)
            pos = names.get_indexer(pd.Index(gid.fillna("__none__").astype(str)))
            used = set(names.get_indexer(pd.Index(tname.dropna().astype(str)))) - {-1}
            why, seen = {}, set()
            for g in range(len(tid)):
                p = pos[g]
                if before[g] == 0 and p >= 0 and tot[p] > 0 and p not in used and p not in seen:
                    ch[g] = int(p)
                    why[g] = "false_zero_mfas_id"
                    seen.add(p)
            id_row = pos
            name_row_v = names.get_indexer(pd.Index(tname.fillna("__none__").astype(str)))
        for g, r in ch.items():
            log.append(dict(tissue=tissue, species=species, var_name=uni.var_names[g], hsapiens_gene_id=t.hsapiens_gene_id.iloc[g], symbol=t.hsapiens_gene_name.iloc[g], table_id=tid[g], kind=why[g],
                            total_before=int(before[g]), total_after=int(tot[r]) if r >= 0 else 0, row=int(r)))
        print(f"{tissue} {species}: {len(ch)} columns change", pd.Series(list(why.values())).value_counts().to_dict(), flush=True)
        if ch:
            X = set_columns(X, cells, ch, R)
    uni.X = sp.csr_matrix(X).astype(uni.X.dtype)
    return pd.DataFrame(log)


def verify(uni: ad.AnnData, canon: ad.AnnData, tissue: str, table: pd.DataFrame, log: pd.DataFrame) -> dict:
    """Columns off the repair list equal canonical; columns with an ID row equal that row; no ID-route false zero remains."""
    rep = dict(same_shape=uni.shape == canon.shape, same_var=bool((uni.var_names == canon.var_names).all()), same_obs=bool((uni.obs_names == canon.obs_names).all()))
    gi = np.array([int(v.split("_")[1]) for v in uni.var_names])
    t = table.set_index("ortholog_group_id").reindex(gi)
    labels = uni.obs["species"].astype(str).to_numpy()
    Xn, Xc = sp.csc_matrix(uni.X).astype(np.int64), sp.csc_matrix(canon.X).astype(np.int64)
    rep["integer"] = bool(np.all(np.mod(Xn.data, 1) == 0))
    for species in RU.RAW[tissue]:
        cells = np.flatnonzero(labels == species)
        listed = set(log[log.species == species].var_name) if len(log) else set()
        off = np.array([v not in listed for v in uni.var_names])
        d = (Xn[cells] != Xc[cells]).tocsc()
        nn = np.asarray(d.sum(0)).ravel()
        rep[f"{species}_changed_off_list"] = int((nn[off] > 0).sum())
        rep[f"{species}_listed"] = len(listed)
        if species == "Zebrafish":
            continue
        R, names, ids = raw_for(tissue, species, pd.Index(uni.obs_names.astype(str))[cells])
        if ids is None:
            continue
        first = pd.Series(np.arange(len(ids)), index=ids)
        first = first[~first.index.duplicated()]
        id_row = first.reindex(t[f"{species and ENS[species]}_gene_id"].astype(str).to_numpy()).fillna(-1).astype(int).to_numpy()
        have = np.flatnonzero(id_row >= 0)
        same = np.asarray((Xn[cells][:, have] != R[:, id_row[have]]).sum(0)).ravel() == 0
        rep[f"{species}_id_columns"], rep[f"{species}_id_columns_equal"] = int(len(have)), int(same.sum())
        tn, to = np.asarray(R[:, id_row[have]].sum(0)).ravel(), np.asarray(Xn[cells][:, have].sum(0)).ravel()
        rep[f"{species}_false_zero_left"] = int(((tn > 0) & (to == 0)).sum())
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", choices=["liver", "testis"], required=True)
    ap.add_argument("--mode", choices=["control", "repair"], required=True)
    a = ap.parse_args()
    out = RU.DATA / f"{a.mode}_{a.tissue}"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(RU.TABLES[a.tissue])
    adata = load_raw(a.tissue)
    species_list = sorted(adata.obs["species"].unique().tolist())
    ens = [ENS[s] for s in species_list]
    prior = compute_conservation_prior(table)
    cons, per = download_and_process_esm2(table, species_list=ens, saturn_data_dir=RU.REPO / "data/raw/saturn_esm2")
    pca = pca_esm2(cons, n_components=64)
    var = compute_esm2_interspecies_variance(per)
    per_sp = {ENS[s]: adata[adata.obs["species"] == s].copy() for s in species_list}
    uni = construct_unified_adata(per_species_adatas=per_sp, ortholog_table=table, gene_features=np.zeros((len(table), 72), dtype=np.float32), conservation_prior=prior)
    canon = ad.read_h5ad(PROC / CFG[a.tissue]["uni"])
    rep = dict(tissue=a.tissue, mode=a.mode, shape=list(uni.shape), constructed_equals_canonical=int((sp.csr_matrix(uni.X) != sp.csr_matrix(canon.X)).nnz) == 0)
    log = pd.DataFrame(columns=["species", "var_name"])
    if a.mode == "repair":
        log = repair(uni, a.tissue, table)
        log.to_csv(out / "repair_log.csv", index=False)
        rep["n_changes"] = log.groupby(["species", "kind"]).size().rename("n").reset_index().to_dict("records")
        rep["verify"] = verify(uni, canon, a.tissue, table, log)
    stats = compute_expression_stats(uni)
    feats, scaler = assemble_gin_features(esm2_pca=pca, esm2_interspecies_var=var, ortholog_table=table, expression_stats=stats)
    uni.varm["gene_features"] = feats
    uni.uns["p_null"] = precompute_p_null(uni)
    with open(out / "gin_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    p4 = out / "unified.h5ad"
    uni.write_h5ad(p4)
    table.to_csv(out / "ortholog_table.csv", index=False)
    for pair, name in CFG[a.tissue]["subsets"].items():
        tag = "_".join(s.lower() for s in pair)
        sub = build_subset(p4, pair, out / f"{tag}.h5ad")
        c = ad.read_h5ad(PROC / name)
        r = dict(shape=list(sub.shape), canonical_shape=list(c.shape), genes_gained=int(len(set(sub.var_names) - set(c.var_names))), genes_lost=int(len(set(c.var_names) - set(sub.var_names))))
        if a.mode == "control":
            r["same_var"] = bool(c.n_vars == sub.n_vars and (c.var_names == sub.var_names).all())
            if r["same_var"]:
                d = np.abs(np.asarray(c.varm["gene_features"]) - np.asarray(sub.varm["gene_features"]))
                r.update(X_equal=int((sp.csr_matrix(c.X) != sp.csr_matrix(sub.X)).nnz) == 0, prior_max_abs=float(np.abs(c.var.conservation_prior.to_numpy() - sub.var.conservation_prior.to_numpy()).max()),
                         features_max_abs=float(d.max()), features_nonpca_max_abs=float(d[:, 64:].max()), p_null_max_rel=float(np.abs(c.uns["p_null"] / sub.uns["p_null"] - 1).max()))
        rep[tag] = r
    if a.mode == "control":
        rep["unified_X_equal"] = int((sp.csr_matrix(canon.X) != sp.csr_matrix(uni.X)).nnz) == 0
        rep["unified_prior_max_abs"] = float(np.abs(canon.var.conservation_prior.to_numpy() - uni.var.conservation_prior.to_numpy()).max())
    (out / "build_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
