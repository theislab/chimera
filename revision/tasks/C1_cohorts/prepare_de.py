"""C1: per-unit pseudobulks for the disease contrasts (plan sections 13b, 13c). No Chimera readout is read here.

Output per case and species: results/revision/C1/<case>/pb_<species>.h5ad, X summed counts, layers['n_detected'], obs unit,
cell_ontology_class, condition (control or disease), covariates, n_nuclei; var_names symbol, var['gene_ids'] Ensembl ID.
Same QC as the models: at least 200 detected genes, no mitochondrial cut.

    python revision/tasks/C1_cohorts/prepare_de.py ad_human | ad_mouse | lung_human | lung_mouse
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp
from anndata.io import read_elem

from revision.tasks.C1_cohorts import configs as CF
from revision.tasks.C1_cohorts import orthologs as OR

SEAAD = {"microglia": CF.DATA / "ad_seaad/ab53c08f-03a3-4e0e-973a-44beabd78f60.h5ad",
         "astrocyte": CF.DATA / "ad_seaad/53f96bc7-3f23-444c-aa4f-4c69d62906bd.h5ad",
         "oligodendrocyte": CF.DATA / "ad_seaad/021a2e09-19c6-42eb-9e52-3ed2cf3d6032.h5ad"}
ZHOU = {"GSM4173504_WT_1": "control", "GSM4173505_WT_2": "control", "GSM4173506_WT_3": "control",
        "GSM4173510_WT_5XFAD_1": "disease", "GSM4173511_WT_5XFAD_2": "disease", "GSM4173512_WT_5XFAD_3": "disease"}
# Fixed in plan section 13b before any contrast.
MARKERS = {"microglia": ["Csf1r", "Cx3cr1", "P2ry12"], "astrocyte": ["Aqp4", "Gfap", "Aldh1l1"], "oligodendrocyte": ["Mog", "Plp1", "Mbp"],
           "opc": ["Pdgfra", "Cspg4"], "excitatory": ["Slc17a7"], "inhibitory": ["Gad1", "Gad2"], "endothelial": ["Flt1"]}
HAB = {"Macrophages": "macrophage", "AT2": "at2", "Endothelial Cells": "endothelial", "T Cells": "t", "cDCs": "dc", "Fibroblasts": "fibroblast", "Myofibroblasts": "fibroblast"}
STR = {"AM (Bleo)": "macrophage", "AM (PBS)": "macrophage", "M2 macrophages": "macrophage", "Resolution macrophages": "macrophage", "Recruited macrophages": "macrophage",
       "Fn1+ macrophages": "macrophage", "Cd163-/Cd11c+ IMs": "macrophage", "Cd163+/Cd11c- IMs": "macrophage", "AT2 cells": "at2", "Activated AT2 cells": "at2",
       "VECs": "endothelial", "Vcam1+ VECs": "endothelial", "CECs": "endothelial", "T-lymphocytes": "t", "Themis+ T-lymphocytes": "t", "T cell subset": "t",
       "DCs": "dc", "Cd103+ DCs": "dc", "Ccl17+ DCs": "dc", "Fibroblasts": "fibroblast", "Myofibroblasts": "fibroblast"}


def out_dir(case: str):
    d = CF.RES / case
    d.mkdir(parents=True, exist_ok=True)
    return d


def pb(X: sp.csr_matrix, obs: pd.DataFrame, var: pd.DataFrame, covs: list[str]) -> ad.AnnData:
    """X cells x genes; obs: unit, cell_ontology_class, condition, covs."""
    key = obs["unit"].astype(str) + "|" + obs["cell_ontology_class"].astype(str)
    cod, uniq = pd.factorize(key, sort=True)
    G = sp.csr_matrix((np.ones(len(cod)), (cod, np.arange(len(cod)))), shape=(len(uniq), len(cod)))
    Xf = X.astype(np.float64)
    S = (G @ Xf).toarray()
    Dn = (G @ (Xf > 0).astype(np.float64)).toarray()
    first = obs.groupby(key, observed=True)[["unit", "cell_ontology_class", "condition", *covs]].first().loc[uniq]
    first["n_nuclei"] = np.bincount(cod)
    first.index = uniq
    a = ad.AnnData(X=sp.csr_matrix(np.rint(S)), obs=first, var=var.copy())
    a.layers["n_detected"] = sp.csr_matrix(np.rint(Dn))
    return a


def unique_symbols(sym: pd.Series) -> pd.Index:
    return pd.Index(ad.utils.make_index_unique(pd.Index(sym.astype(str).to_numpy())), name=None)


def ad_human() -> None:
    parts, repo = [], {}
    for t, path in SEAAD.items():
        with h5py.File(path, "r") as f:
            o = f["obs"]
            obs = pd.DataFrame({c: read_elem(o[c]) for c in ("donor_id", "ADNC", "assay", "sex", "Neurotypical reference")}).astype(str)
            var = f["raw/var"]
            ids = pd.Index(read_elem(var[var.attrs["_index"]])).astype(str)
            names = pd.Series(read_elem(var["feature_name"]).astype(str))
            keep = (obs["assay"] == "10x 3' v3") & obs["ADNC"].isin(["High", "Not AD", "Low"]) & (obs["Neurotypical reference"] == "False")
            rows = np.flatnonzero(keep.to_numpy())
            g = f["raw/X"]
            indptr = g["indptr"][()]
            bl = []
            for p0 in range(0, len(rows), 20000):
                rb = rows[p0:p0 + 20000]
                lo, hi = int(indptr[rb[0]]), int(indptr[rb[-1] + 1])
                dat, ind = g["data"][lo:hi], g["indices"][lo:hi]
                st, ln = indptr[rb] - lo, indptr[rb + 1] - indptr[rb]
                off = np.r_[0, np.cumsum(ln)]
                sel = np.repeat(st - off[:-1], ln) + np.arange(off[-1])
                d = dat[sel].astype(np.float32)
                if (d != np.round(d)).any():
                    raise ValueError("not integers")
                bl.append(sp.csr_matrix((d, ind[sel], off), shape=(len(rb), len(ids))))
            X = sp.vstack(bl).tocsr()
        o2 = obs.iloc[rows].reset_index(drop=True)
        ng = np.asarray((X > 0).sum(1)).ravel()
        qc = ng >= CF.MIN_GENES
        repo[t] = dict(n_listed=int(len(rows)), n_qc=int(qc.sum()), by_adnc=o2[qc]["ADNC"].value_counts().to_dict())
        X, o2 = X[qc], o2[qc]
        o2 = o2.assign(unit=o2["donor_id"], cell_ontology_class=t, condition=np.where(o2["ADNC"] == "High", "disease", "control"), adnc_group=o2["ADNC"])
        v = pd.DataFrame({"gene_ids": ids, "feature_name": names.to_numpy()}, index=unique_symbols(names))
        parts.append(pb(X, o2, v, ["sex", "adnc_group"]))
    a = ad.concat(parts, join="inner", merge="same")
    a.var["gene_ids"] = parts[0].var.loc[a.var_names, "gene_ids"]
    a.write_h5ad(out_dir("ad") / "pb_human.h5ad")
    (out_dir("ad") / "pb_human_report.json").write_text(json.dumps(repo, indent=2))


def ad_mouse() -> None:
    bl, ob, var0 = [], [], None
    with tarfile.open(CF.DATA / "ad_5xfad" / "GSE140511_RAW.tar") as tar:
        for gsm, cond in ZHOU.items():
            rd = lambda s: io.BytesIO(gzip.decompress(tar.extractfile(f"{gsm}_{s}").read()))  # noqa: E731
            X = sio.mmread(rd("matrix.mtx.gz")).T.tocsr().astype(np.float32)
            feats = pd.read_csv(rd("features.tsv.gz"), sep="\t", header=None, dtype=str)
            bc = pd.read_csv(rd("barcodes.tsv.gz"), sep="\t", header=None, dtype=str)[0]
            if (X.data != np.round(X.data)).any():
                raise ValueError("not integers")
            if var0 is None:
                var0 = feats
            elif not var0[0].equals(feats[0]):
                raise ValueError("feature order differs")
            o = pd.DataFrame(dict(unit=gsm, condition=cond, barcode=bc.to_numpy()))
            ng = np.asarray((X > 0).sum(1)).ravel()
            qc = ng >= CF.MIN_GENES
            bl.append(X[qc])
            ob.append(o[qc])
            print(gsm, X.shape, int(qc.sum()), flush=True)
    X = sp.vstack(bl).tocsr()
    obs = pd.concat(ob, ignore_index=True)
    obs.index = obs["unit"] + "_" + obs["barcode"]
    names = var0[1]
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame({"gene_ids": var0[0].to_numpy()}, index=unique_symbols(names)))
    counts = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=3000, batch_key="unit")
    b = a[:, a.var.highly_variable].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=50, random_state=0)
    sc.pp.neighbors(b, n_neighbors=15, use_rep="X_pca", transformer="sklearn", random_state=0)
    sc.tl.leiden(b, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, random_state=0)
    cl = b.obs["leiden"].astype(str)
    rows = []
    for c in sorted(cl.unique(), key=int):
        m = (cl == c).to_numpy()
        sc_ = {k: float(np.asarray(a[m][:, [g for g in v if g in a.var_names]].X.mean(0)).mean()) for k, v in MARKERS.items()}
        s = pd.Series(sc_).sort_values(ascending=False)
        rows.append(dict(cluster=c, n=int(m.sum()), best=s.index[0], best_score=s.iloc[0], second=s.index[1], second_score=s.iloc[1], margin=s.iloc[0] - s.iloc[1]))
    tab = pd.DataFrame(rows)
    # A cluster gets its best class when the best marker mean leads the second by at least 0.2 log1p CP10K; otherwise unresolved.
    # First threshold 0.5 left no astrocyte cluster (Aqp4, Gfap, Aldh1l1 are low in nuclei; best astrocyte mean 0.73 vs 0.52); changed on
    # marker means and cluster sizes only, before any contrast.
    tab["label"] = np.where(tab["margin"] >= 0.2, tab["best"], "unresolved")
    mp = dict(zip(tab["cluster"], tab["label"]))
    obs["cell_ontology_class"] = cl.map(mp).to_numpy()
    tab.to_csv(out_dir("ad") / "mouse_cluster_labels.csv", index=False)
    pd.crosstab(obs["unit"], obs["cell_ontology_class"]).to_csv(out_dir("ad") / "mouse_counts_unit_type.csv")
    keep = obs["cell_ontology_class"].isin(["microglia", "astrocyte", "oligodendrocyte"]).to_numpy()
    pbm = pb(counts[keep], obs[keep], a.var[["gene_ids"]], [])
    pbm.write_h5ad(out_dir("ad") / "pb_mouse.h5ad")


def stream_h5ad(path, keep_col_fn, cols):
    """Rows selected by keep_col_fn(obs) from raw/X of one CELLxGENE file, integer check on what is read."""
    with h5py.File(path, "r") as f:
        o = f["obs"]
        obs = pd.DataFrame({c: read_elem(o[c]) for c in cols}).astype(str)
        var = f["raw/var"]
        ids = pd.Index(read_elem(var[var.attrs["_index"]])).astype(str)
        names = pd.Series(read_elem(var["feature_name"]).astype(str))
        rows = np.flatnonzero(keep_col_fn(obs).to_numpy())
        g = f["raw/X"]
        indptr = g["indptr"][()]
        bl = []
        for p0 in range(0, len(rows), 20000):
            rb = rows[p0:p0 + 20000]
            lo, hi = int(indptr[rb[0]]), int(indptr[rb[-1] + 1])
            dat, ind = g["data"][lo:hi], g["indices"][lo:hi]
            st, ln = indptr[rb] - lo, indptr[rb + 1] - indptr[rb]
            off = np.r_[0, np.cumsum(ln)]
            sel = np.repeat(st - off[:-1], ln) + np.arange(off[-1])
            d = dat[sel].astype(np.float32)
            if (d != np.round(d)).any():
                raise ValueError("not integers")
            bl.append(sp.csr_matrix((d, ind[sel], off), shape=(len(rb), len(ids))))
        X = sp.vstack(bl).tocsr()
    return X, obs.iloc[rows].reset_index(drop=True), ids, names


MUSCLE_DE_H = {"FB": "fb", "MuSC": "musc", "SMC": "smc", "Pericyte": "pericyte", "CapEC": "capec", "ArtEC": "artec", "VenEC": "venec", "Macrophage": "macrophage"}
MUSCLE_DE_M = {"FB": "fb", "MuSC": "musc", "SMC": "smc", "Pericyte": "pericyte", "CapEC": "capec", "ArtEC": "artec", "VenEC": "venec", "Macrophage-LYVE1": "macrophage", "Macrophage-LAM": "macrophage"}
MUS_H_OLD = ["seventh decade stage", "eighth decade stage"]
MUS_H_YOUNG = ["young adult stage", "third decade stage", "fourth decade stage"]
MUS_H_DE = ["339C", "343B", "362C", "367C", "411C", "464C", "470BR"]  # v2 cells; 640C is the one old v3 donor, 583B and 591C train


def muscle_human() -> None:
    """Cells of the cell fraction, donors not in training; old (7th, 8th decade) vs young; 6th decade not used."""
    f = CF.DATA / "muscle_ageing/5fd93c55-8047-48fb-905c-6ef2f7767483.h5ad"
    X, o, ids, names = stream_h5ad(f, lambda ob: (ob["batch"] == "cells") & ob["development_stage"].isin(MUS_H_OLD + MUS_H_YOUNG) & ob["annotation_level0"].isin(MUSCLE_DE_H) & ob["donor_id"].isin(MUS_H_DE), ["donor_id", "development_stage", "batch", "assay", "sex", "annotation_level0"])
    qc = np.asarray((X > 0).sum(1)).ravel() >= CF.MIN_GENES
    X, o = X[qc], o[qc]
    o = o.assign(unit=o["donor_id"], cell_ontology_class=o["annotation_level0"].map(MUSCLE_DE_H), condition=np.where(o["development_stage"].isin(MUS_H_OLD), "disease", "control"))
    v = pd.DataFrame({"gene_ids": ids}, index=unique_symbols(names))
    a = pb(X, o, v, ["sex"])
    a.write_h5ad(out_dir("muscle") / "pb_human.h5ad")


def muscle_mouse() -> None:
    """19-month donors 4, 6, 8 vs 3-month donors 3, 5, 7 (donors 1, 2 are in training)."""
    f = CF.DATA / "muscle_ageing/7085b988-463e-4183-b33d-838e1f6ad739.h5ad"
    X, o, ids, names = stream_h5ad(f, lambda ob: ob["donor_id"].isin(["Donor3", "Donor5", "Donor7", "Donor4", "Donor6", "Donor8"]) & (ob["suspension_type"] == "cell") & ob["annotation"].isin(MUSCLE_DE_M), ["donor_id", "development_stage", "annotation", "suspension_type"])
    qc = np.asarray((X > 0).sum(1)).ravel() >= CF.MIN_GENES
    X, o = X[qc], o[qc]
    o = o.assign(unit=o["donor_id"], cell_ontology_class=o["annotation"].map(MUSCLE_DE_M), condition=np.where(o["development_stage"].str.startswith("19"), "disease", "control"))
    v = pd.DataFrame({"gene_ids": ids}, index=unique_symbols(names))
    a = pb(X, o, v, [])
    a.write_h5ad(out_dir("muscle") / "pb_mouse.h5ad")


def sym_ids(symbols: pd.Series, tag: str) -> pd.Series:
    """Symbols through the release of the matrix (checklist 2b): Habermann Ensembl 84 (33,660 of 33,694 names, Cell Ranger GRCh38 1.2), Strunz Ensembl 75 (23,400 of 23,400)."""
    return symbols.map(OR.release_names(tag))


def lung_human() -> None:
    F = CF.DATA / "lung_fibrosis"
    meta = pd.read_csv(F / "GSE135893_IPF_metadata.csv.gz", index_col=0)
    bc = pd.read_csv(F / "GSE135893_barcodes.tsv.gz", header=None)[0].astype(str)
    genes = pd.read_csv(F / "GSE135893_genes.tsv.gz", header=None, sep="\t")[0].astype(str)
    X = sio.mmread(gzip.open(F / "GSE135893_matrix.mtx.gz")).T.tocsr().astype(np.float32)
    if (X.data != np.round(X.data)).any():
        raise ValueError("not integers")
    pos = pd.Series(np.arange(len(bc)), index=bc)
    meta = meta[meta["Diagnosis"].isin(["IPF", "Control"]) & meta["celltype"].isin(HAB)].copy()
    meta = meta[meta.index.isin(pos.index)]
    X = X[pos.loc[meta.index].to_numpy()]
    ng = np.asarray((X > 0).sum(1)).ravel()
    qc = ng >= CF.MIN_GENES
    X, meta = X[qc], meta[qc]
    meta = meta.assign(unit=meta["Sample_Name"], cell_ontology_class=meta["celltype"].map(HAB), condition=np.where(meta["Diagnosis"] == "IPF", "disease", "control"), source=meta["Sample_Source"])
    ids = sym_ids(genes, "hs84")
    v = pd.DataFrame({"gene_ids": ids.astype(object).where(ids.notna(), "NA_" + genes).to_numpy()}, index=unique_symbols(genes))
    a = pb(X, meta, v, ["source"])
    a.write_h5ad(out_dir("lung") / "pb_human.h5ad")
    (out_dir("lung") / "pb_human_report.json").write_text(json.dumps(dict(n_symbols=len(genes), n_mapped=int(ids.notna().sum()), n_cells=int(len(meta))), indent=2))


def lung_mouse() -> None:
    F = CF.DATA / "lung_fibrosis"
    info = pd.read_csv(F / "GSE141259_WholeLung_cellinfo.csv.gz", index_col=0)
    bc = pd.read_csv(F / "GSE141259_WholeLung_barcodes.txt.gz", header=None)[0].astype(str)
    genes = pd.read_csv(F / "GSE141259_WholeLung_genes.txt.gz", header=None)[0].astype(str)
    X = sio.mmread(gzip.open(F / "GSE141259_WholeLung_rawcounts.mtx.gz")).T.tocsr().astype(np.float32)
    if (X.data != np.round(X.data)).any():
        raise ValueError("not integers")
    pos = pd.Series(np.arange(len(bc)), index=bc)
    m = info[info["grouping"].isin(["PBS", "d10", "d14", "d21"]) & info["cell.type"].isin(STR)].copy()
    X = X[pos.loc[m.index].to_numpy()]
    ng = np.asarray((X > 0).sum(1)).ravel()
    qc = ng >= CF.MIN_GENES
    X, m = X[qc], m[qc]
    m = m.assign(unit=m["identifier"], cell_ontology_class=m["cell.type"].map(STR), condition=np.where(m["grouping"] == "PBS", "control", "disease"), day=m["grouping"])
    ids = sym_ids(genes, "mm75")
    v = pd.DataFrame({"gene_ids": ids.astype(object).where(ids.notna(), "NA_" + genes).to_numpy()}, index=unique_symbols(genes))
    a = pb(X, m, v, ["day"])
    a.write_h5ad(out_dir("lung") / "pb_mouse.h5ad")
    (out_dir("lung") / "pb_mouse_report.json").write_text(json.dumps(dict(n_symbols=len(genes), n_mapped=int(ids.notna().sum()), n_cells=int(len(m)), median_genes=float(np.median(ng[qc]))), indent=2))


if __name__ == "__main__":
    {"ad_human": ad_human, "ad_mouse": ad_mouse, "lung_human": lung_human, "lung_mouse": lung_mouse, "muscle_human": muscle_human, "muscle_mouse": muscle_mouse}[sys.argv[1]]()
