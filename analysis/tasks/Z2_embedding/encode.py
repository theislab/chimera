"""Held-out cells through trained encoders, Chimera and scVI, the same cells.
Liver H+Mouse models: TS (10x 3' v3) and TMS (10x) hepatocytes, TMS author zones. Cells only; no gene-level gold score is built or read.
Testis H+Mouse models: macaque cells of the testis H+Macaque object, a species neither model saw, on the shared testis ortholog table; fine stages.
Each cell set also rescaled to the training median library on the model's genes, to separate depth from the rest ("depth" = scaled).
python -m analysis.tasks.Z2_embedding.encode prep
python -m analysis.tasks.Z2_embedding.encode model <cohort> <chimera run | scvi:<dir>>
Output: results/analysis/Z2/encoded/<cohort>__<name>.parquet"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

from analysis.common import gene_ids
from analysis.tasks.Z2_embedding import common as K, rules as RU
from analysis.tasks.Z2_embedding.train_patched import apply_patch

OUT = RU.OUT / "encoded"
EXT = K.CACHE / "external"


def _take(X, pos):
    X = sp.csc_matrix(X).astype(np.float32)
    cols = [X[:, q] if q >= 0 else sp.csr_matrix((X.shape[0], 1), dtype=np.float32) for q in pos]
    return sp.hstack(cols).tocsr(), float((pos >= 0).mean())


def prep():
    EXT.mkdir(parents=True, exist_ok=True)
    # liver
    a = K.read("liv_hmouse")
    tb = gene_ids.load_table("liver").reindex([int(v.split("_")[1]) for v in a.var_names])
    ts = ad.read_h5ad(RU.REPO / "data/analysis/gold/G2/ts_liver.h5ad")
    t = ts[((ts.obs.assay.astype(str) == "10x 3' v3") & (ts.obs.free_annotation.astype(str) == "hepatocyte")).to_numpy()]
    A1, f1 = _take(t.raw[:, :].X, pd.Index(t.raw.var_names).get_indexer(tb.hsapiens_gene_id))
    tms = ad.read_h5ad(RU.REPO / "data/analysis/gold/G3/tms_liver_10x.h5ad")
    u = tms[tms.obs.free_annotation.astype(str).str.startswith("Hepatocyte").to_numpy()]
    A2, f2 = _take(u.raw[:, :].X, pd.Index(u.raw.var_names).get_indexer(tb.mmusculus_gene_id))
    sp.save_npz(EXT / "ts_hep.npz", A1)
    sp.save_npz(EXT / "tms_hep.npz", A2)
    pd.DataFrame(dict(free_annotation=u.obs.free_annotation.astype(str).to_numpy())).to_csv(EXT / "tms_hep_obs.csv", index=False)
    lib = np.asarray(a.X.sum(1)).ravel()
    hep = a.obs.cell_type.astype(str).to_numpy() == "Hepatocytes"
    spc = a.obs.species.astype(str).to_numpy()
    meta = dict(ts_share=f1, tms_share=f2, ts_lib=float(np.median(np.asarray(A1.sum(1)))), tms_lib=float(np.median(np.asarray(A2.sum(1)))),
                train_human_hep_lib=float(np.median(lib[hep & (spc == "Human")])), train_mouse_hep_lib=float(np.median(lib[hep & (spc == "Mouse")])))
    # testis: macaque through H+Mouse models
    b = K.read("test_hmouse")
    m = K.read("test_hmac")
    q = m[m.obs.species.astype(str).to_numpy() == "Macaque"]
    A3, f3 = _take(q.X, pd.Index(q.var_names).get_indexer(b.var_names))
    sp.save_npz(EXT / "macaque_testis.npz", A3)
    q.obs[["cell_type", "cell_ontology_class"]].astype(str).reset_index(drop=True).to_csv(EXT / "macaque_testis_obs.csv", index=False)
    libb = np.asarray(b.X.sum(1)).ravel()
    meta.update(mac_share=f3, mac_lib=float(np.median(np.asarray(A3.sum(1)))), train_human_testis_lib=float(np.median(libb[b.obs.species.astype(str).to_numpy() == "Human"])))
    (EXT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


def chimera_encoder(run: str, cohort: str):
    import torch
    sys.path.insert(0, str(RU.REPO / "analysis/tasks/T2_5_celltype_margin"))
    import run as T25
    d = RU.RUNS / run
    fm = d / "z2_input_mode.txt"
    apply_patch(fm.read_text().strip() if fm.exists() else "raw")
    a = K.read(cohort)
    comp = T25.load_components(d, a, json.loads((d / "config.json").read_text()))
    if comp["source"] != "ChimeraModel.load" and fm.exists() and fm.read_text().strip() != "raw":
        raise RuntimeError("state-dict fallback does not carry the input patch")
    z_train = K.load_z(d / "final_state.npz")

    def enc(X, spname):
        s = comp["species_names"].index(spname)
        out = []
        for st in range(0, X.shape[0], 512):
            x = torch.tensor(X[st:st + 512].toarray(), dtype=torch.float32)
            with torch.no_grad():
                out.append(comp["encode"](x, torch.full((x.shape[0],), s, dtype=torch.long), torch.zeros(x.shape[0], dtype=torch.long))[0].numpy())
        return np.vstack(out)
    return enc, z_train, a


def scvi_encoder(path: str, cohort: str):
    import scvi
    a = K.read(cohort)
    m = scvi.model.SCVI.load(path, adata=a)
    f = Path(path) / "batch_key.txt"
    key = f.read_text().strip() if f.exists() else ("sample_batch" if cohort.startswith("test") else "batch")   # Y5_aspects models carry the library key on testis
    bmap = a.obs.groupby("species", observed=True)[key].first().astype(str).to_dict()
    z_train = np.load(Path(path) / "z_mean.npy")

    def enc(X, spname):
        q = ad.AnnData(X=X.tocsr(), obs=pd.DataFrame({key: [bmap[spname]] * X.shape[0]}, index=[str(i) for i in range(X.shape[0])]), var=pd.DataFrame(index=a.var_names))
        scvi.model.SCVI.prepare_query_anndata(q, m)
        return m.get_latent_representation(q)
    return enc, z_train, a


def scale_to(X, target):
    L = np.maximum(np.asarray(X.sum(1)).ravel(), 1)
    return sp.diags(target / L).dot(X).tocsr()


def knn_mean(z_from, z_to, y_to, k=RU.K):
    return y_to[NearestNeighbors(n_neighbors=k).fit(z_to).kneighbors(z_from, return_distance=False)].mean(1)


def self_mean(z, y, k=RU.K):
    return y[NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z, return_distance=False)[:, 1:]].mean(1)


def liver_rows(enc, var_names):
    meta = json.loads((EXT / "meta.json").read_text())
    A1, A2 = sp.load_npz(EXT / "ts_hep.npz"), sp.load_npz(EXT / "tms_hep.npz")
    fa = pd.read_csv(EXT / "tms_hep_obs.csv").free_annotation.to_numpy()
    y = np.where(fa == "Hepatocyte (Pericentral)", 1, np.where(fa == "Hepatocyte (Periportal)", 0, -1))
    g = K.symbols("liv_hmouse", var_names)
    sym = g.symbol.astype(str).to_numpy()
    chrom = g.ens.map(pd.read_parquet(RU.REPO / "results/analysis/Y5/chromosomes.parquet").set_index("hsapiens_gene_id").chrom.astype(str)).astype(str).to_numpy()
    sc = {}
    for tag, chrs in (("all", None), ("odd", K.ODD), ("even", K.EVEN)):
        s1, _, _ = K.marker_score(K.lognorm(A1), np.arange(A1.shape[0]), sym, chrom, RU.HEP_ZONE, chrs)
        s2, _, _ = K.marker_score(K.lognorm(A2), np.arange(A2.shape[0]), sym, chrom, RU.HEP_ZONE, chrs)
        sc[tag] = (s1, s2)
    rows = []
    for depth, (B1, B2) in (("asis", (A1, A2)), ("depth", (scale_to(A1, meta["train_human_hep_lib"]), scale_to(A2, meta["train_mouse_hep_lib"])))):
        zt, zm = enc(B1, "Human"), enc(B2, "Mouse")
        lab = y >= 0
        Y5 = cross_val_predict(LogisticRegression(max_iter=2000), zm[lab], y[lab], cv=5, method="predict_proba")[:, 1]
        al, ak = K._auroc(zm[lab], y[lab], False)
        for m, v in (("tms_auroc_y5", roc_auc_score(y[lab], Y5)), ("tms_auroc_lr", al), ("tms_auroc_knn", ak)):
            rows.append(dict(set="tms", metric=m, depth=depth, marker="author", value=v, n=int(lab.sum())))
        for tag, (s1, s2) in sc.items():
            rows.append(dict(set="tms", metric="rho_self", depth=depth, marker=tag, value=spearmanr(s2, self_mean(zm, s2))[0], n=len(s2)))
            rows.append(dict(set="ts", metric="rho_self", depth=depth, marker=tag, value=spearmanr(s1, self_mean(zt, s1))[0], n=len(s1)))
            rows.append(dict(set="ts_to_tms", metric="rho_cross", depth=depth, marker=tag, value=spearmanr(s1, knn_mean(zt, zm, s2))[0], n=len(s1)))
            rows.append(dict(set="tms_to_ts", metric="rho_cross", depth=depth, marker=tag, value=spearmanr(s2, knn_mean(zm, zt, s1))[0], n=len(s2)))
        rows.append(dict(set="tms", metric="z_scale_sd", depth=depth, marker="na", value=float(zm.std(0).mean()), n=len(zm)))
    return rows


def testis_rows(enc, z_train, a):
    meta = json.loads((EXT / "meta.json").read_text())
    A3 = sp.load_npz(EXT / "macaque_testis.npz")
    o = pd.read_csv(EXT / "macaque_testis_obs.csv")
    hs = a.obs.species.astype(str).to_numpy() == "Human"
    hct = a.obs.cell_type.astype(str).to_numpy()
    rows = []
    for depth, B in (("asis", A3), ("depth", scale_to(A3, meta["train_human_testis_lib"]))):
        z = enc(B, "Human")
        for t, (cl, m) in {k: (v[0], v[2]) for k, v in RU.LABEL_TASKS["test_hmac"].items() if k != "round_vs_elongated"}.items():
            y = o.cell_type.map(m).to_numpy(float)
            q = np.isfinite(y)
            al, ak = K._auroc(z[q], y[q], len(np.unique(y[q])) > 2)
            rows.append(dict(set="macaque", metric="stage_auroc_lr", depth=depth, marker=t, value=al, n=int(q.sum())))
            rows.append(dict(set="macaque", metric="stage_auroc_knn", depth=depth, marker=t, value=ak, n=int(q.sum())))
            yh = pd.Series(hct).map(m).to_numpy(float)
            qh = np.isfinite(yh) & hs
            rows.append(dict(set="macaque_to_human_in_model", metric="rho_cross", depth=depth, marker=t, value=spearmanr(y[q], knn_mean(z[q], z_train[qh], yh[qh]))[0], n=int(q.sum())))
    return rows


def main():
    if sys.argv[1] == "prep":
        prep()
        return
    c, m = sys.argv[2], sys.argv[3]
    OUT.mkdir(parents=True, exist_ok=True)
    if m.startswith("scvi:"):
        path = m[5:]
        name = "scvi_" + Path(path).name
        f = OUT / f"{c}__{name}.parquet"
        if f.exists():
            return
        enc, z, a = scvi_encoder(path, c)
    else:
        name = m
        f = OUT / f"{c}__{name}.parquet"
        if f.exists():
            return
        enc, z, a = chimera_encoder(m, c)
    rows = liver_rows(enc, a.var_names) if c == "liv_hmouse" else testis_rows(enc, z, a)
    d = pd.DataFrame(rows)
    d.insert(0, "model", name)
    d.insert(0, "cohort", c)
    d.to_parquet(f)
    print(name, flush=True)


if __name__ == "__main__":
    main()
