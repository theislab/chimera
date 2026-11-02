"""Y5, comparator seeds and T3.1b (run after the owner asked for generous CPU use; amendment 8).

train <cohort> <seed>   scVI as the existing baselines (n_latent 10, n_hidden 128, 100 epochs, library batch key), model saved under
                        results/revision/Y5/scvi/ so it can encode new cells. Nothing is written to training/runs.
b1z_seeds               Leiden and centroid scores on every seed of scVI, CAMEX, SATURN that exists, against Chimera's seeds, odd and even.
encode <seed|scvi seed> TS and TMS hepatocytes through the trained encoder of a liver H+Mouse model.
t31b                    Independent-cohort zone tests on those latents.

Manifest for the array: results/revision/Y5/seeds_manifest.tsv
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from revision.common import cohorts as C, gene_ids
from revision.tasks.Y5_aspects import build as B, common as K, f3 as F3, rules as RU

SC = K.OUT / "scvi"
ENC = K.OUT / "F3" / "encoded"
BK = {"liver": "batch", "testis": "sample_batch"}


def train(c: str, seed: int):
    import scvi
    SC.mkdir(parents=True, exist_ok=True)
    d = SC / f"{c}_s{seed}"
    if (d / "z_mean.npy").exists():
        return
    scvi.settings.seed = seed
    a = C.read_adata(c)
    a.X = sp.csr_matrix(a.X)
    scvi.model.SCVI.setup_anndata(a, batch_key=BK[F3.tissue(c)])
    m = scvi.model.SCVI(a, n_latent=10, n_hidden=128)
    m.train(max_epochs=100, accelerator="cpu", enable_progress_bar=False)
    m.save(str(d), overwrite=True, save_anndata=False)
    np.save(d / "z_mean.npy", m.get_latent_representation())


def more_latents(c: str) -> dict:
    stem = gene_ids.H5ADS[c]
    out = {}
    for f in sorted(SC.glob(f"{c}_s*/z_mean.npy")):
        out["scvinew_" + f.parent.name.rsplit("_", 1)[1]] = np.load(f)
    for t in ("camex", "saturn"):
        for s in (42, 43, 44):
            f = C.RUNS_DIR / f"baselines_{t}" / f"{stem}-s{s}" / "z_mean.npy"
            if f.exists():
                out[f"{t}_s{s}"] = np.load(f)
    return out


def b1z_seeds():
    from revision.tasks.T4_2_gene_baselines import run as T42
    rows = []
    for c, gold in (("liv_hmouse", "G_H3"), ("liv_4sp", "G_H3"), ("test_3sp", "G_H4"), ("test_hmouse", "G_H4")):
        co = C.load_cohort(c, keep_adata=True, covariates=True)
        a, g0 = co.adata, co.genes
        X = sp.csr_matrix(a.X).astype(np.float64)
        spc = a.obs["species"].astype(str).to_numpy()
        lib = a.obs[T42.LIBRARY_KEY[F3.tissue(c)]].astype(str).to_numpy()
        S = sorted(set(spc))
        cov = g0[["prior_raw", "log_cp10k_spmean", "detection_spmean"]]
        g = K.genes(c)
        covs = K.CPAT + (["H4_nsp"] if gold == "G_H4" else [])
        L = more_latents(c)
        L.update({k: v for k, v in F3.latents(c).items() if k.startswith("chimera")})
        for nm, z in L.items():
            if z.shape[0] != X.shape[0]:
                continue
            cl = T42.leiden_clusters(np.asarray(z, np.float32))
            xz, fz, _ = T42.pseudobulk(X, spc, cl, lib, T42.MIN_CELLS_CLUSTER)
            Ez, _, _ = T42.pooled_E(xz, fz, S, X.shape[1])
            tab = pd.crosstab(cl, spc)
            sh = (tab >= T42.MIN_CELLS_CLUSTER).sum(1).ge(2)
            g2 = g.assign(_v=g.hsapiens_gene_id.map(pd.Series(T42.residualise(pd.Series(Ez, index=g0.index), cov).to_numpy(), index=g0.hsapiens_gene_id.values).to_dict()))
            for half in ("odd", "even"):
                T = g2[(g2.half == half) & g2[[gold, "block", "_v"] + covs].notna().all(1)]
                v = K.scan(T, gold, ["_v"], covs, draws=2).rho_p.iloc[0] if len(T) > 100 else np.nan
                rows.append(dict(cohort=c, latent=nm, half=half, rho_p=v, n=len(T), n_shared=int(sh.sum()), cells_in_shared=float(tab[sh].sum().sum() / tab.sum().sum())))
            print(c, nm, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(K.OUT / "F3/b1z_seeds.csv", index=False)


def external_counts(var_table: pd.DataFrame):
    """TS (10x 3' v3) and TMS hepatocytes in the model's gene order; missing genes are zero, and counted."""
    import anndata as ad
    ts = ad.read_h5ad(K.REPO / "data/revision/gold/G2/ts_liver.h5ad")
    m = (ts.obs.assay.astype(str) == "10x 3' v3") & (ts.obs.free_annotation.astype(str) == "hepatocyte")
    t = ts[m.to_numpy()]
    X1 = sp.csr_matrix(t.raw[:, :].X).astype(np.float32)
    pos = pd.Index(t.raw.var_names).get_indexer(var_table.hsapiens_gene_id)
    tms = ad.read_h5ad(K.REPO / "data/revision/gold/G3/tms_liver_10x.h5ad")
    fa = tms.obs.free_annotation.astype(str)
    m2 = fa.str.startswith("Hepatocyte")
    u = tms[m2.to_numpy()]
    X2 = sp.csr_matrix(u.raw[:, :].X).astype(np.float32)
    pos2 = pd.Index(u.raw.var_names).get_indexer(var_table.mmusculus_gene_id)
    def take(X, p):
        ok = p >= 0
        out = sp.lil_matrix((X.shape[0], len(p)), dtype=np.float32).tocsc()
        return sp.hstack([X[:, q] if q >= 0 else sp.csr_matrix((X.shape[0], 1), dtype=np.float32) for q in p]).tocsr(), float(ok.mean())
    A1, f1 = take(X1, pos)
    A2, f2 = take(X2, pos2)
    return A1, A2, t.obs.copy(), u.obs.copy(), f1, f2


def encode(which: str, seed: int):
    import torch
    ENC.mkdir(parents=True, exist_ok=True)
    f = ENC / f"{which}_s{seed}.npz"
    if f.exists():
        return
    c = "liv_hmouse"
    a = C.read_adata(c)
    tb = gene_ids.load_table("liver").reindex(a.var_names)
    A1, A2, o1, o2, f1, f2 = external_counts(tb)
    if which == "chimera":
        sys.path.insert(0, str(K.REPO / "revision/tasks/T2_5_celltype_margin"))
        import run as T25
        import json
        run = B.seed_policy(c)[seed]
        cfg = json.loads((C.RUNS_DIR / run / "config.json").read_text())
        comp = T25.load_components(C.RUNS_DIR / run, a, cfg)
        def enc(X, spname):
            out = []
            s = comp["species_names"].index(spname)
            for st in range(0, X.shape[0], 512):
                x = torch.tensor(X[st:st + 512].toarray(), dtype=torch.float32)
                si = torch.full((x.shape[0],), s, dtype=torch.long)
                bi = torch.zeros(x.shape[0], dtype=torch.long)
                with torch.no_grad():
                    out.append(comp["encode"](x, si, bi)[0].numpy())
            return np.vstack(out)
    else:
        import anndata as ad
        import scvi
        m = scvi.model.SCVI.load(str(SC / f"{c}_s{seed}"), adata=a)
        bmap = a.obs.groupby("species", observed=True)["batch"].first().astype(str).to_dict()
        def enc(X, spname):
            q = ad.AnnData(X=X.tocsr(), obs=pd.DataFrame({"batch": [bmap[spname]] * X.shape[0]}, index=[str(i) for i in range(X.shape[0])]), var=pd.DataFrame(index=a.var_names))
            scvi.model.SCVI.prepare_query_anndata(q, m)
            return m.get_latent_representation(q)
    np.savez_compressed(f, z_ts=enc(A1, "Human"), z_tms=enc(A2, "Mouse"), f1=f1, f2=f2)
    print(which, seed, f1, f2, flush=True)


def t31b():
    a = C.read_adata("liv_hmouse", backed=True)
    tb = gene_ids.load_table("liver").reindex(a.var_names)
    a.file.close()
    A1, A2, o1, o2, f1, f2 = external_counts(tb)
    g = K.genes("liv_hmouse").set_index("var_name").reindex(tb.index)
    sym = g.symbol.astype(str).to_numpy()
    chrom = g.chrom.astype(str).to_numpy()
    fa = o2.free_annotation.astype(str).to_numpy()
    y = np.where(fa == "Hepatocyte (Pericentral)", 1, np.where(fa == "Hepatocyte (Periportal)", 0, -1))
    def score(X, chrs):
        L = np.asarray(X.sum(1)).ravel()
        Xn = np.log1p(sp.diags(1e4 / np.maximum(L, 1)).dot(X).toarray())
        def m(lst):
            ix = [i for i, s in enumerate(sym) if s in lst and chrom[i] in chrs and (X[:, i] > 0).mean() >= .01]
            V = Xn[:, ix]
            return ((V - V.mean(0)) / V.std(0)).mean(1), len(ix)
        u, nu = m(RU.ZONE["hepatocyte"]["up"])
        d, nd = m(RU.ZONE["hepatocyte"]["dn"])
        return u - d, nu, nd
    rows = []
    sc = {h: (score(A1, chrs), score(A2, chrs)) for h, chrs in (("odd", RU.ODD), ("even", RU.EVEN))}
    for f in sorted(ENC.glob("*.npz")):
        z = np.load(f)
        zt, zm = z["z_ts"], z["z_tms"]
        lab = y >= 0
        pr = cross_val_predict(LogisticRegression(max_iter=2000), zm[lab], y[lab], cv=5, method="predict_proba")[:, 1]
        auc = roc_auc_score(y[lab], pr)
        ind = NearestNeighbors(n_neighbors=30).fit(zm).kneighbors(zt, return_distance=False)
        ind2 = NearestNeighbors(n_neighbors=30).fit(zt).kneighbors(zm, return_distance=False)
        for h, ((s1, n1u, n1d), (s2, n2u, n2d)) in sc.items():
            rows.append(dict(model=f.stem, half=h, tms_zone_auc=auc, rho_cross_ts_to_tms=spearmanr(s1, s2[ind].mean(1))[0], rho_cross_tms_to_ts=spearmanr(s2, s1[ind2].mean(1))[0],
                             rho_self_tms=spearmanr(s2, s2[NearestNeighbors(n_neighbors=31).fit(zm).kneighbors(zm, return_distance=False)[:, 1:]].mean(1))[0],
                             rho_self_ts=spearmanr(s1, s1[NearestNeighbors(n_neighbors=31).fit(zt).kneighbors(zt, return_distance=False)[:, 1:]].mean(1))[0],
                             n_up_ts=n1u, n_dn_ts=n1d, n_up_tms=n2u, n_dn_tms=n2d, n_ts=len(s1), n_tms=len(s2), genes_ts=float(z["f1"]), genes_tms=float(z["f2"])))
    pd.DataFrame(rows).to_csv(K.OUT / "F3/t31b.csv", index=False)


def manifest():
    rows = []
    for c in ("liv_hmouse", "liv_4sp", "test_3sp", "test_hmouse"):
        for s in range(42, 47):
            rows.append(f"train {c} {s}")
    for s in B.seed_policy("liv_hmouse"):
        rows.append(f"encode chimera {s}")
    for s in range(42, 47):
        rows.append(f"encode scvi {s}")
    (K.OUT / "seeds_manifest.tsv").write_text("\n".join(rows) + "\n")
    print(len(rows))


if __name__ == "__main__":
    k = sys.argv[1]
    if k == "manifest":
        manifest()
    elif k == "row":
        line = (K.OUT / "seeds_manifest.tsv").read_text().splitlines()[int(sys.argv[2])].split()
        train(line[1], int(line[2])) if line[0] == "train" else encode(line[1], int(line[2]))
    elif k == "b1z_seeds":
        b1z_seeds()
    elif k == "t31b":
        t31b()
