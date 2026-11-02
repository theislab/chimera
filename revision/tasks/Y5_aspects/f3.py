"""Y5 F3: cell states inside an author label.

Sections: zone (T3.1), stage (T3.2), goldwithin + within (T3.3), b1z (T3.4), clusters (T3.5), encode (T3.1b).
Usage: python -m revision.tasks.Y5_aspects.f3 <section> [--cohort c]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, rankdata
from sklearn.neighbors import NearestNeighbors
from revision.common import cohorts as C, gene_ids
from revision.tasks.Y5_aspects import build as B, common as K, rules as RU

OUT = K.OUT / "F3"
LIB = {"liver": "batch", "testis": "sample_batch"}


def tissue(c):
    return "testis" if c.startswith("test") else "liver"


def latents(c: str) -> dict[str, np.ndarray]:
    """name -> z, every latent on this cohort's cells; Chimera healthy seeds plus each tool."""
    stem = gene_ids.H5ADS[c]
    bk = LIB[tissue(c)]
    out = {f"chimera_s{s}": np.asarray(C.load_state(r)["z_mean"], float) for s, r in B.seed_policy(c).items()}
    R = C.RUNS_DIR
    for s in (42, 43, 44):
        f = R / "baselines_scvi" / f"{stem}-s{s}-e100-nl10-bk_{bk}" / "z_mean.npy"
        if f.exists():
            out[f"scvi_s{s}"] = np.load(f)
    if c == "test_3sp":
        for s in (43, 44):
            f = R / "baselines_scvi" / f"{stem}-s{s}-e100-nl10" / "z_mean.npy"
            if f.exists():
                out[f"scvi_nobk_s{s}"] = np.load(f)
    for t, pat in (("scanvi", f"{stem}-s42-bk_{bk}"), ("sysvi", f"{stem}-s42-bk_{bk}"), ("harmony", f"{stem}-s42-bk_{bk}"),
                   ("pca", stem), ("saturn", f"{stem}-s42"), ("camex", f"{stem}-s42")):
        f = R / f"baselines_{t}" / pat / "z_mean.npy"
        if f.exists():
            out[t] = np.load(f)
    return out


def load_cells(c: str):
    a = C.read_adata(c)
    X = sp.csr_matrix(a.X).astype(np.float32)
    obs = a.obs[["species", "cell_type", "cell_ontology_class", LIB[tissue(c)]]].astype(str).copy()
    obs.columns = ["species", "cell_type", "cls", "lib"]
    return X, obs, np.asarray(a.var_names).astype(str)


def lognorm(X):
    L = np.asarray(X.sum(1)).ravel()
    return sp.diags(1e4 / np.maximum(L, 1)).dot(X)


def zone_scores(X, obs, g, cell_type: str, types: list[str]):
    """Literature axis score per cell, per chromosome half: mean z(up) - mean z(dn), z within (species, label)."""
    Xn = lognorm(X).tocsc()
    sym = g.set_index("symbol")
    out = {}
    cells = np.flatnonzero(obs.cell_type.isin(types))
    m = RU.ZONE[cell_type]
    info = []
    for half, chrs in (("odd", RU.ODD), ("even", RU.EVEN)):
        sc = np.full(X.shape[0], np.nan)
        for spc in obs.species.iloc[cells].unique():
            idx = cells[(obs.species.iloc[cells] == spc).to_numpy()]
            ups, dns = [], []
            for lst, arr in ((m["up"], ups), (m["dn"], dns)):
                for s in lst:
                    if s not in sym.index:
                        continue
                    row = sym.loc[[s]].iloc[0]
                    if str(row.chrom) not in chrs:
                        continue
                    j = g.index.get_loc(row.name) if False else int(row["_pos"])
                    v = np.log1p(np.asarray(Xn[idx, j].todense()).ravel())
                    if (v > 0).mean() >= 0.01 and v.std() > 0:
                        arr.append((v - v.mean()) / v.std())
            info.append(dict(cell_type=cell_type, half=half, species=spc, n_up=len(ups), n_dn=len(dns), n_cells=len(idx)))
            if ups and dns:
                sc[idx] = np.mean(ups, 0) - np.mean(dns, 0)
        out[half] = sc
    return out, pd.DataFrame(info)


def cross_rho(z, sc, spcs, keep, k=RU.KNN_CROSS):
    """rho_self per species and rho_cross per ordered species pair, on cells keep with finite score."""
    rows = []
    ok = keep & np.isfinite(sc)
    S = np.unique(spcs[ok])
    for a in S:
        ia = np.flatnonzero(ok & (spcs == a))
        if len(ia) < 40:
            continue
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(ia))).fit(z[ia])
        ind = nn.kneighbors(z[ia], return_distance=False)[:, 1:]
        rows.append(dict(a=a, b=a, kind="self", rho=spearmanr(sc[ia], sc[ia][ind].mean(1))[0], n=len(ia)))
        for b in S:
            if a == b:
                continue
            ib = np.flatnonzero(ok & (spcs == b))
            if len(ib) < 40:
                continue
            nn = NearestNeighbors(n_neighbors=min(k, len(ib))).fit(z[ib])
            ind = nn.kneighbors(z[ia], return_distance=False)
            rows.append(dict(a=a, b=b, kind="cross", rho=spearmanr(sc[ia], sc[ib][ind].mean(1))[0], n=len(ia)))
    return rows


def perm_null_cross(z, sc, spcs, keep, a, b, n_perm=200, seed=0, k=RU.KNN_CROSS):
    ok = keep & np.isfinite(sc)
    ia, ib = np.flatnonzero(ok & (spcs == a)), np.flatnonzero(ok & (spcs == b))
    nn = NearestNeighbors(n_neighbors=min(k, len(ib))).fit(z[ib])
    ind = nn.kneighbors(z[ia], return_distance=False)
    rng = np.random.default_rng(seed)
    obs = spearmanr(sc[ia], sc[ib][ind].mean(1))[0]
    null = [spearmanr(sc[ia], rng.permutation(sc[ib])[ind].mean(1))[0] for _ in range(n_perm)]
    return obs, float((1 + sum(n >= obs for n in null)) / (n_perm + 1)), float(np.mean(null)), float(np.std(null))


def zone(cohorts=("liv_hmouse", "liv_4sp", "liv_hmonkey")):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, info = [], []
    for c in cohorts:
        X, obs, var = load_cells(c)
        g = K.genes(c)
        g["_pos"] = pd.Index(var).get_indexer(g.var_name)
        spcs = obs.species.to_numpy()
        L = latents(c)
        for ct, types in RU.LIVER_TYPE.items():
            scs, i = zone_scores(X, obs, g, ct, types)
            i["cohort"] = c
            info.append(i)
            keep = obs.cell_type.isin(types).to_numpy()
            for half, sc in scs.items():
                for name, z in L.items():
                    if z.shape[0] != X.shape[0]:
                        print("skip", c, name, z.shape, flush=True)
                        continue
                    for r in cross_rho(z, sc, spcs, keep):
                        rows.append(dict(cohort=c, cell_type=ct, half=half, latent=name, **r))
        print(c, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "zone_rho.csv", index=False)
    pd.concat(info).to_csv(OUT / "zone_markers.csv", index=False)


def stage(cohorts=("test_3sp", "test_hmac")):
    """Fine stage rank within coarse class, across species, held-out libraries."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cohorts:
        X, obs, var = load_cells(c)
        spcs = obs.species.to_numpy()
        rk = obs.cell_type.map(RU.STAGE_RANK).to_numpy(float)
        clsm = obs.cell_type.map(RU.STAGE_CLASS).to_numpy(object)
        libs = {s: sorted(obs.lib[obs.species == s].unique()) for s in ("Human", "Macaque")}
        half_of = {}
        for s, l in libs.items():
            for i, x in enumerate(l):
                half_of[x] = "discovery" if i % 2 == 0 else "replication"
        hlf = obs.lib.map(half_of).to_numpy(object)
        for name, z in latents(c).items():
            if z.shape[0] != X.shape[0]:
                continue
            for half in ("discovery", "replication"):
                for cl in ("SG", "SC", "rSD"):
                    for a, b in (("Human", "Macaque"), ("Macaque", "Human")):
                        ia = np.flatnonzero((spcs == a) & (clsm == cl) & (hlf == half))
                        ib = np.flatnonzero((spcs == b) & (clsm == cl))          # all target-species cells
                        if len(ia) < 30 or len(ib) < 30:
                            continue
                        nn = NearestNeighbors(n_neighbors=RU.KNN_CROSS).fit(z[ib])
                        ind = nn.kneighbors(z[ia], return_distance=False)
                        pred = rk[ib][ind].mean(1)
                        lt = rk[ib][ind]
                        mode = np.array([np.bincount(v.astype(int)).argmax() for v in lt])
                        rows.append(dict(cohort=c, latent=name, half=half, cls=cl, a=a, b=b, n=len(ia),
                                         rho=spearmanr(rk[ia], pred)[0], transfer=float((mode == rk[ia]).mean()),
                                         chance=float(pd.Series(rk[ib]).value_counts(normalize=True).max())))
        print(c, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "stage_rho.csv", index=False)


# ------------------------------------------------------------------ T3.3 within-label evidence
def gold_zone():
    """G_zone from TMS author zones and TS marker score; independent cohorts. Leave-parity-out markers on TS."""
    import anndata as ad
    from revision.tasks.T4_3_scoreboard import golds as TG
    tbl = gene_ids.load_table("liver")
    print(tbl.columns.tolist()[:20])
    mouse_col = [k for k in tbl.columns if "mmusculus" in k and "gene_id" in k][0]
    human_col = "hsapiens_gene_id"
    chrom = pd.read_parquet(K.OUT / "chromosomes.parquet").set_index("hsapiens_gene_id").chrom
    tms = ad.read_h5ad(K.REPO / "data/revision/gold/G3/tms_liver_10x.h5ad")
    keep = tms.obs.free_annotation.astype(str).isin(["Hepatocyte (Pericentral)", "Hepatocyte (Periportal)"])
    t = tms[keep.to_numpy()]
    y = (t.obs.free_annotation.astype(str) == "Hepatocyte (Pericentral)").to_numpy()
    X = sp.csr_matrix(t.raw[:, :].X).astype(np.float64)
    var = pd.Index(t.raw.var_names)
    Xl = np.log1p(lognorm(X).toarray())
    m1, m0 = Xl[y].mean(0), Xl[~y].mean(0)
    s = np.sqrt(Xl[y].var(0) / y.sum() + Xl[~y].var(0) / (~y).sum())
    zt = pd.Series((m1 - m0) / np.where(s > 0, s, np.inf), index=var)
    print("TMS PC/PP", y.sum(), (~y).sum(), "var", var[:3].tolist())
    # TMS raw var names may be ENSMUSG or symbols; try both
    mp = tbl.set_index(mouse_col)[human_col].to_dict()
    if not any(v in mp for v in var[:200]):
        mcol = [k for k in tbl.columns if "mmusculus" in k and "name" in k.lower()][0]
        mp = tbl.set_index(mcol)[human_col].to_dict()
    tmsz = pd.Series({mp[v]: zt[v] for v in var if v in mp})
    # TS: per donor, gene vs marker score built from other-parity markers
    ts = ad.read_h5ad(K.REPO / "data/revision/gold/G2/ts_liver.h5ad")
    h = ts[(ts.obs.free_annotation.astype(str) == "hepatocyte").to_numpy()]
    Xh = sp.csr_matrix(h.raw[:, :].X).astype(np.float64)
    Xhl = np.log1p(lognorm(Xh).toarray())
    v2 = pd.Index(h.raw.var_names).str.replace(r"\.\d+$", "", regex=True)
    sym2 = pd.read_csv(K.REPO / "data/revision/x3_annotations/hgnc_complete_set.txt", sep="\t", usecols=["symbol", "ensembl_gene_id"]).dropna().set_index("symbol").ensembl_gene_id
    pos = {g: i for i, g in enumerate(v2)}
    ch2 = chrom.reindex(v2).to_numpy()
    res = {}
    donors = h.obs.donor_id.astype(str).to_numpy()
    for parity, chrs in (("odd", RU.ODD), ("even", RU.EVEN)):
        use = lambda L: [pos[sym2[s]] for s in L if s in sym2 and sym2[s] in pos and str(chrom.get(sym2[s])) in chrs]
        up, dn = use(RU.ZONE["hepatocyte"]["up"]), use(RU.ZONE["hepatocyte"]["dn"])
        zs = []
        for d in np.unique(donors):
            m = donors == d
            if m.sum() < 100:
                continue
            V = Xhl[m]
            zf = lambda ix: ((V[:, ix] - V[:, ix].mean(0)) / np.where(V[:, ix].std(0) > 0, V[:, ix].std(0), np.inf)).mean(1)
            sc = zf(up) - zf(dn)
            sc = (sc - sc.mean()) / sc.std()
            Vc = (V - V.mean(0)) / np.where(V.std(0) > 0, V.std(0), np.inf)
            zs.append(np.arctanh(np.clip((Vc * sc[:, None]).mean(0), -0.999, 0.999)) * np.sqrt(m.sum() - 3))
        res[parity] = np.nanmean(zs, 0)
    # score from even markers applies to odd genes, and the reverse
    z_ts = np.where(np.isin(ch2, list(RU.ODD)), res["even"], np.where(np.isin(ch2, list(RU.EVEN)), res["odd"], np.nan))
    tsz = pd.Series(z_ts, index=v2)
    tsz = tsz[~tsz.index.duplicated()]
    d = pd.DataFrame({"z_tms": tmsz}).join(pd.DataFrame({"z_ts": tsz}), how="outer")
    d["G_zone"] = np.sign(d.z_tms * d.z_ts) * np.sqrt(np.abs(d.z_tms * d.z_ts))
    d["G_zone_abs"] = np.sqrt(np.abs(d.z_tms * d.z_ts))
    d.index.name = "hsapiens_gene_id"
    OUT.mkdir(parents=True, exist_ok=True)
    d.reset_index().to_parquet(OUT / "gold_zone.parquet")
    print(d.describe().T)


def gold_stage():
    """G_stage: within coarse class stage pattern among the five Murat held-out primates (no model saw them)."""
    import itertools
    from revision.tasks.X7_architecture_outputs import golds as XG
    human, inp = XG.murat_inputs(fine=True)
    print(list(inp), {k: list(v["prof"]) for k, v in list(inp.items())[:1]})
    order = list(RU.STAGE_RANK)
    n = len(human)
    P = {}
    for s, d in inp.items():
        M = np.full((n, len(order)), np.nan)
        for j, t in enumerate(order):
            if t in d["prof"]:
                M[d["rows"], j] = d["prof"][t]
                # detection: keep value only for genes detected in the stage
                det = d["det"][t]
                M[d["rows"][det < 0.01], j] = np.nan if False else M[d["rows"][det < 0.01], j]
        cl = pd.Series(order).map(RU.STAGE_CLASS).to_numpy()
        for c in np.unique(cl):
            M[:, cl == c] -= np.nanmean(M[:, cl == c], axis=1, keepdims=True)
        P[s] = M
    Z = []
    sp_list = list(P)
    for a, b in itertools.combinations(sp_list, 2):
        A, B_ = P[a], P[b]
        ok = np.isfinite(A).all(1) & np.isfinite(B_).all(1)
        ra = (A - A.mean(1, keepdims=True))
        rb = (B_ - B_.mean(1, keepdims=True))
        with np.errstate(all="ignore"):
            r = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
        r[~ok] = np.nan
        Z.append(np.arctanh(np.clip(r, -0.999, 0.999)))
    Z = np.array(Z)
    d = pd.DataFrame({"hsapiens_gene_id": human, "G_stage": np.nanmean(Z, 0), "n_pairs": np.isfinite(Z).sum(0)})
    d = d[d.n_pairs >= 3]
    d.to_parquet(OUT / "gold_stage.parquet")
    print(d.describe().T)


def nb_ll(x, mu, th):
    from scipy.special import gammaln
    return (gammaln(x + th) - gammaln(th) - gammaln(x + 1) + th * (np.log(th) - np.log(th + mu)) + x * (np.log(mu) - np.log(th + mu)))


def theta_mom(Xd):
    m = Xd.mean(0)
    v = Xd.var(0)
    th = m ** 2 / np.maximum(v - m, 1e-3)
    return np.clip(th, 0.05, 100.0)


def knn_margin(X, z, spcs, cells, seed_k=RU.KNN_CROSS, level_free=False):
    """No-model margin on any latent: gene's fit from other-species neighbours minus species-and-label flat rate."""
    L = np.asarray(X.sum(1)).ravel().astype(float)
    S = np.unique(spcs[cells])
    G = X.shape[1]
    tot = np.zeros(G)
    nsp = 0
    for a in S:
        ia = cells[spcs[cells] == a]
        others = [b for b in S if b != a]
        ib = cells[np.isin(spcs[cells], others)]
        Xa = X[ia].toarray().astype(np.float32)
        Xb = sp.csr_matrix(X[ib]).astype(np.float32)
        pa = (Xa.sum(0) + .5) / (L[ia].sum() + .5 * G)
        pb_all = (np.asarray(Xb.sum(0)).ravel() + .5) / (L[ib].sum() + .5 * G)
        ind = NearestNeighbors(n_neighbors=seed_k).fit(z[ib]).kneighbors(z[ia], return_distance=False)
        th = theta_mom(Xa)
        ll_flat = np.zeros(G)
        ll_knn = np.zeros(G)
        for st in range(0, len(ia), 500):
            b = np.arange(st, min(st + 500, len(ia)))
            nb = ind[b]
            rows = np.repeat(np.arange(len(b)), nb.shape[1])
            W = sp.csr_matrix((np.ones(rows.size), (rows, nb.ravel())), shape=(len(b), len(ib)))
            cnt = W.dot(Xb).toarray()
            lib = W.dot(L[ib])
            p = (cnt + .5) / (lib[:, None] + .5 * G)
            if level_free:
                p = p * (pa / pb_all)
            mu1 = p * L[ia[b]][:, None] + 1e-8
            mu0 = pa[None, :] * L[ia[b]][:, None] + 1e-8
            ll_knn += nb_ll(Xa[b], mu1, th).sum(0)
            ll_flat += nb_ll(Xa[b], mu0, th).sum(0)
        tot += (ll_knn - ll_flat) / len(ia)
        nsp += 1
    return tot / nsp


def dispersion_rank(X, cells, spcs):
    """Mean over species of residual log dispersion on log mean, within the label."""
    out = []
    for a in np.unique(spcs[cells]):
        ia = cells[spcs[cells] == a]
        Xn = lognorm(X[ia]).toarray()
        m, v = Xn.mean(0), Xn.var(0)
        ok = m > 0
        lm, ld = np.log(m[ok]), np.log(v[ok] / m[ok] + 1e-12)
        bins = pd.qcut(lm, 20, labels=False, duplicates="drop")
        r = pd.Series(ld).groupby(bins).transform(lambda s: (s - s.median()) / (s.std() + 1e-9)).to_numpy()
        o = np.full(X.shape[1], np.nan)
        o[ok] = r
        out.append(o)
    return np.nanmean(out, 0)


def within(cohorts=("liv_hmouse", "liv_4sp", "test_3sp", "test_hmouse")):
    """Gene table: S_within per label group (healthy seeds), comparators on the same cells."""
    OUT.mkdir(parents=True, exist_ok=True)
    gz = pd.read_parquet(OUT / "gold_zone.parquet")
    gs = pd.read_parquet(OUT / "gold_stage.parquet")
    for c in cohorts:
        X, obs, var = load_cells(c)
        g = K.genes(c).set_index("var_name").reindex(var).reset_index()
        spcs = obs.species.to_numpy()
        tis = tissue(c)
        key = "cell_ontology_class" if tis == "testis" else "cell_type"
        lab = obs.cls.to_numpy() if tis == "testis" else obs.cell_type.to_numpy()
        targets = {"testis": ["Spermatogonia", "Spermatocytes", "spermatids"], "liver": ["Hepatocytes", "Endothelial", "fibroblasts"]}[tis]
        forward = {}
        for f in sorted((K.OUT / "forward").glob(f"{c}_s*.npz")):
            with np.load(f, allow_pickle=False) as d:
                grp = d[f"{key}__groups"]
                forward[f.stem] = {(int(a), str(b)): (d[f"{key}__sum_within"][i], d[f"{key}__sum_meanz"][i], int(n)) for i, (a, b, n) in enumerate(grp)}
        t = g[["var_name", "hsapiens_gene_id", "symbol", "chrom", "half", "masked", "block", "prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared", "S", "D_raw", "B1b", "B1z"]].copy()
        for lb in targets:
            cells = np.flatnonzero(lab == lb)
            sw, sm = [], []
            for run, gp in forward.items():
                vals = [(v[0] / v[2], v[1] / v[2]) for (s, l), v in gp.items() if l == lb and v[2] >= RU.PROFILE_MIN_CELLS]
                sw.append(np.mean([v[0] for v in vals], 0))
                sm.append(np.mean([v[1] for v in vals], 0))
            t[f"Swithin_{lb}"] = np.mean(sw, 0)
            t[f"Sflat_{lb}"] = np.mean(sm, 0)
            t[f"disp_{lb}"] = dispersion_rank(X, cells, spcs)
            Xn = np.log1p(lognorm(X[cells]).toarray())
            t[f"lev_{lb}"] = Xn.mean(0)
            t[f"det_{lb}"] = np.asarray((X[cells] > 0).mean(0)).ravel()
            L = latents(c)
            names = [n for n in L if n.startswith("chimera_")][:3]
            if "scvi_s42" in L:
                names += ["scvi_s42", "scvi_s43"] if "scvi_s43" in L else ["scvi_s42"]
            for t2 in ("harmony", "sysvi", "pca"):
                if t2 in L:
                    names.append(t2)
            for nm in names:
                t[f"knn_{nm}_{lb}"] = knn_margin(X, L[nm], spcs, cells)
            t[f"knnchim_{lb}"] = t[[f"knn_{n}_{lb}" for n in names if n.startswith("chimera_")]].mean(1)
            print(c, lb, len(cells), flush=True)
        t = t.merge(gz, on="hsapiens_gene_id", how="left").merge(gs, on="hsapiens_gene_id", how="left")
        t.to_parquet(OUT / f"within__{c}.parquet")


def within_tests():
    rows = []
    specs = [("liv_hmouse", "Hepatocytes", "G_zone"), ("liv_4sp", "Hepatocytes", "G_zone"), ("liv_hmouse", "Hepatocytes", "G_zone_abs"), ("liv_4sp", "Hepatocytes", "G_zone_abs"),
             ("test_3sp", "Spermatogonia", "G_stage"), ("test_3sp", "Spermatocytes", "G_stage"), ("test_3sp", "spermatids", "G_stage"),
             ("test_hmouse", "Spermatogonia", "G_stage"), ("test_hmouse", "Spermatocytes", "G_stage"), ("test_hmouse", "spermatids", "G_stage")]
    for c, lb, gold in specs:
        t = pd.read_parquet(OUT / f"within__{c}.parquet")
        scols = [f"Swithin_{lb}", f"Sflat_{lb}", f"disp_{lb}"] + [k for k in t.columns if k.startswith("knn_") and k.endswith(lb)]
        scols += ["S", "B1b", "B1z"]
        for half in ("odd", "even"):
            for gset, m in (("unmasked", ~t.masked.astype(bool)),):
                T = t[(t.half == half) & m & t[[gold, f"lev_{lb}", f"det_{lb}", "block", "prior_raw"] + scols].notna().all(1)]
                if len(T) < 100:
                    continue
                r = K.scan(T, gold, scols, [f"lev_{lb}", f"det_{lb}", "prior_raw"], seed=K.useed(c, lb, gold, half), delta_to=f"Swithin_{lb}")
                r["cohort"], r["label"], r["gold"], r["half"] = c, lb, gold, half
                rows.append(r)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "within_rho.csv", index=False)


# ------------------------------------------------------------------ T3.4
def b1z(cohorts=("liv_hmouse", "liv_4sp", "test_3sp", "test_hmouse")):
    from revision.tasks.T4_2_gene_baselines import run as T42
    OUT.mkdir(parents=True, exist_ok=True)
    for c in cohorts:
        co = C.load_cohort(c, keep_adata=True, covariates=True)
        a, g = co.adata, co.genes
        X = sp.csr_matrix(a.X).astype(np.float64)
        species = a.obs["species"].astype(str).to_numpy()
        library = a.obs[T42.LIBRARY_KEY[tissue(c)]].astype(str).to_numpy()
        species_list = sorted(set(species))
        cov = g[["prior_raw", "log_cp10k_spmean", "detection_spmean"]]
        res = pd.DataFrame({"var_name": g.index, "hsapiens_gene_id": g.hsapiens_gene_id.values})
        meta = []
        for nm, z in latents(c).items():
            if z.shape[0] != X.shape[0]:
                print("skip", nm, flush=True)
                continue
            cl = T42.leiden_clusters(np.asarray(z, np.float32))
            xz, fz, uz = T42.pseudobulk(X, species, cl, library, T42.MIN_CELLS_CLUSTER)
            Ez, nEz, _ = T42.pooled_E(xz, fz, species_list, X.shape[1])
            res[f"B1z_{nm}"] = T42.residualise(pd.Series(Ez, index=g.index), cov).to_numpy()
            res[f"B1zraw_{nm}"] = Ez
            res[f"B1zn_{nm}"] = nEz
            meta.append(dict(cohort=c, latent=nm, n_clusters=len(set(cl)), defined=float(np.isfinite(Ez).mean())))
            pd.DataFrame({"cl": cl}).to_parquet(OUT / f"clusters__{c}__{nm}.parquet")
            print(c, nm, meta[-1], flush=True)
        res.to_parquet(OUT / f"b1z__{c}.parquet")
        pd.DataFrame(meta).to_csv(OUT / f"b1z_meta__{c}.csv", index=False)


def b1z_tests():
    """Each latent against the mean of Chimera's healthy seeds, on the genes both define. Latents with no shared cluster are reported."""
    from revision.tasks.T4_2_gene_baselines import run as T42
    rows, ks = [], []
    for c, gold in (("liv_hmouse", "G_H3"), ("liv_4sp", "G_H3"), ("test_3sp", "G_H4"), ("test_hmouse", "G_H4")):
        g = K.genes(c)
        z = pd.read_parquet(OUT / f"b1z__{c}.parquet").drop(columns=["var_name"])
        g = g.merge(z, on="hsapiens_gene_id")
        names = [k for k in z.columns if k.startswith("B1z_")]
        live = [k for k in names if g[k].notna().mean() > 0]
        chim = [k for k in live if "chimera" in k]
        g["B1z_chimera_mean"] = g[chim].mean(1)
        covs = K.CPAT + (["H4_nsp"] if gold == "G_H4" else [])
        a = C.read_adata(c, backed=True)
        spc = a.obs["species"].astype(str).to_numpy()
        a.file.close()
        for nm in names:
            cl = pd.read_parquet(OUT / f"clusters__{c}__{nm[4:]}.parquet").cl.to_numpy()
            tab = pd.crosstab(cl, spc)
            ks.append(dict(cohort=c, latent=nm[4:], n_clusters=len(tab), n_shared=int((tab >= T42.MIN_CELLS_CLUSTER).sum(1).ge(2).sum()),
                           cells_in_shared=float(tab[(tab >= T42.MIN_CELLS_CLUSTER).sum(1).ge(2)].sum().sum() / tab.sum().sum()), defined=float(g[nm].notna().mean())))
        for half in ("odd", "even"):
            for other in [k for k in live if k not in chim] + chim + ["B1b", "S"]:
                T = g[(g.half == half) & g[[gold, "block", other, "B1z_chimera_mean"] + covs].notna().all(1)]
                if len(T) < 100:
                    continue
                r = K.scan(T, gold, ["B1z_chimera_mean", other], covs, seed=K.useed(c, gold, half, other), delta_to="B1z_chimera_mean", draws=1000)
                r = r[r.y == other]
                r["chimera_rho"] = K.scan(T, gold, ["B1z_chimera_mean"], covs, draws=2).rho_p.iloc[0]
                r["cohort"], r["gold"], r["half"] = c, gold, half
                rows.append(r)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "b1z_rho.csv", index=False)
    pd.DataFrame(ks).to_csv(OUT / "b1z_shared_clusters.csv", index=False)


def b1z_sweep(cohorts=("liv_hmouse", "test_3sp", "test_hmouse", "test_hmac", "liv_4sp")):
    """T3.4 sensitivity: every latent at several Leiden resolutions; label-free testis runs; fine-label direct E (human, macaque)."""
    from revision.tasks.T4_2_gene_baselines import run as T42
    default = T42.LEIDEN_RESOLUTION
    for c in cohorts:
        f = OUT / f"b1z_sweep__{c}.parquet"
        if f.exists():
            continue
        co = C.load_cohort(c, keep_adata=True, covariates=True)
        a, g = co.adata, co.genes
        X = sp.csr_matrix(a.X).astype(np.float64)
        species = a.obs["species"].astype(str).to_numpy()
        library = a.obs[T42.LIBRARY_KEY[tissue(c)]].astype(str).to_numpy()
        species_list = sorted(set(species))
        cov = g[["prior_raw", "log_cp10k_spmean", "detection_spmean"]]
        res = pd.DataFrame({"hsapiens_gene_id": g.hsapiens_gene_id.values})
        meta = []
        L = latents(c)
        chim = [k for k in L if k.startswith("chimera")][:3]
        use = {k: L[k] for k in chim}
        for k in ("scvi_s42", "scanvi", "sysvi", "harmony", "saturn", "camex"):
            if k in L:
                use[k] = L[k]
        if c.startswith("test"):
            for r in sorted(p.name for p in C.RUNS_DIR.glob(f"rev-ctl-{c}-nolab-s*")):
                z = np.asarray(C.load_state(r)["z_mean"], float)
                if np.isfinite(z).all():
                    use["chimeranolab_" + r.rsplit("-", 1)[-1]] = z
        for nm, z in use.items():
            if z.shape[0] != X.shape[0]:
                continue
            for rs in (default / 4, default / 2, default, default * 2, default * 4):
                T42.LEIDEN_RESOLUTION = rs
                cl = T42.leiden_clusters(np.asarray(z, np.float32))
                xz, fz, uz = T42.pseudobulk(X, species, cl, library, T42.MIN_CELLS_CLUSTER)
                Ez, nEz, _ = T42.pooled_E(xz, fz, species_list, X.shape[1])
                key = f"{nm}__r{rs:g}"
                res[key] = T42.residualise(pd.Series(Ez, index=g.index), cov).to_numpy()
                tab = pd.crosstab(cl, species)
                shared = (tab >= T42.MIN_CELLS_CLUSTER).sum(1).ge(2)
                meta.append(dict(cohort=c, latent=nm, res=rs, n_clusters=len(tab), n_shared=int(shared.sum()), cells_in_shared=float(tab[shared].sum().sum() / tab.sum().sum()), defined=float(np.isfinite(Ez).mean())))
                print(c, key, meta[-1]["n_shared"], flush=True)
        T42.LEIDEN_RESOLUTION = default
        if c == "test_hmac":
            lab = a.obs["cell_type"].astype(str).to_numpy()
            xz, fz, uz = T42.pseudobulk(X, species, lab, library, T42.MIN_CELLS_TYPE)
            Ez, nEz, _ = T42.pooled_E(xz, fz, species_list, X.shape[1])
            res["direct_fine_labels"] = T42.residualise(pd.Series(Ez, index=g.index), cov).to_numpy()
        res.to_parquet(f)
        pd.DataFrame(meta).to_csv(OUT / f"b1z_sweep_meta__{c}.csv", index=False)


def b1z_sweep_tests():
    rows = []
    for c, gold in (("liv_hmouse", "G_H3"), ("liv_4sp", "G_H3"), ("test_3sp", "G_H4"), ("test_hmouse", "G_H4"), ("test_hmac", "G_H4")):
        f = OUT / f"b1z_sweep__{c}.parquet"
        if not f.exists():
            continue
        z = pd.read_parquet(f)
        src = "test_hmouse" if c == "test_hmac" else c
        gg = K.genes(c)
        if gold not in gg:
            ref = K.genes(src)[["hsapiens_gene_id", gold, "H4_nsp", "tau_shared", "block"]].drop_duplicates("hsapiens_gene_id")
            gg = gg.merge(ref, on="hsapiens_gene_id", how="left")
        g = gg.merge(z, on="hsapiens_gene_id")
        keys = [k for k in z.columns if k != "hsapiens_gene_id"]
        covs = K.CPAT + (["H4_nsp"] if gold == "G_H4" else [])
        for half in ("odd", "even"):
            base = g[(g.half == half) & g[[gold, "block"] + covs].notna().all(1)]
            for k in keys:
                T = base[base[k].notna()]
                if len(T) < 100:
                    rows.append(dict(cohort=c, half=half, key=k, n=len(T), rho_p=np.nan))
                    continue
                r = K.scan(T, gold, [k], covs, seed=K.useed(c, half, k), draws=300).iloc[0]
                rows.append(dict(cohort=c, half=half, key=k, n=len(T), rho_p=r.rho_p, lo=r.ci_lo, hi=r.ci_hi))
            # common genes across every default-resolution latent that exists
            dk = [k for k in keys if k.endswith("__r1") or k == "direct_fine_labels"]
            alive = [k for k in dk if base[k].notna().mean() > .3]
            T = base[base[alive].notna().all(1)]
            if len(T) > 100 and alive:
                ref = [k for k in alive if k.startswith("chimera_")][0]
                r = K.scan(T, gold, alive, covs, seed=K.useed(c, half, "common"), delta_to=ref, draws=1000)
                r["cohort"], r["half"], r["universe"] = c, half, "common"
                rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "b1z_sweep_rho.csv", index=False)


def clusters():
    rows = []
    for c in ("liv_hmouse", "liv_4sp", "test_3sp"):
        X, obs, var = load_cells(c)
        g = K.genes(c).set_index("var_name").reindex(var)
        sym = g.symbol.astype(str).to_numpy()
        nameable = g.nameable.fillna(False).to_numpy() if "nameable" in g else np.ones(len(g), bool)
        Xn = np.log1p(lognorm(X).toarray())
        for f in sorted(OUT.glob(f"clusters__{c}__*.parquet")):
            nm = f.stem.split("__")[2]
            if not (nm.startswith("chimera") or nm.startswith("scvi_s42")):
                continue
            cl = pd.read_parquet(f).cl.to_numpy()
            for k in np.unique(cl):
                m = cl == k
                sp_share = obs.species[m].value_counts(normalize=True)
                lb = obs.cell_type[m].value_counts(normalize=True)
                lfc = Xn[m].mean(0) - Xn[~m].mean(0)
                top = np.argsort(-lfc)[:12]
                rows.append(dict(cohort=c, latent=nm, cluster=k, n=int(m.sum()), species=";".join(f"{a}:{b:.2f}" for a, b in sp_share.items()),
                                 max_species=float(sp_share.iloc[0]), label=";".join(f"{a}:{b:.2f}" for a, b in lb.head(3).items()),
                                 top_genes=",".join(sym[top])))
    pd.DataFrame(rows).to_csv(OUT / "clusters_described.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("section")
    p.add_argument("--cohort", default=None)
    a = p.parse_args()
    fn = dict(zone=zone, stage=stage, gold_zone=gold_zone, gold_stage=gold_stage, within=within, within_tests=within_tests, b1z=b1z, b1z_tests=b1z_tests, b1z_sweep=b1z_sweep, b1z_sweep_tests=b1z_sweep_tests, clusters=clusters)[a.section]
    fn((a.cohort,)) if a.cohort else fn()
