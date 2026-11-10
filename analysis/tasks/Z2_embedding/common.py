"""Z2 shared: cohort cells, within-label targets (cached), latents, metrics. Every latent gets the same cells, the same k, the same functions."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.common import gene_ids
from analysis.tasks.Z2_embedding import rules as RU

CACHE = RU.OUT / "cache"
ODD = {str(i) for i in range(1, 23, 2)}
EVEN = {str(i) for i in range(2, 23, 2)}
LIB_KEY = {"test_hmouse": "sample_batch", "test_3sp": "sample_batch", "test_hmac": "sample_batch", "kidney": "sample_batch", "islet": "sample_batch", "bone_marrow": "batch"}


def read(c: str):
    a = ad.read_h5ad(RU.REPO / RU.H5AD[c])
    a.X = sp.csr_matrix(a.X).astype(np.float32)
    return a


def symbols(c: str, var_names) -> pd.DataFrame:
    tab = RU.TABLE.get(c, "van_zyl" if c.startswith("vz_") else c)
    return pd.DataFrame(dict(symbol=gene_ids.symbols(tab, var_names).values, ens=gene_ids.ensembl(tab, var_names).values), index=pd.Index(var_names))


def lognorm(X):
    L = np.asarray(X.sum(1)).ravel()
    Xn = sp.diags(1e4 / np.maximum(L, 1)).dot(X).tocsr()
    Xn.data = np.log1p(Xn.data)
    return Xn


def marker_score(Xn, cells: np.ndarray, sym: np.ndarray, chrom: np.ndarray, markers: dict, chrs=None, min_det: float = RU.MIN_DET):
    """mean z(up) - mean z(dn) within the given cells; z within them. Markers below detection or outside chrs leave. Returns score, n_up, n_dn."""
    M = Xn[cells]
    out = {}
    for side in ("up", "dn"):
        vs = []
        for s in markers[side]:
            for j in np.flatnonzero(sym == s):
                if chrs is not None and chrom[j] not in chrs:
                    continue
                v = np.asarray(M[:, j].todense()).ravel()
                if (v > 0).mean() >= min_det and v.std() > 0:
                    vs.append((v - v.mean()) / v.std())
        out[side] = vs
    if not out["up"] or not out["dn"]:
        return None, len(out["up"]), len(out["dn"])
    return np.mean(out["up"], 0) - np.mean(out["dn"], 0), len(out["up"]), len(out["dn"])


def prep(c: str) -> None:
    """Targets per cohort, cached: tasks and within-label PCs. Depends on counts and labels only, never on a latent."""
    f = CACHE / f"{c}_targets.npz"
    if f.exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    a = read(c)
    X = a.X
    Xn = lognorm(X)
    spc = a.obs["species"].astype(str).to_numpy().astype("U")
    lab = a.obs[RU.LABEL_KEY[c]].astype(str).to_numpy().astype("U")
    tasks, info = {}, []
    if c in RU.SCORE_TASKS:
        g = symbols(c, a.var_names)
        sym = g.symbol.astype(str).to_numpy()
        chrom_tab = pd.read_parquet(RU.REPO / "results/analysis/Y5/chromosomes.parquet").set_index("hsapiens_gene_id").chrom.astype(str)
        chrom = g.ens.map(chrom_tab).astype(str).to_numpy()
        for t, (types, m) in RU.SCORE_TASKS[c].items():
            for suffix, chrs in (("", None), ("__odd", ODD), ("__even", EVEN)):
                y = np.full(X.shape[0], np.nan)
                for s in np.unique(spc):
                    cells = np.flatnonzero((spc == s) & np.isin(lab, types))
                    if len(cells) < RU.MIN_TASK_CELLS:
                        continue
                    sc, nu, nd = marker_score(Xn, cells, sym, chrom, m, chrs)
                    info.append(dict(cohort=c, task=t + suffix, species=s, n_up=nu, n_dn=nd, n_cells=len(cells)))
                    if sc is not None:
                        y[cells] = sc
                tasks[f"score:{t}{suffix}"] = y
    for t, (cl, col, m) in RU.LABEL_TASKS.get(c, {}).items():
        v = a.obs[col].astype(str).map(m).to_numpy(float)
        y = np.where(lab == cl, v, np.nan)
        for s in np.unique(spc):
            q = (spc == s) & np.isfinite(y)
            info.append(dict(cohort=c, task=t, species=s, n_up=-1, n_dn=-1, n_cells=int(q.sum()), levels=json.dumps({str(k): int((y[q] == k).sum()) for k in np.unique(y[q])}) if q.any() else ""))
        tasks[f"rank:{t}"] = y
    # within-label PCs, with and without library means
    lk = LIB_KEY.get(c)
    libs = a.obs[lk].astype(str).to_numpy() if lk else None
    pcs = {}
    lib_size = np.log1p(np.asarray(X.sum(1)).ravel())
    for s in np.unique(spc):
        for l in np.unique(lab[spc == s]):
            cells = np.flatnonzero((spc == s) & (lab == l))
            if len(cells) < RU.MIN_LABEL_CELLS:
                continue
            M = Xn[cells]
            mu = np.asarray(M.mean(0)).ravel()
            var = np.asarray(M.multiply(M).mean(0)).ravel() - mu ** 2
            hv = np.argsort(-var)[: RU.N_HVG]
            D = np.asarray(M[:, hv].todense(), dtype=np.float32)
            for tag, Dc in (("plain", D - D.mean(0)), ("libcentred", _lib_centre(D, libs[cells]) if libs is not None else None)):
                if Dc is None:
                    continue
                k = min(RU.N_PCS_WITHIN, min(Dc.shape) - 1)
                sv = TruncatedSVD(n_components=k, random_state=0).fit(Dc)
                pcs[f"{tag}|{s}|{l}|scores"] = sv.transform(Dc).astype(np.float32)
                pcs[f"{tag}|{s}|{l}|var"] = sv.explained_variance_.astype(np.float32)
            pcs[f"cells|{s}|{l}"] = cells
    np.savez_compressed(f, spc=spc, lab=lab, lib_size=lib_size, **tasks, **pcs)
    pd.DataFrame(info).to_csv(CACHE / f"{c}_task_info.csv", index=False)


def _lib_centre(D, libs):
    out = D.copy()
    for l in np.unique(libs):
        q = libs == l
        out[q] -= out[q].mean(0)
    return out


def targets(c: str):
    return np.load(CACHE / f"{c}_targets.npz", allow_pickle=False)


def _auroc(z, y, multi: bool):
    """CV AUROC, standardised logistic regression and kNN (k 30). Five stratified folds, seed 0. Macro one-vs-rest if more than two levels."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    lv = np.unique(y)
    pl, pk = np.zeros((len(y), len(lv))), np.zeros((len(y), len(lv)))
    for tr, te in skf.split(z, y):
        pl[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(z[tr], y[tr]).predict_proba(z[te])
        pk[te] = KNeighborsClassifier(n_neighbors=min(RU.K, len(tr) - 1)).fit(z[tr], y[tr]).predict_proba(z[te])
    if multi:
        return roc_auc_score(y, pl, multi_class="ovr", average="macro"), roc_auc_score(y, pk, multi_class="ovr", average="macro")
    return roc_auc_score(y, pl[:, 1]), roc_auc_score(y, pk[:, 1])


def clisi(z, y, perplexity: int = RU.PERPLEXITY) -> float:
    """scib-style cLISI on the levels of y: (K - median LISI) / (K - 1); 1 is pure."""
    n = len(y)
    k = min(3 * perplexity, n - 1)
    d, idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z)
    d, idx = d[:, 1:] ** 2, idx[:, 1:]
    H = np.log(perplexity)
    beta = np.ones(n)
    lo, hi = np.zeros(n), np.full(n, np.inf)
    for _ in range(50):
        p = np.exp(-d * beta[:, None])
        s = p.sum(1, keepdims=True).clip(min=1e-300)
        pn = p / s
        h = -(pn * np.log(pn.clip(min=1e-300))).sum(1)
        big = h > H
        lo[big] = beta[big]
        hi[~big] = beta[~big]
        beta = np.where(np.isinf(hi), beta * 2, (lo + hi) / 2)
    p = np.exp(-d * beta[:, None])
    pn = p / p.sum(1, keepdims=True).clip(min=1e-300)
    lv = np.unique(y)
    L = 1 / np.sum([np.square((pn * (y[idx] == v)).sum(1)) for v in lv], 0).clip(min=1e-12)
    return float((len(lv) - np.median(L)) / (len(lv) - 1))


def graph_conn(z, y, k: int = RU.K_GRAPH) -> float:
    idx = NearestNeighbors(n_neighbors=min(k + 1, len(y))).fit(z).kneighbors(z, return_distance=False)
    vals = []
    for v in np.unique(y):
        q = np.flatnonzero(y == v)
        if len(q) < 2:
            continue
        pos = -np.ones(len(y), int)
        pos[q] = np.arange(len(q))
        r, cc = [], []
        for i, nb in zip(q, idx[q]):
            for j in nb[1:]:
                if pos[j] >= 0:
                    r.append(pos[i])
                    cc.append(pos[j])
        g = sp.csr_matrix((np.ones(len(r)), (r, cc)), shape=(len(q), len(q)))
        _, lab = connected_components(g, directed=False)
        vals.append(np.bincount(lab).max() / len(q))
    return float(np.mean(vals))


def task_rows(z, y_all, spc, task: str) -> list[dict]:
    kind, name = task.split(":", 1)
    rows = []
    ok = np.isfinite(y_all)
    S = [s for s in np.unique(spc[ok]) if (ok & (spc == s)).sum() >= RU.MIN_TASK_CELLS]
    for a in S:
        ia = np.flatnonzero(ok & (spc == a))
        ind = NearestNeighbors(n_neighbors=min(RU.K + 1, len(ia))).fit(z[ia]).kneighbors(z[ia], return_distance=False)[:, 1:]
        rows.append(dict(task=name, kind=kind, metric="rho_self", a=a, b=a, value=spearmanr(y_all[ia], y_all[ia][ind].mean(1))[0], n=len(ia)))
        for b in S:
            if a == b:
                continue
            ib = np.flatnonzero(ok & (spc == b))
            ind = NearestNeighbors(n_neighbors=min(RU.K, len(ib))).fit(z[ib]).kneighbors(z[ia], return_distance=False)
            rows.append(dict(task=name, kind=kind, metric="rho_cross", a=a, b=b, value=spearmanr(y_all[ia], y_all[ib][ind].mean(1))[0], n=len(ia)))
        ya = y_all[ia]
        za = z[ia]
        if kind == "score":
            t1, t2 = np.quantile(ya, [1 / 3, 2 / 3])
            q = (ya <= t1) | (ya >= t2)
            za2, y2 = za[q], (ya[q] >= t2).astype(int)
        else:
            vc = pd.Series(ya).value_counts()
            keep = vc.index[vc >= 20]
            q = np.isin(ya, keep)
            za2, y2 = za[q], ya[q]
        if len(np.unique(y2)) < 2:
            continue
        al, ak = _auroc(za2, y2, len(np.unique(y2)) > 2)
        rows.append(dict(task=name, kind=kind, metric="auroc_lr", a=a, b=a, value=al, n=len(y2)))
        rows.append(dict(task=name, kind=kind, metric="auroc_knn", a=a, b=a, value=ak, n=len(y2)))
        if kind == "rank":
            rows.append(dict(task=name, kind=kind, metric="clisi", a=a, b=a, value=clisi(za2, y2), n=len(y2)))
            rows.append(dict(task=name, kind=kind, metric="graph_conn", a=a, b=a, value=graph_conn(za2, y2), n=len(y2)))
    return rows


def labelfree_rows(z, T) -> list[dict]:
    """Within-label variance captured: CV kNN regression (k 30) of the top PCs from z, variance-weighted; library R2 the same way; share of z variance inside labels."""
    rows = []
    keys = [k.split("|") for k in T.files if k.startswith("cells|")]
    kf = KFold(5, shuffle=True, random_state=0)
    sc = T["lib_size"]
    tot, within = 0.0, 0.0
    zc = z - z.mean(0)
    for _, s, l in keys:
        cells = T[f"cells|{s}|{l}"]
        Z = z[cells]
        within += ((Z - Z.mean(0)) ** 2).sum()
        tot += len(cells)
        pred = {}
        for tag in ("plain", "libcentred"):
            if f"{tag}|{s}|{l}|scores" not in T.files:
                continue
            P, w = T[f"{tag}|{s}|{l}|scores"], T[f"{tag}|{s}|{l}|var"]
            ph = np.zeros_like(P)
            for tr, te in kf.split(Z):
                ph[te] = KNeighborsRegressor(n_neighbors=min(RU.K, len(tr))).fit(Z[tr], P[tr]).predict(Z[te])
            r2 = np.array([r2_score(P[:, i], ph[:, i]) for i in range(P.shape[1])])
            rows.append(dict(task="labelfree", kind=tag, metric="within_var_r2", a=s, b=l, value=float((r2.clip(min=0) * w).sum() / w.sum()), n=len(cells)))
            rows.append(dict(task="labelfree", kind=tag, metric="pc1_r2", a=s, b=l, value=float(max(r2[0], 0)), n=len(cells)))
        y = sc[cells]
        ph = np.zeros(len(y))
        for tr, te in kf.split(Z):
            ph[te] = KNeighborsRegressor(n_neighbors=min(RU.K, len(tr))).fit(Z[tr], y[tr]).predict(Z[te])
        rows.append(dict(task="labelfree", kind="plain", metric="lib_r2", a=s, b=l, value=float(max(r2_score(y, ph), 0)), n=len(cells)))
    rows.append(dict(task="labelfree", kind="plain", metric="z_within_share", a="all", b="all", value=float(within / (zc ** 2).sum()), n=int(tot)))
    return rows


def global_rows(z, spc, lab) -> list[dict]:
    import sys
    sys.path.insert(0, str(RU.REPO / "training"))
    from compare_vs_scvi import compute_integration_metrics
    m = compute_integration_metrics(z, spc, lab, k=RU.K_MIX, seed=42)["global"]
    return [dict(task="global", kind="global", metric=k, a="all", b="all", value=float(v), n=len(spc)) for k, v in m.items() if np.isscalar(v) and not isinstance(v, str)]


def all_rows(c: str, z: np.ndarray, with_global: bool = True) -> pd.DataFrame:
    T = targets(c)
    spc, lab = T["spc"], T["lab"]
    if z.shape[0] != len(spc):
        raise ValueError(f"{c}: {z.shape} vs {len(spc)} cells")
    rows = []
    for k in T.files:
        if k.startswith(("score:", "rank:")):
            rows += task_rows(z, T[k], spc, k)
    rows += labelfree_rows(z, T)
    if with_global:
        rows += global_rows(z, spc, lab)
    return pd.DataFrame(rows)


def chimera_runs(c: str) -> dict[int, str]:
    r = pd.read_csv(RU.REPO / "results/analysis/H1/runs.tsv", sep="\t")
    r = r[(r.cohort == c) & r.kept.astype(bool) & r.healthy.astype(bool)]
    return {int(s): str(x) for s, x in zip(r.seed, r.run)}


def latents(c: str) -> dict[str, tuple[str, int, Path]]:
    """name -> (method, seed, path). Chimera healthy seeds, then scVI on the Chimera batch key, Y5's scVI, then any tool latent already on disk."""
    stem = Path(RU.H5AD[c]).stem
    out = {}
    for s, run in chimera_runs(c).items():
        out[f"chimera_s{s}"] = ("Chimera", s, RU.RUNS / run / "final_state.npz")
    for f in sorted((RU.OUT / "scvi").glob(f"{c}_s*/z_mean.npy")):
        s = int(f.parent.name.rsplit("_s", 1)[1])
        out[f"scvi_s{s}"] = ("scVI", s, f)
    for f in sorted((RU.REPO / "results/analysis/Y5/scvi").glob(f"{c}_s*/z_mean.npy")):
        s = int(f.parent.name.rsplit("_s", 1)[1])
        m = "scVI" if c.startswith("liv") else "scVI-library"
        out[f"{'scvi' if m == 'scVI' else 'scvilib'}_s{s}"] = (m, s, f)
    bl = RU.RUNS
    for d in sorted((bl / "baselines_scvi").glob(f"{stem}-s*-e100-nl10*")):
        s = int(d.name.split("-s")[1].split("-")[0])
        if not (d / "z_mean.npy").exists() or "bk_sample" in d.name:
            continue
        out.setdefault(f"scvi_s{s}", ("scVI", s, d / "z_mean.npy"))
    for t, name in (("pca", "PCA"), ("harmony", "Harmony"), ("sysvi", "sysVI"), ("scanvi", "scANVI"), ("camex", "CAMEX"), ("saturn", "SATURN")):
        for d in sorted((bl / f"baselines_{t}").glob(f"{stem}*")):
            if not (d / "z_mean.npy").exists():
                continue
            s = int(d.name.split("-s")[1].split("-")[0]) if "-s" in d.name.replace(stem, "") else 0
            key = f"{t}_s{s}" if "bk_" not in d.name else f"{t}_s{s}"
            out.setdefault(key, (name, s, d / "z_mean.npy"))
    int_dir = RU.REPO / "results/analysis/B2/integration/latents" / c
    for t, name in (("PCA", "PCA"), ("Harmony", "Harmony"), ("sysVI", "sysVI"), ("scANVI", "scANVI"), ("CAMEX", "CAMEX"), ("SATURN", "SATURN")):
        f = int_dir / t / "z_mean.npy"
        if f.exists():
            out.setdefault(f"{t.lower()}_s42", (name, 42, f))
    return out


def load_z(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            return np.asarray(d["z_mean"], float)
    return np.asarray(np.load(path), float)
