"""Test L: S_fwd against a label-free direct margin (rules.py). Reads no gold in `latent`.
    python -m analysis.tasks.D1_sparse_genes.labelfree grid             write lf_grid.tsv
    python -m analysis.tasks.D1_sparse_genes.labelfree latent --index I  one latent, clusters, Sdir on them
    python -m analysis.tasks.D1_sparse_genes.labelfree score            head-to-heads, results/analysis/D1/L_rho.csv, L_calls.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from analysis.tasks.B2_benchmark import perturb as P
from analysis.tasks.D1_sparse_genes import rules as RU

OUT = RU.RES / "L"
GRID = RU.TASK / "lf_grid.tsv"


def grid():
    rows = []
    for c in RU.L_COHORTS:
        for s in RU.L_SEEDS:
            rows.append(dict(cohort=c, setting="full", rep=0, method="scvi", seed=s))
        rows.append(dict(cohort=c, setting="full", rep=0, method="pca", seed=0))
        for r in range(3):
            rows.append(dict(cohort=c, setting="depth25", rep=r, method="scvi", seed=42))
            rows.append(dict(cohort=c, setting="depth25", rep=r, method="pca", seed=0))
    pd.DataFrame(rows).to_csv(GRID, sep="\t", index=False)
    print(len(rows))


def pca_latent(X, spc):
    import anndata as ad
    import scanpy as sc
    from sklearn.decomposition import TruncatedSVD

    a = ad.AnnData(X.astype(np.float32).copy())
    a.obs["species"] = spc
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=RU.L_HVG, batch_key="species", subset=True)
    Y = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float64)
    for s in sorted(set(spc)):
        m = spc == s
        Y[m] -= Y[m].mean(0)
    return TruncatedSVD(RU.L_PCS, random_state=0).fit_transform(Y)


def latent(index: int):
    import anndata as ad

    from analysis.common import gene_ids
    from analysis.tasks.H1_harmonise import rules as H1R
    from analysis.tasks.T4_2_gene_baselines import run as T42
    from analysis.tasks.X6_adds_beyond.degrade import sdir

    RU.register()
    r = pd.read_csv(GRID, sep="\t", keep_default_na=False).iloc[index].to_dict()
    c, setting, rep, method, seed = r["cohort"], r["setting"], int(r["rep"]), r["method"], int(r["seed"])
    out = OUT / f"{c}__{setting}__r{rep}__{method}_s{seed}.parquet"
    if out.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    h5 = RU.H5AD_DIR / f"lf-{c}-{setting}-r{rep}.h5ad"
    if setting == "full":
        path = RU.REPO / H1R.COHORTS[c]["h5ad"]
    else:
        path = h5
        if not h5.exists():
            tmp = h5.with_suffix(f".{os.getpid()}.tmp.h5ad")
            P.build(dict(cohort=c, setting=setting, rep=rep), out_path=tmp, check=False)
            tmp.replace(h5)
    a = ad.read_h5ad(path)
    X = P.csr(a.X).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    if method == "scvi":
        sys.path.insert(0, str(RU.REPO / "training"))
        from compare_vs_scvi import get_or_train_scvi

        z = get_or_train_scvi(path, seed, 100, 10, 128, batch_key="species", accelerator="cpu")
    else:
        z = pca_latent(X, spc)
    cl = T42.leiden_clusters(np.asarray(z, dtype=np.float32))
    s = sdir(X, spc, cl)
    ids = np.asarray(gene_ids.ensembl(H1R.COHORTS[c]["table"], a.var_names))
    ct = pd.crosstab(cl, spc)
    both = ct.loc[(ct > 0).all(1)].sum().sum() / ct.sum().sum()
    lab = a.obs["b2_label"].astype(str) if "b2_label" in a.obs else P.base_labels(a, c)
    from sklearn.metrics import adjusted_rand_score

    pd.DataFrame(dict(hsapiens_gene_id=ids, Sdir_c=s)).to_parquet(out)
    (OUT / f"{out.stem}.json").write_text(json.dumps(dict(cohort=c, setting=setting, rep=rep, method=method, seed=seed, n_clusters=int(ct.shape[0]), share_cells_in_clusters_with_all_species=float(both),
                                                         ari_labels=float(adjusted_rand_score(np.asarray(lab), cl)), n_cells=int(a.n_obs)), indent=1))
    print(out, flush=True)


def lf(c, setting, method, reps=None):
    fr = []
    for f in sorted(OUT.glob(f"{c}__{setting}__r*__{method}_s*.parquet")):
        rep = int(f.name.split("__")[2][1:])
        if reps is None or rep in reps:
            d = pd.read_parquet(f).dropna(subset=["hsapiens_gene_id"])
            fr.append(d[~d.hsapiens_gene_id.duplicated(keep=False)].set_index("hsapiens_gene_id")["Sdir_c"])
    return pd.concat(fr, axis=1).mean(1) if fr else None


def score():
    from joblib import Parallel, delayed

    from analysis.tasks.D1_sparse_genes import b_depth as BD
    from analysis.tasks.D1_sparse_genes import common as CM

    def unit(c, kind, gold, covs, setn, spl):
        lab = dict(cohort=c, kind=kind, set=setn, split=spl)
        if setn == "masked":
            t = CM.load(c)
            rows = (t.masked & t.clean).to_numpy()
            sc = ["S_fwd", "Sdir_scvi", "Sdir_pca", "Sdir", "Lfc"]
            sv, sp = lf(c, "full", "scvi"), lf(c, "full", "pca")
            t["Sdir_scvi"] = sv.reindex(t.hsapiens_gene_id).to_numpy()
            t["Sdir_pca"] = sp.reindex(t.hsapiens_gene_id).to_numpy()
        else:
            t, used = BD.frame(c, "depth25")
            if t is None:
                return [dict(lab, gold=gold, note="Chimera did not finish")]
            reps = sorted(int(u["run"].rsplit("-r", 1)[1]) for u in used if not u["discard"])
            if not reps:
                return [dict(lab, gold=gold, note="Chimera did not finish")]
            rows = t["sparse_f"].to_numpy()
            sc = ["S_fwd_f", "Sdir_scvi", "Sdir_pca", "Sdir_f", "Lfc_f"]
            t["Sdir_scvi"] = lf(c, "depth25", "scvi", reps).reindex(t.hsapiens_gene_id).to_numpy()
            t["Sdir_pca"] = lf(c, "depth25", "pca", reps).reindex(t.hsapiens_gene_id).to_numpy()
        rw = rows & CM.split_rows(t, spl) & t[gold].notna().to_numpy()
        if rw.sum() < CM.MIN_N:
            return [dict(lab, gold=gold, n=int(rw.sum()), note="too few genes")]
        return [dict(x, method=x["score"].replace("_f", "") if x["score"].endswith("_f") else x["score"]) for x in CM.rho_rows(t, gold, covs, sc, rw, sc[0], RU.N_DRAWS, lab)]

    u = []
    for c in RU.L_COHORTS:
        for kind, gold, covs in CM.golds(c):
            if kind in RU.L_GOLDS[c]:
                for setn in ("masked", "sparse_f"):
                    for spl in ("odd", "even", "all"):
                        u.append((c, kind, gold, covs, setn, spl))
    res = Parallel(n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), backend="loky")(delayed(unit)(*x) for x in u)
    d = pd.DataFrame([r for x in res for r in x])
    d.to_csv(RU.RES / "L_rho.csv", index=False)
    calls = []
    for (c, kind, setn), g in d[d.split.isin(["odd", "even"])].groupby(["cohort", "kind", "set"]):
        r = dict(cohort=c, kind=kind, set=setn, primary=kind == RU.L_GOLDS[c][0])
        lead, trail = True, True
        for spl in ("odd", "even"):
            x = g[(g.split == spl) & g.score.isin(["Sdir_scvi", "Sdir_pca"])]
            if len(x) < 2:
                lead = trail = False
                r["note"] = "not scored"
                continue
            b = x.loc[x.rho_p.idxmax()]
            r[f"Sfwd_minus_best_{spl}"] = -b.delta
            r[f"best_{spl}"] = b.score
            r[f"lo_{spl}"] = -b.delta_hi
            r[f"hi_{spl}"] = -b.delta_lo
            lead &= (-b.delta >= RU.SIG) and (-b.delta_hi > 0)
            trail &= (b.delta >= RU.SIG) and (b.delta_lo > 0)
        r["call"] = "leads" if lead else "trails" if trail else "ties"
        calls.append(r)
    pd.DataFrame(calls).to_csv(RU.RES / "L_calls.csv", index=False)
    print(pd.DataFrame(calls).round(3).to_string(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--index", type=int)
    a = ap.parse_args()
    if a.cmd == "grid":
        grid()
    elif a.cmd == "latent":
        latent(a.index)
    else:
        score()
