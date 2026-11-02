"""What the encoder reads. For hepatocytes of liver H+Mouse and human testis: raw-count variance concentration (top genes' share of summed variance, raw and log1p), and encoder sensitivity
(rms dz/dx_g x SD(x_g) over 500 cells, summed over z dims) by expression rank, Chimera healthy seeds against scVI, where scVI's derivative is taken with respect to log1p counts, its real input.
python -m revision.tasks.Z2_embedding.diag_encoder  Output: results/revision/Z2/diag_encoder.csv"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch

from revision.tasks.Z2_embedding import common as K, rules as RU


def conc(M):
    v = np.asarray(M.multiply(M).mean(0)).ravel() - np.asarray(M.mean(0)).ravel() ** 2
    v = np.sort(v)[::-1]
    return {f"top{n}": float(v[:n].sum() / v.sum()) for n in (1, 5, 10, 50, 100)}


def sens_share(J_sd, order):
    """share of total squared sensitivity by expression rank"""
    t = (J_sd ** 2).sum()
    s = (J_sd[order] ** 2)
    n = len(order)
    return {f"sens_top{k}": float(s[:k].sum() / t) for k in (10, 50, 100, n // 10)} | {"sens_bottom50pct": float(s[n // 2:].sum() / t)}


def grads(fun, x):
    x = x.clone().requires_grad_(True)
    z = fun(x)
    tot = torch.zeros(x.shape[1])
    for k in range(z.shape[1]):
        g, = torch.autograd.grad(z[:, k].sum(), x, retain_graph=True)
        tot += g.pow(2).mean(0)
    return tot.sqrt().numpy()


def main():
    rows = []
    sys.path.insert(0, str(RU.REPO / "revision/tasks/T2_5_celltype_margin"))
    import run as T25
    import scvi
    for c, cell_type, sp_name in (("liv_hmouse", "Hepatocytes", "Human"), ("liv_hmouse", "Hepatocytes", "Mouse"), ("test_hmouse", "Pachytene_Spermatocytes", "Human")):
        a = K.read(c)
        q = (a.obs.cell_type.astype(str).to_numpy() == cell_type) & (a.obs.species.astype(str).to_numpy() == sp_name)
        M = a.X[q]
        rows.append(dict(cohort=c, cells=cell_type, species=sp_name, model="input_raw", **conc(M)))
        L = M.copy()
        L.data = np.log1p(L.data)
        rows.append(dict(cohort=c, cells=cell_type, species=sp_name, model="input_log1p", **conc(L)))
        rows.append(dict(cohort=c, cells=cell_type, species=sp_name, model="lib", median=float(np.median(np.asarray(M.sum(1)))), cv=float(np.asarray(M.sum(1)).std() / np.asarray(M.sum(1)).mean())))
        mu = np.asarray(M.mean(0)).ravel()
        order = np.argsort(-mu)
        idx = np.random.default_rng(0).choice(M.shape[0], min(500, M.shape[0]), replace=False)
        X = torch.tensor(M[idx].toarray(), dtype=torch.float32)
        sd_raw = M[idx].toarray().std(0)
        sd_log = np.log1p(M[idx].toarray()).std(0)
        g = K.symbols(c, a.var_names).symbol.astype(str).to_numpy()
        zone = np.isin(g, RU.HEP_ZONE["up"] + RU.HEP_ZONE["dn"])
        for s, run in list(K.chimera_runs(c).items())[:5]:
            comp = T25.load_components(RU.RUNS / run, a, json.loads((RU.RUNS / run / "config.json").read_text()))
            si = torch.zeros(X.shape[0], dtype=torch.long)
            J = grads(lambda x: comp["encode"](x, si, si)[0], X) * sd_raw
            r = sens_share(J, order)
            r["zone_share"] = float((J[zone] ** 2).sum() / (J ** 2).sum())
            rows.append(dict(cohort=c, cells=cell_type, species=sp_name, model=f"chimera_s{s}", **r))
        dirs = sorted((RU.REPO / "results/revision/Y5/scvi").glob(f"{c}_s*"))
        for d in dirs[:3]:
            if not (d / "model.pt").exists():
                continue
            m = scvi.model.SCVI.load(str(d), adata=a)
            m.module.eval()
            enc = m.module.z_encoder

            def fun(lx):
                b = torch.zeros(lx.shape[0], 1)
                return enc(lx, b)[0].loc if False else enc(lx)[0].loc
            J = grads(fun, torch.log1p(X)) * sd_log
            r = sens_share(J, order)
            r["zone_share"] = float((J[zone] ** 2).sum() / (J ** 2).sum())
            rows.append(dict(cohort=c, cells=cell_type, species=sp_name, model=f"scvi_{d.name}", **r))
        pd.DataFrame(rows).to_csv(RU.OUT / "diag_encoder.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
