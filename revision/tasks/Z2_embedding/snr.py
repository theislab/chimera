"""Where each latent keeps depth, species and zone, and how noisy the posterior is along each axis. On z_mean PCs (training cells of a cohort):
pc_share; posterior variance along the PC over z_mean variance along it (noise ratio, from each cell's diagonal posterior); CV AUROC of species from z (standardised LR, all cells and inside the largest label);
rate = mean KL(q(z|x) || p) per cell for the fitted prior: for Chimera the VAMP estimate on one sample, as the loss; for scVI the N(0, I) closed form, compared with n_latent x free bits.
python -m revision.tasks.Z2_embedding.snr   Output: results/revision/Z2/snr.csv, snr_pcs.csv"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revision.tasks.Z2_embedding import common as K, rules as RU


TAG = "_".join(sys.argv[1:]) or "default"


def species_auc(z, y):
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), z, y, cv=5, method="predict_proba")
    return roc_auc_score(y, p, multi_class="ovr") if p.shape[1] > 2 else roc_auc_score(y, p[:, 1])


def stats(c, name, method, seed, zm, lv, rate):
    T = K.targets(c)
    spc, lab, lib = T["spc"], T["lab"], T["lib_size"]
    codes = pd.factorize(spc, sort=True)[0]
    big = pd.Series(lab).value_counts().index[0]
    q = lab == big
    zc = zm - zm.mean(0)
    u, s, vt = np.linalg.svd(zc, full_matrices=False)
    var = s ** 2 / len(zm)
    # diagonal posterior projected on PC k: sum_d v_kd^2 sigma_d^2
    post = (np.exp(lv)[:, None, :] * (vt ** 2)[None, :, :]).sum(-1).mean(0)
    pcs = pd.DataFrame(dict(cohort=c, latent=name, method=method, seed=seed, pc=np.arange(len(var)) + 1, share=var / var.sum(), noise_ratio=post / var,
                            r_loglib=[abs(np.corrcoef(u[:, k], lib)[0, 1]) for k in range(len(var))],
                            species_auc=[roc_auc_score(codes == 0, u[:, k]) for k in range(len(var))]))
    pcs["species_auc"] = (pcs.species_auc - 0.5).abs() + 0.5
    r = dict(cohort=c, latent=name, method=method, seed=seed, rate=rate, median_noise_ratio=float(np.median(post / var)), noise_ratio_all=float(post.sum() / var.sum()),
             sp_auc_all=species_auc(zm, codes), sp_auc_biglabel=species_auc(zm[q], codes[q]) if len(np.unique(codes[q])) > 1 else np.nan)
    return r, pcs


def chimera_post(c, run):
    sys.path.insert(0, str(RU.REPO / "revision/tasks/T2_5_celltype_margin"))
    from chimera import ChimeraModel
    a = K.read(c)
    try:
        m = ChimeraModel.load(str(RU.RUNS / run / "model"), adata=a, accelerator="cpu")
    except Exception as exc:
        print("load failed", run, type(exc).__name__, flush=True)
        return None
    mod = m.module.eval()
    X = a.X
    zs, ls, kl = [], [], []
    sp0, ba0 = torch.zeros(1), None
    for st in range(0, X.shape[0], 512):
        x = torch.tensor(X[st:st + 512].toarray(), dtype=torch.float32)
        e = mod._encoder_input(x, torch.zeros(x.shape[0], mod.n_species), torch.zeros(x.shape[0], mod.n_batch))
        with torch.no_grad():
            h = mod.encoder(e)
            mu, lv = mod.z_mean_linear(h), mod.z_logvar_linear(h)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            kl.append(mod._vamp_kl(z, mu, lv).numpy())
        zs.append(mu.numpy()); ls.append(lv.numpy())
    return np.vstack(zs), np.vstack(ls), float(np.concatenate(kl).mean())


def scvi_post(c, path):
    import scvi
    a = K.read(c)
    m = scvi.model.SCVI.load(path, adata=a)
    q = m.get_latent_representation(return_dist=True)
    mu, var = q
    lv = np.log(var)
    return mu, lv, float((0.5 * (mu ** 2 + var - lv - 1)).sum(1).mean())


def main():
    rows, pcs = [], []
    for c in sys.argv[1:] or ("liv_hmouse", "test_hmouse"):
        for s, run in list(K.chimera_runs(c).items())[:10]:
            post = chimera_post(c, run)
            if post is None:
                continue
            r, p = stats(c, run, "Chimera", s, *post)
            rows.append(r); pcs.append(p); print(c, run, flush=True)
            pd.DataFrame(rows).to_csv(RU.OUT / f"snr_{TAG}.csv", index=False)
        for f in sorted((RU.REPO / "results/revision/Y5/scvi").glob(f"{c}_s*/model.pt")) + sorted((RU.OUT / "scvi").glob(f"{c}_s*/model.pt")):
            d = f.parent
            mu, lv, kl = scvi_post(c, str(d))
            r, p = stats(c, d.name, "scVI", int(d.name.rsplit("_s", 1)[1]), mu, lv, kl)
            rows.append(r); pcs.append(p); print(c, d.name, flush=True)
            pd.DataFrame(rows).to_csv(RU.OUT / f"snr_{TAG}.csv", index=False)
    pd.concat(pcs).to_csv(RU.OUT / f"snr_pcs_{TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    t = pd.DataFrame(rows)
    print(t.groupby(["cohort", "method"])[["rate", "noise_ratio_all", "median_noise_ratio", "sp_auc_all", "sp_auc_biglabel"]].agg(["mean", "min", "max"]).round(3).to_string())
    print(pd.concat(pcs).groupby(["cohort", "method", "pc"])[["share", "noise_ratio", "r_loglib", "species_auc"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
