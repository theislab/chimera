"""Post-hoc views of a trained latent, no retraining. resid: z minus its linear fit on log library (pooled over species, coefficients from the training cells). whiten: PCA whitening of z.
resid_whiten: both. Same metrics as measure; held-out TMS and TS hepatocytes go through the encoder, then the training-cell transform, the library being the cell's own on the model's genes.
python -m analysis.tasks.Z2_embedding.posthoc <cohort> [--encode]   Output: results/analysis/Z2/posthoc/<cohort>__<latent>__<view>.parquet, encoded rows posthoc_tms.csv"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression  # noqa: F401

from analysis.tasks.Z2_embedding import common as K, encode as E, rules as RU

OUT = RU.OUT / "posthoc"


class View:
    def __init__(self, z, lib, name):
        self.name = name
        Y = np.c_[np.ones(len(lib)), lib]
        self.b = np.linalg.lstsq(Y, z, rcond=None)[0]
        base = z - Y @ self.b if "resid" in name else z
        self.mu = base.mean(0)
        u, s, vt = np.linalg.svd(base - self.mu, full_matrices=False)
        self.vt, self.s = vt, s / np.sqrt(len(z) - 1)

    def __call__(self, z, lib):
        x = z - np.c_[np.ones(len(lib)), lib] @ self.b if "resid" in self.name else z
        return ((x - self.mu) @ self.vt.T) / self.s.clip(min=1e-8) if "whiten" in self.name else x


def main():
    c = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    T = K.targets(c)
    lib = T["lib_size"]
    want = [(n, v) for n, v in K.latents(c).items() if v[0] in ("Chimera", "scVI")]
    for name, (m, s, p) in want:
        z = K.load_z(p)
        if z.shape[0] != len(lib):
            continue
        for vw in ("resid", "whiten", "resid_whiten"):
            f = OUT / f"{c}__{name}__{vw}.parquet"
            if f.exists():
                continue
            d = K.all_rows(c, View(z, lib, vw)(z, lib))
            d.insert(0, "view", vw); d.insert(0, "seed", s); d.insert(0, "method", m); d.insert(0, "latent", name); d.insert(0, "cohort", c)
            d.to_parquet(f)
        print(c, name, flush=True)
    if "--encode" in sys.argv and c == "liv_hmouse":
        A1, A2 = sp.load_npz(E.EXT / "ts_hep.npz"), sp.load_npz(E.EXT / "tms_hep.npz")
        fa = pd.read_csv(E.EXT / "tms_hep_obs.csv").free_annotation.to_numpy()
        y = np.where(fa == "Hepatocyte (Pericentral)", 1, np.where(fa == "Hepatocyte (Periportal)", 0, -1))
        lab = y >= 0
        ltms = np.log1p(np.asarray(A2.sum(1)).ravel())
        rows = []
        models = list(K.chimera_runs(c).values()) + [f"scvi:results/analysis/Y5/scvi/liv_hmouse_s{s}" for s in range(42, 47)]
        for mdl in models:
            enc, z, a = (E.scvi_encoder(mdl[5:], c) if mdl.startswith("scvi:") else E.chimera_encoder(mdl, c))
            zm = enc(A2, "Mouse")
            for vw in ("none", "resid", "whiten", "resid_whiten"):
                x = zm if vw == "none" else View(z, lib, vw)(zm, ltms)
                al, ak = K._auroc(x[lab], y[lab], False)
                rows.append(dict(model=mdl, view=vw, tms_auroc_lr=al, tms_auroc_knn=ak))
            print(mdl, flush=True)
            pd.DataFrame(rows).to_csv(RU.OUT / "posthoc_tms.csv", index=False)


if __name__ == "__main__":
    main()
