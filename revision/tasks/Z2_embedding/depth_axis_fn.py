"""Depth share of a latent."""
import numpy as np
from sklearn.linear_model import LinearRegression
from revision.tasks.Z2_embedding import rules as RU


def one(z, T):
    spc, lab, y = T["spc"], T["lab"], T["lib_size"]
    zc = z - z.mean(0)
    sv = np.linalg.svd(zc, compute_uv=False) ** 2
    r = dict(pc1_share=sv[0] / sv.sum(), r2_lib_from_z=LinearRegression().fit(zc, y).score(zc, y))
    keep, tot, expl, wts, r2s = np.zeros(len(y), bool), 0.0, 0.0, [], []
    for s in np.unique(spc):
        for l in np.unique(lab[spc == s]):
            q = (spc == s) & (lab == l)
            if q.sum() < RU.MIN_LABEL_CELLS:
                continue
            Z = z[q] - z[q].mean(0)
            Y = (y[q] - y[q].mean())[:, None]
            b = np.linalg.lstsq(Y, Z, rcond=None)[0]
            tot += (Z ** 2).sum()
            expl += ((Y @ b) ** 2).sum()
            r2s.append(LinearRegression().fit(Z, Y.ravel()).score(Z, Y.ravel()))
            wts.append(q.sum())
    r.update(within_r2_lib_from_z=float(np.average(r2s, weights=wts)), within_z_var_by_lib=expl / tot)
    return r
