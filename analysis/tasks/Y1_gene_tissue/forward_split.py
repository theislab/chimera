"""Addition J: split the model's own margin, S = pattern gain + level cost, on existing checkpoints (T2_5_celltype_margin forward pass, posterior mean z).

mu1 = exp(shared(z)) L, mu0 = p_null[s] sum(mu1), c_gs = sum x / sum mu1 per species and gene, mu_p = c_gs mu1.
Pm = mean over species of mean over cells of LLR(mu_p vs mu0); Lm = the same of LLR(mu1 vs mu_p). Pm + Lm = forward S, exact. Per species and group sums kept.

    python -I launch.py analysis.tasks.Y1_gene_tissue.forward_split <tissue> <seed> [--threads 4]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from analysis.common import cohorts as C
from analysis.tasks.Y1_gene_tissue import common as Y

sys.path.insert(0, str(Y.ROOT / "analysis" / "tasks" / "T2_5_celltype_margin"))
import run as T25RUN  # noqa: E402


def main():
    import scipy.sparse as sp
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("tissue")
    p.add_argument("seed", type=int)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    d = Y.TISSUES[a.tissue]
    run = d["runs"][a.seed]
    ad_ = Y.read(a.tissue)
    run_dir = C.RUNS_DIR / run
    cfg = json.loads((run_dir / "config.json").read_text())
    comp = T25RUN.load_components(run_dir, ad_, cfg)
    X = ad_.X.tocsr() if sp.issparse(ad_.X) else sp.csr_matrix(ad_.X)
    N, G = X.shape
    sidx, bidx = comp["species_codes"], comp["batch_codes"]
    sp_names = comp["species_names"]
    S = len(sp_names)
    bs = 512

    def batches():
        with torch.no_grad():
            for st in range(0, N, bs):
                idx = np.arange(st, min(st + bs, N))
                x = torch.tensor(X[idx].toarray(), dtype=torch.float32)
                s_t, b_t = torch.tensor(sidx[idx], dtype=torch.long), torch.tensor(bidx[idx], dtype=torch.long)
                L = x.sum(1, keepdim=True).double()
                zmu, _ = comp["encode"](x, s_t, b_t)
                mu1 = torch.exp(comp["shared_rate"](zmu).double()) * L
                yield idx, x.double(), s_t, b_t, mu1

    sx, sm = np.zeros((S, G)), np.zeros((S, G))
    for idx, x, s_t, b_t, mu1 in batches():
        np.add.at(sx, sidx[idx], x.numpy())
        np.add.at(sm, sidx[idx], mu1.numpy())
    cgs = torch.tensor((sx + 0.5) / (sm + 0.5))
    lab = ad_.obs["y1_group"].astype(str).to_numpy()
    pairs = sorted({(int(s), l) for s, l in zip(sidx, lab)})
    gi = {k: i for i, k in enumerate(pairs)}
    cg = np.array([gi[(int(s), l)] for s, l in zip(sidx, lab)])
    Pg, Lg = np.zeros((len(pairs), G)), np.zeros((len(pairs), G))
    for idx, x, s_t, b_t, mu1 in batches():
        Lt = mu1.sum(1, keepdim=True).clamp(min=1.0)
        mu0 = (comp["p_null"][s_t] * Lt).clamp(min=1e-8)
        mup = cgs[s_t] * mu1
        th = comp["theta"][b_t]
        np.add.at(Pg, cg[idx], T25RUN.nb_llr(x, mup, mu0, th).numpy())
        np.add.at(Lg, cg[idx], T25RUN.nb_llr(x, mu1, mup, th).numpy())
    n = np.bincount(cg, minlength=len(pairs))
    out = dict(var_names=np.array(ad_.var_names, dtype=str), species=np.array(sp_names, dtype=str), n_species=np.bincount(sidx, minlength=S), run=np.array(run), Pg=Pg, Lg=Lg, n=n,
               g_species=np.array([k[0] for k in pairs]), g_label=np.array([k[1] for k in pairs], dtype=str), c_gs=cgs.numpy())
    Y.FW.mkdir(parents=True, exist_ok=True)
    tmp = Y.FW / f"split_{a.tissue}_s{a.seed}.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.replace(Y.FW / f"split_{a.tissue}_s{a.seed}.npz")
    print("done", a.tissue, a.seed, flush=True)


if __name__ == "__main__":
    main()
