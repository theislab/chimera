"""T11.6 V2: Chimera per-species residual, by species and class, from the checkpoints.

Per seed in D and per gene g, species s, class k (``cell_ontology_class``): RMS over cells of the
residual head's log-rate (``output_residual`` on the mean-z decoder state, the species and batch one-hots),
the same quantity as ``residual_rms`` in ``final_state.npz``, split by species and class. Lower means
the gene in that species needs less correction from the shared program. Output
``results/analysis/T11/residuals/res_<dataset>_s<seed>.npz``; nothing is written to the run directory.

    PYTHONPATH=<chimera repo> python analysis/tasks/T11_6_vz_glaucoma/residuals.py [--threads 12]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import scipy.sparse as sp

from analysis.tasks.T11_5_vz_genes import vz

OUT = vz.OUT / "residuals"


def one(ds: str, seed: int, batch_size: int = 512) -> None:
    import torch
    import torch.nn.functional as F
    from scvi import REGISTRY_KEYS

    from chimera import ChimeraModel

    a = vz.read_h5ad(ds)
    model = ChimeraModel.load(str(vz.RUNS / vz.seed_runs(ds)[seed] / "model"), adata=a, accelerator="cpu")
    m = model.module.eval()
    mgr = model.adata_manager
    sc = np.asarray(mgr.get_from_registry(REGISTRY_KEYS.LABELS_KEY)).ravel().astype(int)
    bc = np.asarray(mgr.get_from_registry(REGISTRY_KEYS.BATCH_KEY)).ravel().astype(int)
    species = [str(x) for x in mgr.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping]
    lab = a.obs[vz.CLASS_KEY].astype(str).to_numpy()
    pairs = sorted({(int(s), l) for s, l in zip(sc, lab)})
    gi = {p: i for i, p in enumerate(pairs)}
    grp = np.array([gi[(int(s), l)] for s, l in zip(sc, lab)])
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    sq = np.zeros((len(pairs), a.n_vars))
    with torch.no_grad():
        for st in range(0, a.n_obs, batch_size):
            idx = np.arange(st, min(st + batch_size, a.n_obs))
            x = torch.tensor(X[idx].toarray(), dtype=torch.float32)
            s_oh = F.one_hot(torch.tensor(sc[idx], dtype=torch.long), int(m.n_species)).float()
            b_oh = F.one_hot(torch.tensor(bc[idx], dtype=torch.long), int(m.n_batch)).float()
            z = m.z_mean_linear(m.encoder(m._encoder_input(x, s_oh, b_oh)))
            res = m.output_residual(torch.cat([m.decoder_hidden(z), s_oh, b_oh], dim=-1)).numpy().astype(np.float64)
            np.add.at(sq, grp[idx], res ** 2)
    n = np.bincount(grp, minlength=len(pairs)).astype(float)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"res_{ds}_s{seed}.tmp.npz"
    np.savez_compressed(tmp, rms=np.sqrt(sq / n[:, None]), n=n, species_idx=np.array([p[0] for p in pairs]),
                        labels=np.array([p[1] for p in pairs], dtype=str), species=np.array(species, dtype=str),
                        var_names=np.array(a.var_names, dtype=str))
    tmp.replace(OUT / f"res_{ds}_s{seed}.npz")
    print(ds, seed, "done", flush=True)


def main() -> int:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=12)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    for ds in vz.usable_datasets():
        for s in vz.seeds_in_D(ds):
            if not (OUT / f"res_{ds}_s{s}.npz").exists():
                one(ds, s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
