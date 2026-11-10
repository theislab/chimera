"""X7 (EXPLORATORY): checkpoint-only forward pass for the per-cell residual share and species margins.

Definitions in rules.py. Nothing is trained and nothing is written to run dirs. The model is loaded with
``ChimeraModel.load(..., accelerator="cpu")`` as in T2.5; alpha is checked against final_state.npz, the
shared-only and full reconstructions against ``get_shared_only_expression`` / ``get_expression`` on the first
cells, and the species margin sums of the canonical runs against the T2.5 forward npz.

Per cell: tv (A1), gain (A2), lshare (A3), library size, z mean. Per (species, cell_type) and gene: sums of
r_cg, |r_cg|, the full-vs-shared log-likelihood gain, the composition shift p_full - p_sh, and the T2.5
margin l_cg (mean z). Per species and gene: sum of l_cg.

Usage: python -m analysis.tasks.X7_architecture_outputs.forward --runs liv_hmouse,liv_hmouse_s43 [--threads 5]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.tasks.X7_architecture_outputs import rules as RU

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "X7"
FWD = OUT / "forward"
T25 = gene_ids.REPO_ROOT / "results" / "analysis" / "T2_5" / "forward"


def run_table() -> dict:
    """label -> (cohort, run)."""
    from analysis.tasks.T4_3_scoreboard import inputs as I

    out = {c: (c, r) for c, r in C.CANONICAL.items()}
    for c, seeds in RU.SEEDS.items():
        if c in C.SEED_RUNS:
            names = {int(r.rsplit("-s", 1)[-1]): r for r in C.SEED_RUNS[c]}
        else:
            names = {}
        try:
            names.update({int(k): v for k, v in I.load(c, gene_ids.REPO_ROOT / "results/analysis/T4_3/cache")["meta"]["seeds"].items()})
        except Exception:  # noqa: BLE001 - cohorts without a T4.3 cache use SEED_RUNS only
            pass
        for s in seeds:
            out[f"{c}_s{s}"] = (c, names[s])
    return out


def nb_llr(x, mu1, mu0, th):
    """log NB(x | mu1, th) - log NB(x | mu0, th) (T2.5 formula)."""
    import torch

    e = 1e-8
    return (th * (torch.log(th + mu0 + e) - torch.log(th + mu1 + e))
            + x * (torch.log(mu1 + e) - torch.log(th + mu1 + e) - torch.log(mu0 + e) + torch.log(th + mu0 + e)))


def forward(label: str, cohort: str, run: str, threads: int, batch_size: int = 512) -> dict:
    import scipy.sparse as sp
    import torch
    import torch.nn.functional as F
    from scvi import REGISTRY_KEYS

    from chimera import ChimeraModel

    torch.set_num_threads(threads)
    t0 = time.time()
    run_dir = C.RUNS_DIR / run
    cfg = json.loads((run_dir / "config.json").read_text())
    if not cfg.get("use_observed_lib_size", True):
        raise NotImplementedError("assumes use_observed_lib_size=True")
    adata = C.read_adata(cohort)
    model = ChimeraModel.load(str(run_dir / "model"), adata=adata, accelerator="cpu")
    m = model.module.eval()
    mgr = model.adata_manager
    s_all = np.asarray(mgr.get_from_registry(REGISTRY_KEYS.LABELS_KEY)).ravel().astype(int)
    b_all = np.asarray(mgr.get_from_registry(REGISTRY_KEYS.BATCH_KEY)).ravel().astype(int)
    species = [str(x) for x in mgr.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping]
    if [species[i] for i in s_all[:50]] != list(adata.obs["species"].astype(str)[:50]):
        raise AssertionError("species codes do not match obs")
    n_s, n_b = int(m.n_species), int(m.n_batch)
    fs = C.load_state(run)
    with torch.no_grad():
        if getattr(m, "_use_per_gene_alpha", False):
            alpha = torch.sigmoid(torch.logit(m.conservation_prior) + m.per_gene_correction)
        else:
            alpha, _ = m.gin(m.gene_features, m.conservation_prior)
        bo = m.batch_offset().double()
    alpha_np = alpha.detach().numpy().astype(float)
    if not np.allclose(alpha_np, fs["alpha"], atol=1e-5):
        raise AssertionError(f"{run}: alpha differs from final_state ({np.abs(alpha_np - fs['alpha']).max():.2e})")
    one_m_alpha = (1.0 - alpha.detach()).double()
    theta = F.softplus(m.dispersion.detach()).double()
    p_null = m.p_null.detach().double()

    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    N, G = X.shape
    lab = adata.obs["cell_type"].astype(str).to_numpy()
    pairs = sorted({(int(s), l) for s, l in zip(s_all, lab)})
    gi = {p: i for i, p in enumerate(pairs)}
    cg = np.array([gi[(int(s), l)] for s, l in zip(s_all, lab)])
    nG = len(pairs)
    acc = {k: np.zeros((nG, G)) for k in ("r", "absr", "gain", "dp", "l")}
    sp_l = np.zeros((n_s, G))

    def encode(x, s_t, b_t):
        s_oh = F.one_hot(s_t, n_s).float()
        b_oh = F.one_hot(b_t, n_b).float()
        h = m.encoder(m._encoder_input(x, s_oh, b_oh))
        return m.z_mean_linear(h), s_oh, b_oh

    # pass 1: z and abar
    Z = np.zeros((N, int(m.n_latent)), dtype=np.float32)
    a_sum = torch.zeros(G, dtype=torch.float64)
    with torch.no_grad():
        for st in range(0, N, batch_size):
            idx = np.arange(st, min(st + batch_size, N))
            x = torch.tensor(X[idx].toarray(), dtype=torch.float32)
            s_t = torch.tensor(s_all[idx]); b_t = torch.tensor(b_all[idx])
            zm, _, _ = encode(x, s_t, b_t)
            Z[idx] = zm.numpy()
            a = m.output_shared(m.decoder_hidden(zm)).double() + bo[b_t]
            a_sum += a.sum(0)
    abar = a_sum / N
    tv = np.zeros(N); gain = np.zeros(N); lshare = np.zeros(N); lib = np.zeros(N)
    check = {}
    with torch.no_grad():
        for st in range(0, N, batch_size):
            idx = np.arange(st, min(st + batch_size, N))
            x = torch.tensor(X[idx].toarray(), dtype=torch.float32)
            xd = x.double()
            s_t = torch.tensor(s_all[idx]); b_t = torch.tensor(b_all[idx])
            zm = torch.tensor(Z[idx])
            s_oh = F.one_hot(s_t, n_s).float(); b_oh = F.one_hot(b_t, n_b).float()
            h = m.decoder_hidden(zm)
            sh = m.output_shared(h).double()
            res = m.output_residual(torch.cat([h, s_oh, b_oh], -1)).double()
            a = sh + bo[b_t]
            r = one_m_alpha * res
            L = xd.sum(1, keepdim=True)
            mu_f = torch.exp(a + r) * L
            mu_s = torch.exp(a) * L
            th = theta[b_t]
            llg = nb_llr(xd, mu_f, mu_s, th)
            pf = mu_f / mu_f.sum(1, keepdim=True)
            ps = mu_s / mu_s.sum(1, keepdim=True)
            dpp = pf - ps
            tv[idx] = 0.5 * dpp.abs().sum(1).numpy()
            gain[idx] = (llg.sum(1) / L.squeeze(1).clamp(min=1)).numpy()
            ar = r.abs().sum(1)
            lshare[idx] = (ar / (ar + (a - abar).abs().sum(1))).numpy()
            lib[idx] = L.squeeze(1).numpy()
            # T2.5 margin, mean z (no batch offset; null on the shared total)
            mu1 = torch.exp(sh) * L
            Lt = mu1.sum(1, keepdim=True).clamp(min=1.0)
            mu0 = (p_null[s_t] * Lt).clamp(min=1e-8)
            l = nb_llr(xd, mu1, mu0, th)
            gidx = cg[idx]
            for k, v in (("r", r), ("absr", r.abs()), ("gain", llg), ("dp", dpp), ("l", l)):
                np.add.at(acc[k], gidx, v.numpy())
            np.add.at(sp_l, s_all[idx], l.numpy())
            if st == 0:
                check["mu_sh"] = mu_s[:64].numpy()
                check["mu_f"] = mu_f[:64].numpy()
    # API check on the first 64 cells
    ref_sh = model.get_shared_only_expression(indices=np.arange(64), batch_size=64)
    ref_f = model.get_expression(indices=np.arange(64), batch_size=64)
    err_sh = float(np.max(np.abs(np.log(check["mu_sh"] + 1e-12) - np.log(ref_sh + 1e-12))))
    err_f = float(np.max(np.abs(np.log(check["mu_f"] + 1e-12) - np.log(ref_f + 1e-12))))
    if err_sh > 1e-3 or err_f > 1e-3:
        raise AssertionError(f"{run}: reconstruction differs from the model API (log err {err_sh:.2e}, {err_f:.2e})")
    n_grp = np.bincount(cg, minlength=nG)
    out = dict(var_names=np.array(adata.var_names, dtype=str), species=np.array(species, dtype=str),
               obs_species=s_all, obs_batch=b_all, obs_cell_type=lab.astype(str), obs_names=np.array(adata.obs_names, dtype=str),
               tv=tv, gain=gain, lshare=lshare, lib=lib, z=Z, alpha=alpha_np,
               groups=np.array([(s, l, n) for (s, l), n in zip(pairs, n_grp)],
                               dtype=[("species_idx", "i4"), ("label", "U64"), ("n", "i8")]),
               sp_l=sp_l, n_species=np.bincount(s_all, minlength=n_s), api_log_err=np.array([err_sh, err_f]),
               run=np.array(run), cohort=np.array(cohort))
    for k, v in acc.items():
        out[f"sum_{k}"] = v
    # canonical: T2.5 species sums must match
    f25 = T25 / f"{label}.npz"
    if f25.exists():
        with np.load(f25) as z25:
            ref = z25["sp_sum_meanz"]
        if not np.allclose(ref, sp_l, rtol=1e-4, atol=1e-3):
            raise AssertionError(f"{label}: species margin sums differ from T2.5 ({np.abs(ref - sp_l).max():.3e})")
        out["t25_match"] = np.array(True)
    print(f"  {label}: {N} cells, {G} genes, {time.time() - t0:.0f}s; api err {err_sh:.1e}/{err_f:.1e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    tab = run_table()
    labels = list(tab) if a.runs == "all" else a.runs.split(",")
    FWD.mkdir(parents=True, exist_ok=True)
    for lab in labels:
        f = FWD / f"{lab}.npz"
        if f.exists() and not a.force:
            print("exists", lab, flush=True)
            continue
        cohort, run = tab[lab]
        print("==", lab, run, flush=True)
        o = forward(lab, cohort, run, a.threads)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, **o)
        tmp.replace(f)


if __name__ == "__main__":
    main()
