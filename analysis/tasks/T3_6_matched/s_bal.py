"""T3.6 offline: composition-balanced fit-quality margin S_bal from the canonical checkpoint.

No training. A checkpoint-only forward pass (posterior-mean z; components loaded as in
T2.5) gives, per cell, the log-likelihood ratio of the shared decoder against two nulls:

* the species-wide null ``p_null`` the model uses (gives S, the margin as trained), and
* a composition-balanced null over the shared cell types K* (present in every species):
  ``pbal[s] = sum_k wbar_k * (sum_{c in s,k} x_c + 1) / (sum_{c in s,k} lib_c + G)`` with
  ``wbar_k`` the species-mean proportion of type k, renormalised over K*.

``S = mean_s mean_{c in s} l_c``; ``S_bal = (1/|species|) sum_s sum_{k in K*} wbar_k mean_{c in s,k} lb_c``.
``S_shared`` is the part of S contributed by shared types. Reported (A2 section 2.3):
rho(S_bal, S), sign flips in the top half of genes by |S|, the share of Var(S) from shared
types, and Spearman of the ridge projection of S_bal and of S onto the GIN features with
the correction c. Reference values (A2, 4-sp): 0.89, 7.5%, 81%, 0.84 vs 0.81.

Usage::

    python analysis/tasks/T3_6_matched/s_bal.py [--cohorts liv_4sp] [--threads 8] [--outdir DIR] [--force]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args

T2_5 = Path(__file__).resolve().parents[1] / "T2_5_celltype_margin" / "run.py"
UREA = ["CPS1", "OTC", "ASS1", "ASL", "ARG1", "NAGS"]
OXPHOS_REGEX = r"^(NDUF[ABSCV]\d|SDH[ABCD]$|UQCR|COX[4-8]|ATP5)"
REFERENCE = {"liv_4sp": dict(spearman_Sbal_S=0.89, signflip_top_half=0.075, var_share_shared=0.81,
                             rho_projSbal_c=0.84, rho_projS_c=0.81)}


def _t25():
    spec = importlib.util.spec_from_file_location("t2_5_run", T2_5)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def forward(cohort: str, threads: int, batch_size: int = 512) -> dict:
    import scipy.sparse as sp
    import torch

    t25 = _t25()
    torch.set_num_threads(threads)
    t0 = time.time()
    run = C.CANONICAL[cohort]
    run_dir = C.RUNS_DIR / run
    cfg = json.loads((run_dir / "config.json").read_text())
    adata = C.read_adata(cohort)
    comp = t25.load_components(run_dir, adata, cfg)
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    N, G = X.shape
    s_idx = np.asarray(comp["species_codes"])
    b_idx = np.asarray(comp["batch_codes"])
    species = comp["species_names"]
    K = len(species)
    ct = adata.obs["cell_type"].astype(str).to_numpy()
    sp_lab = adata.obs["species"].astype(str).to_numpy()
    if not all(species[s] == l for s, l in zip(s_idx[:1000], sp_lab[:1000])):
        raise AssertionError("species codes do not match labels")
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64)
    n_s = np.bincount(s_idx, minlength=K).astype(float)
    shared = sorted(set.intersection(*[set(ct[s_idx == s]) for s in range(K)]))
    prop = np.array([[((s_idx == s) & (ct == k)).sum() / n_s[s] for k in shared] for s in range(K)])
    wbar = prop.mean(0)
    wbar = wbar / wbar.sum()
    pbal = np.zeros((K, G))
    for s in range(K):
        for j, k in enumerate(shared):
            m = (s_idx == s) & (ct == k)
            pbal[s] += wbar[j] * (np.asarray(X[m].sum(0)).ravel() + 1.0) / (lib[m].sum() + G)
    pbal_t = torch.tensor(pbal)
    pairs = sorted({(int(s), k) for s, k in zip(s_idx, ct)})
    gi = {p: i for i, p in enumerate(pairs)}
    grp = np.array([gi[(int(s), k)] for s, k in zip(s_idx, ct)])
    sum_l = np.zeros((len(pairs), G))
    sum_lb = np.zeros((len(pairs), G))
    with torch.no_grad():
        for st in range(0, N, batch_size):
            idx = np.arange(st, min(st + batch_size, N))
            x = torch.tensor(X[idx].toarray(), dtype=torch.float32)
            s_t = torch.tensor(s_idx[idx], dtype=torch.long)
            b_t = torch.tensor(b_idx[idx], dtype=torch.long)
            L = x.sum(1, keepdim=True).double()
            zmu, _ = comp["encode"](x, s_t, b_t)
            mu1 = torch.exp(comp["shared_rate"](zmu).double()) * L
            Lt = mu1.sum(1, keepdim=True).clamp(min=1.0)
            th = comp["theta"][b_t]
            xd = x.double()
            l0 = t25.nb_llr(xd, mu1, (comp["p_null"][s_t] * Lt).clamp(min=1e-8), th).numpy()
            lb = t25.nb_llr(xd, mu1, (pbal_t[s_t] * Lt).clamp(min=1e-8), th).numpy()
            np.add.at(sum_l, grp[idx], l0)
            np.add.at(sum_lb, grp[idx], lb)
    n = np.bincount(grp, minlength=len(pairs)).astype(float)
    g_sp = np.array([p[0] for p in pairs])
    g_ct = np.array([p[1] for p in pairs])
    is_sh = np.isin(g_ct, shared)
    w = n / n_s[g_sp] / K
    mean_l = sum_l / n[:, None]
    S = (mean_l * w[:, None]).sum(0)
    S_shared = (mean_l[is_sh] * w[is_sh, None]).sum(0)
    S_bal = np.zeros(G)
    for j, k in enumerate(shared):
        for s in range(K):
            m = np.where((g_ct == k) & (g_sp == s))[0][0]
            S_bal += wbar[j] * (sum_lb[m] / n[m]) / K
    print(f"  {cohort}: {comp['source']}, {N} cells, {G} genes, shared {shared}, {time.time() - t0:.0f}s",
          flush=True)
    return dict(run=run, S=S, S_bal=S_bal, S_shared=S_shared, S_ema=comp["S_ema"], shared=shared,
                wbar=dict(zip(shared, wbar.round(4).tolist())), source=comp["source"],
                var_names=np.asarray(adata.var_names), F=np.asarray(adata.varm["gene_features"], dtype=float))


def ridge_proj(F: np.ndarray):
    X = np.c_[np.ones(len(F)), F]
    Inv = np.linalg.inv(X.T @ X + np.eye(X.shape[1]))
    return lambda y: X @ (Inv @ (X.T @ y))


def summarise(cohort: str, fw: dict) -> tuple[dict, pd.DataFrame]:
    st = C.load_state(fw["run"])
    c = st["correction"].astype(float)
    D = (st["alpha"] - st["prior"]).astype(float)
    S, Sb, Ssh = fw["S"], fw["S_bal"], fw["S_shared"]
    proj = ridge_proj(fw["F"])
    top = np.abs(S) > np.percentile(np.abs(S), 50)
    sym = gene_ids.symbols(cohort, fw["var_names"]).astype(str)
    row = dict(cohort=cohort, run=fw["run"], forward_source=fw["source"], shared_types=";".join(fw["shared"]),
               wbar=json.dumps(fw["wbar"]),
               pearson_S_Sema=float(pearsonr(S, fw["S_ema"])[0]),
               spearman_Sbal_S=float(spearmanr(Sb, S)[0]), pearson_Sbal_S=float(pearsonr(Sb, S)[0]),
               signflip_top_half=float((np.sign(Sb[top]) != np.sign(S[top])).mean()),
               var_share_shared=float(np.cov(Ssh, S)[0, 1] / np.var(S, ddof=1)),
               rho_projSbal_c=float(spearmanr(c, proj(Sb))[0]), rho_projS_c=float(spearmanr(c, proj(S))[0]),
               rho_projSema_c=float(spearmanr(c, proj(fw["S_ema"]))[0]),
               spearman_Sbal_D=float(spearmanr(Sb, D)[0]), spearman_S_D=float(spearmanr(S, D)[0]))
    sets = {"urea": sym.isin(UREA).to_numpy(), "mtDNA": sym.str.startswith("MT-").to_numpy(),
            "oxphos_nuclear_regex": sym.str.contains(OXPHOS_REGEX, regex=True).to_numpy()}
    for k, m in sets.items():
        row[f"{k}_n"] = int(m.sum())
        row[f"{k}_S_mean"] = float(S[m].mean()) if m.any() else np.nan
        row[f"{k}_Sbal_mean"] = float(Sb[m].mean()) if m.any() else np.nan
    for k, v in REFERENCE.get(cohort, {}).items():
        row[f"ref_{k}"] = v
    genes = pd.DataFrame(dict(var_name=fw["var_names"], symbol=sym.values, S=S, S_bal=Sb, S_shared=Ssh,
                              S_g_ema=fw["S_ema"], correction=c, delta_alpha=D))
    return row, genes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohorts", nargs="+", default=["liv_4sp"])
    ap.add_argument("--threads", type=int, default=8)
    add_common_args(ap)
    args = ap.parse_args()
    outdir = args.outdir or str(gene_ids.REPO_ROOT / "results" / "analysis" / "T3_6" / "s_bal")
    out = TaskOutput("T3_6_s_bal", outdir, prerun=args.prerun, force=args.force)
    rows = []
    for cohort in args.cohorts:
        row, genes = summarise(cohort, forward(cohort, args.threads))
        rows.append(row)
        out.csv(genes, f"{cohort}__s_bal_genes.csv")
    summ = pd.DataFrame(rows)
    out.csv(summ, "s_bal_summary.csv")
    print(summ.set_index("cohort").T.to_string())
    out.finish(dict(cohorts=args.cohorts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
