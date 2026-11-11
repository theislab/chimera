"""Cell-type and species decomposition of S_g on the bone-marrow seeds (inputs to tests E3 and E4).

Forward pass (``--forward <seed>``): the checkpoint-only forward pass of analysis/tasks/T2_5_celltype_margin
(``T2_5_celltype_margin/run.py::forward_sums``, code unchanged) on one seed run of
``T12_2b_recipe/seed_map.tsv``, with build 2 as data and the nine harmonized types
(``cell_ontology_class``) as the grouping. Sums go to ``results/analysis/P12/drivers/fw_s<seed>.npz``;
nothing is written to the run directory.

Combine (``--combine``), per gene g, species s, type k (``margin.decompose`` of T2_5_celltype_margin,
posterior-mean z), over the seeds admitted to D (``seed_health.json``: G-V passing seeds):

* species-row contribution R_gsk = w_sk S_gsk (w_sk = n_sk / (n_s K)); C_gk = sum_s R_gsk, so that
  sum_k C_gk = S_g; five-seed means;
* E3 readout: the divergent contribution of type k is max(-C_gk, 0); the gene's top divergent type is
  its argmax (no type when every C_gk >= 0); mouse rows only (max(-R_g,mouse,k, 0)) beside it;
* E4 readout: divergent mass of species s = sum_k max(-R_gsk, 0); share of s = mass_s / sum over the
  three species;
* B1 leave-one-type-out (E3 baseline): increase_k = E_{-k} - E_all (pooled E of T4_2_gene_baselines on the
  same units), positive when type k carries cross-species disagreement; the B1 top type is its argmax (no
  type when every increase <= 0).

    PYTHONPATH=<chimera repo> python analysis/tasks/T12_3_tests/drivers.py --forward 42 [--threads 16]
    PYTHONPATH=<chimera repo> python analysis/tasks/T12_3_tests/drivers.py --combine
"""

from __future__ import annotations

import argparse
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42

ROOT = gene_ids.REPO_ROOT
T25 = ROOT / "analysis" / "tasks" / "T2_5_celltype_margin"
sys.path.insert(0, str(T25))
import margin as MG  # noqa: E402
import run as T25RUN  # noqa: E402

COHORT = "bone_marrow"
KEY = "cell_ontology_class"
UNIT_KEY = "batch"
RES = ROOT / "results" / "analysis" / "P12"
OUT = RES / "drivers"
SEED_MAP = ROOT / "analysis" / "tasks" / "T12_2b_recipe" / "seed_map.tsv"


def seed_runs() -> dict[int, str]:
    m = pd.read_csv(SEED_MAP, sep="\t")
    return {int(r["seed"]): str(r["source"]).split()[0].replace("training/runs/", "") for _, r in m.iterrows()}


def seeds_in_D() -> list[int]:
    return [int(s) for s in json.loads((RES / "seed_health.json").read_text())["seeds_enter_D"]]


def _bm_adata(cohort: str, backed: bool = False):
    if cohort != COHORT:
        raise KeyError(cohort)
    return ad.read_h5ad(gene_ids.h5ad_path(COHORT), backed="r" if backed else None)


def forward(seed: int, threads: int, batch_size: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run = seed_runs()[seed]
    C.read_adata = _bm_adata  # the forward pass of T2_5_celltype_margin reads the cohort through this helper
    T25RUN.GROUP_KEYS = {"liver": [KEY], "testis": [KEY]}  # group by the nine harmonized types
    fw = T25RUN.forward_sums(f"bm_s{seed}", COHORT, run, threads, batch_size)
    fw["run"] = np.array(run)
    tmp = OUT / f"fw_s{seed}.tmp.npz"
    np.savez_compressed(tmp, **fw)
    tmp.replace(OUT / f"fw_s{seed}.npz")


def loo_b1(a: ad.AnnData, types: list[str]) -> pd.DataFrame:
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    species = a.obs["species"].astype(str).to_numpy()
    sl = sorted(set(species))
    x, f, _ = T42.pseudobulk(X, species, a.obs[KEY].astype(str).to_numpy(),
                             a.obs[UNIT_KEY].astype(str).to_numpy(), T42.MIN_CELLS_TYPE)
    E_all, _, _ = T42.pooled_E(x, f, sl, a.n_vars)
    out = {"E_all": E_all}
    for k in types:
        xk = {key: v for key, v in x.items() if key[1] != k}
        fk = {key: v for key, v in f.items() if key[1] != k}
        Ek, _, _ = T42.pooled_E(xk, fk, sl, a.n_vars)
        out[f"increase_{k}"] = Ek - E_all
    return pd.DataFrame(out, index=a.var_names)


def combine() -> None:
    a = ad.read_h5ad(gene_ids.h5ad_path(COHORT))
    seeds = seeds_in_D()
    runs = seed_runs()
    decs, checks = {}, {}
    for s in seeds:
        fw = dict(np.load(OUT / f"fw_s{s}.npz", allow_pickle=False))
        if list(fw["var_names"]) != list(a.var_names):
            raise AssertionError(f"s{s}: var_names differ")
        dec = MG.decompose(fw, KEY)
        st = C.load_state(runs[s])
        ppass = C.gene_covariates(a)["ppass"].to_numpy() if not checks else checks["_ppass"]
        checks["_ppass"] = ppass
        tot = sum(dec["C_gk_meanz"].values())
        checks[int(s)] = dict(
            closure_r=float(pearsonr(ppass * dec["S_full_samp"], st["S_g_ema"])[0]),
            identity_max_err=float(np.abs(tot - dec["S_full_meanz"]).max()))
        decs[s] = dec
    checks.pop("_ppass")
    d0 = decs[seeds[0]]
    species = d0["species"]
    types = d0["cell_types"]

    def rows(dec, sp_i):  # (K types, G) species-row contributions
        lab, w, S = dec["labels"], dec["w"], dec["S_gsk_meanz"]
        out = []
        for k in types:
            m = (dec["species_idx"] == sp_i) & (lab == k)
            out.append((S[m] * w[m, None]).sum(0) if m.any() else np.zeros(S.shape[1]))
        return np.stack(out)

    R = {sp: np.mean([rows(d, i) for d in decs.values()], axis=0) for i, sp in enumerate(species)}
    Cg = sum(R.values())
    per_seed = {s: sum(rows(d, i) for i in range(len(species))) for s, d in decs.items()}
    seed_rho = {}
    for s in per_seed:
        others = np.mean([v for t, v in per_seed.items() if t != s], axis=0)
        seed_rho[int(s)] = float(spearmanr(per_seed[s].ravel(), others.ravel())[0])
    genes = pd.DataFrame(index=a.var_names)
    genes["hsapiens_gene_id"] = gene_ids.ensembl(COHORT, a.var_names).values
    genes["symbol"] = gene_ids.symbols(COHORT, a.var_names).values
    for j, k in enumerate(types):
        genes[f"C_{k}"] = Cg[j]
        for sp in species:
            genes[f"R_{sp}_{k}"] = R[sp][j]
    genes["S_total"] = Cg.sum(0)
    div = np.maximum(-Cg, 0)
    genes["top_div_type"] = np.where(div.max(0) > 0, np.array(types)[div.argmax(0)], "")
    mdiv = np.maximum(-R["mouse"], 0)
    genes["top_div_type_mouse_rows"] = np.where(mdiv.max(0) > 0, np.array(types)[mdiv.argmax(0)], "")
    for sp in species:
        genes[f"div_mass_{sp}"] = np.maximum(-R[sp], 0).sum(0)
    tot = sum(genes[f"div_mass_{sp}"] for sp in species)
    for sp in species:
        genes[f"div_share_{sp}"] = np.where(tot > 0, genes[f"div_mass_{sp}"] / np.where(tot > 0, tot, 1), np.nan)
    b = loo_b1(a, types)
    inc = b[[f"increase_{k}" for k in types]].to_numpy()
    for j, k in enumerate(types):
        genes[f"b1_increase_{k}"] = inc[:, j]
    fin = np.isfinite(inc).all(1)
    pos = np.where(fin[:, None], inc, -np.inf)
    genes["b1_top_type"] = np.where(fin & (pos.max(1) > 0), np.array(types)[np.argmax(pos, 1)], "")
    genes["B1_E_all"] = b["E_all"].to_numpy()
    genes.reset_index(names="var_name").to_parquet(OUT / "drivers_genes.parquet")
    meta = dict(seeds={int(s): runs[s] for s in seeds}, species=species, types=types, checks=checks,
                closure_bar=MG.ACCEPT_CLOSURE, seed_loo_spearman_C=seed_rho,
                n_cells_per_type_species={f"{species[s]}|{l}": int(n) for s, l, n in zip(d0["species_idx"], d0["labels"], d0["n"])})
    (OUT / "drivers_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))
    (OUT / "DONE").write_text("ok\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--forward", type=int, default=None)
    p.add_argument("--combine", action="store_true")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args()
    if a.forward is not None:
        forward(a.forward, a.threads, a.batch_size)
    if a.combine:
        combine()
    return 0


if __name__ == "__main__":
    sys.exit(main())
