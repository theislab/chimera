"""E-K4 of the kidney case: cell-type drivers of the count margin (pre-specified analysis plan, section 8c),
descriptive, Family 3.

Forward pass (``--forward <seed>``): the checkpoint-only forward pass of the margin decomposition
(``T2_5_celltype_margin/run.py::forward_sums``, unchanged) on one kidney seed run, with the kidney
healthy object as data and ``cell_type`` (= ``cell_ontology_class``) as the grouping. Sums go to
``results/analysis/P13/drivers/fw_s<seed>.npz``; nothing is written to the run directory.

Combine (``--combine``), per gene g, species s, cell type k (``margin.decompose`` of T2_5_celltype_margin, posterior mean z):
* mouse-row contribution M_gk = w_sk S_gsk for s = Mouse (w_sk = n_sk / (n_s K)); human-row
  contribution likewise; mean over the five seeds;
* divergent mass of a gene in type k = max(-M_gk, 0) (the shared decoder fits mouse cells of k worse
  than the null); share of type k = divergent mass in k / total divergent mass of g;
* E-K4 (plan, section 8c): for Cippa human-specific genes (Table S3 acute primary, Table S9 late secondary), the
  mean share carried by PT + TAL versus immune (myeloid, lymphoid) and stromal (fib, vsm_p) rows,
  beside the same shares for Cippa shared genes and for all vocabulary genes with a defined share
  (the descriptive expectation); per-type shares also by Cippa list;
* baseline beside it (per-type pseudobulk): the leave-one-type-out change in B1 (E; pseudobulk correlation
  of T4_2_gene_baselines over the shared classes on the same object): increase_k = E_{-k} - E_all, positive
  when type k carries cross-species disagreement; share of type k = max(increase_k, 0) / sum.

Descriptive only: no test, no pass rule.

    PYTHONPATH=<chimera repo> python analysis/tasks/P13_kidney/drivers.py --forward 42 [--threads 16]
    PYTHONPATH=<chimera repo> python analysis/tasks/P13_kidney/drivers.py --combine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42

ROOT = gene_ids.REPO_ROOT
T25 = ROOT / "analysis" / "tasks" / "T2_5_celltype_margin"
sys.path.insert(0, str(T25))
import margin as MG  # noqa: E402
import run as T25RUN  # noqa: E402

RES = ROOT / "results" / "analysis" / "P13"
OUT = RES / "drivers"
SEED_MAP = ROOT / "analysis" / "tasks" / "P13_recipe" / "seed_map.tsv"
EPITHELIAL = ["pt", "tal"]
IMMUNE_STROMAL = ["myeloid", "lymphoid", "fib", "vsm_p"]


def seed_runs() -> dict[int, str]:
    m = pd.read_csv(SEED_MAP, sep="\t")
    return {int(r["seed"]): str(r["source"]).split()[0].replace("training/runs/", "") for _, r in m.iterrows()}


def _kidney_adata(cohort: str, backed: bool = False):
    if cohort != "kidney":
        raise KeyError(cohort)
    return ad.read_h5ad(gene_ids.h5ad_path("kidney"), backed="r" if backed else None)


def forward(seed: int, threads: int, batch_size: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run = seed_runs()[seed]
    C.read_adata = _kidney_adata  # the forward pass of T2_5_celltype_margin reads the cohort through this helper
    fw = T25RUN.forward_sums(f"kidney_s{seed}", "kidney", run, threads, batch_size)
    fw["run"] = np.array(run)
    tmp = OUT / f"fw_s{seed}.tmp.npz"
    np.savez_compressed(tmp, **fw)
    tmp.replace(OUT / f"fw_s{seed}.npz")


def cippa_sets(symbols: pd.Series) -> dict[str, np.ndarray]:
    from analysis.tasks.P13_kidney import cippa as CP

    L = CP.lists()
    sym = symbols.astype(str)
    uniq = sym[~sym.duplicated(keep=False)]
    out = {}
    for (tag, name), genes in L.items():
        key = f"{tag}|{name}"
        out[key] = np.isin(sym.to_numpy(), list(set(genes) & set(uniq.to_numpy())))
    return out


def loo_b1(a: ad.AnnData, types: list[str]) -> pd.DataFrame:
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    species = a.obs["species"].astype(str).to_numpy()
    sl = sorted(set(species))
    x, f, _ = T42.pseudobulk(X, species, a.obs["cell_ontology_class"].astype(str).to_numpy(),
                             a.obs["sample_batch"].astype(str).to_numpy(), T42.MIN_CELLS_TYPE)
    E_all, _, info = T42.pooled_E(x, f, sl, a.n_vars)
    shared = info[0]["groups"]
    out = {"E_all": E_all}
    for k in types:
        if k not in shared:
            continue
        xk = {key: v for key, v in x.items() if key[1] != k}
        fk = {key: v for key, v in f.items() if key[1] != k}
        Ek, _, _ = T42.pooled_E(xk, fk, sl, a.n_vars)
        out[f"increase_{k}"] = Ek - E_all
    return pd.DataFrame(out, index=a.var_names)


def combine() -> None:
    a = ad.read_h5ad(gene_ids.h5ad_path("kidney"))
    seeds = seed_runs()
    decs = {}
    for s in seeds:
        fw = dict(np.load(OUT / f"fw_s{s}.npz", allow_pickle=False))
        if list(fw["var_names"]) != list(a.var_names):
            raise AssertionError(f"s{s}: var_names differ")
        decs[s] = MG.decompose(fw, "cell_type")
    d0 = decs[min(decs)]
    species = d0["species"]
    mi = species.index("Mouse")
    hi = species.index("Human")
    types = d0["cell_types"]

    def contrib(dec, sp_i):
        lab, w, S = dec["labels"], dec["w"], dec["S_gsk_meanz"]
        return np.stack([(S[(dec["species_idx"] == sp_i) & (lab == k)] * w[(dec["species_idx"] == sp_i) & (lab == k), None]).sum(0)
                         for k in types])  # (K types, G)

    Mm = np.mean([contrib(d, mi) for d in decs.values()], axis=0)
    Hm = np.mean([contrib(d, hi) for d in decs.values()], axis=0)
    seed_rho = {}
    from scipy.stats import spearmanr
    per_seed = {s: contrib(d, mi) for s, d in decs.items()}
    for s in per_seed:
        others = np.mean([v for t, v in per_seed.items() if t != s], axis=0)
        seed_rho[int(s)] = float(spearmanr(per_seed[s].ravel(), others.ravel())[0])
    div = np.maximum(-Mm, 0)
    tot = div.sum(0)
    share = np.where(tot > 0, div / np.where(tot > 0, tot, 1), np.nan)
    genes = pd.DataFrame(index=a.var_names)
    genes["hsapiens_gene_id"] = gene_ids.ensembl("kidney", a.var_names).values
    genes["symbol"] = gene_ids.symbols("kidney", a.var_names).values
    for j, k in enumerate(types):
        genes[f"mouse_C_{k}"] = Mm[j]
        genes[f"human_C_{k}"] = Hm[j]
        genes[f"div_share_{k}"] = share[j]
    genes["mouse_div_total"] = tot
    genes["mouse_S_total"] = Mm.sum(0)
    b = loo_b1(a, types)
    inc = b[[c for c in b.columns if c.startswith("increase_")]].to_numpy()
    pos = np.maximum(np.nan_to_num(inc, nan=0.0), 0)
    ptot = pos.sum(1)
    btypes = [c.replace("increase_", "") for c in b.columns if c.startswith("increase_")]
    for j, k in enumerate(btypes):
        genes[f"b1_increase_{k}"] = inc[:, j]
        genes[f"b1_share_{k}"] = np.where(ptot > 0, pos[:, j] / np.where(ptot > 0, ptot, 1), np.nan)
    genes["B1_E_all"] = b["E_all"].to_numpy()
    sets = cippa_sets(genes["symbol"])
    sets["all_vocabulary"] = np.ones(len(genes), bool)
    rows = []
    for name, m in sets.items():
        for score, prefix, ks in (("chimera_mouse_rows", "div_share_", types), ("b1_loo", "b1_share_", btypes)):
            S_ = genes[[f"{prefix}{k}" for k in ks]].to_numpy()
            ok = m & np.isfinite(S_).all(1)
            rec = dict(gene_set=name, readout=score, n_genes=int(ok.sum()))
            for j, k in enumerate(ks):
                rec[f"share_{k}"] = float(S_[ok, j].mean()) if ok.any() else np.nan
            rec["share_pt_tal"] = float(sum(rec.get(f"share_{k}", 0) for k in EPITHELIAL if f"share_{k}" in rec))
            rec["share_immune_stromal"] = float(sum(rec.get(f"share_{k}", 0) for k in IMMUNE_STROMAL if f"share_{k}" in rec))
            top = np.array(ks)[np.nanargmax(S_[ok], axis=1)] if ok.any() else np.array([])
            rec["frac_top_pt_tal"] = float(np.isin(top, EPITHELIAL).mean()) if len(top) else np.nan
            rows.append(rec)
    tab = pd.DataFrame(rows)
    genes.reset_index(names="var_name").to_parquet(OUT / "drivers_genes.parquet")
    tab.to_csv(OUT / "ek4.tsv", sep="\t", index=False)
    meta = dict(seeds={int(k): v for k, v in seeds.items()}, types=types, b1_types=btypes,
                mouse_row_seed_loo_spearman=seed_rho,
                n_cells_per_type_species={f"{species[s]}|{l}": int(n) for s, l, n in zip(d0["species_idx"], d0["labels"], d0["n"])})
    (OUT / "ek4_meta.json").write_text(json.dumps(meta, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["gene_set", "readout", "n_genes", "share_pt_tal", "share_immune_stromal", "frac_top_pt_tal"]
        print(tab[cols].round(3).to_string(index=False))
        print(json.dumps(seed_rho, indent=1))
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
