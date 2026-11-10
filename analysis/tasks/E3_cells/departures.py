"""E3 analysis C: species-specific departures in fits with three or more species, per cell type, on both gates
and on the direct pseudobulk analogue (rule in ``rules.py``). Run after ``build.py``.

    PYTHONPATH=$PWD python analysis/tasks/E3_cells/departures.py

Outputs ``results/analysis/E3/departures/``: ``rates.csv`` (cohort, readout, species, cell type, tested, departures),
``genes_<cohort>.parquet`` (per gene and species: departure fraction over tested types, per readout),
``branch.csv`` (Spearman of species departure rate with terminal branch length).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.tasks.E3_cells import rules as R
from analysis.tasks.H1_harmonise import rules as RU

OUT = R.OUT / "departures"
DET_EXPR = 0.10


def load_build(c):
    with np.load(R.OUT / "build" / f"{c}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cube(b: dict, key: str, types: list[str], cols: np.ndarray) -> np.ndarray:
    """species x type x gene array of ``key`` for the shared types."""
    sp = list(range(len(b["species"])))
    lab = b["labels"].astype(str)
    out = np.full((len(sp), len(types), len(cols)), np.nan)
    for s in sp:
        for t, k in enumerate(types):
            r = np.where((b["species_idx"] == s) & (lab == k))[0]
            if len(r):
                out[s, t] = b[key][r[0], cols]
    return out


def departures(V: np.ndarray, direct: bool) -> np.ndarray:
    """bool species x type x gene: species s departs (rule in rules.py)."""
    S = V.shape[0]
    dep = np.zeros(V.shape, bool)
    for s in range(S):
        oth = np.delete(V, s, axis=0)
        mo, so = oth.mean(0), oth.std(0, ddof=1)
        dev = V[s] - mo
        if direct:
            dep[s] = (np.abs(dev) >= R.DIRECT_LFC) & (np.abs(dev) >= R.DEP_SD * so)
        else:
            dep[s] = (V[s] <= V.min(0)) & (dev < 0) & (np.abs(dev) >= R.DEP_SD * so)
    return dep


def one(c: str):
    b = load_build(c)
    g = pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet")
    z = pd.read_parquet(R.RES / "Z4" / "h1" / "genes" / f"{c}.parquet").reindex(g.index)
    vn = pd.Index(b["var_names"])
    cols = vn.get_indexer(g.index)
    species = [str(s) for s in b["species"]]
    lab, n, si = b["labels"].astype(str), b["n"], b["species_idx"]
    types = sorted(k for k in set(lab) if all(((si == s) & (lab == k) & (n >= R.MIN_TYPE_CELLS)).any() for s in range(len(species))))
    x = np.log2(1 + b["cp10k"] * 100)
    b = dict(b, x=x)
    X = cube(b, "x", types, cols)
    DET = cube(b, "det", types, cols)
    expressed = (DET >= DET_EXPR).any(0)  # type x gene
    reads = {"direct": (X, g.count_support.to_numpy() | True, True)}
    for gate, key, sup in (("canonical", "S_can", g.count_support.to_numpy()), ("per-gene", "S_free", z.count_support.fillna(False).to_numpy())):
        if not np.isnan(b[key]).all():
            reads[gate] = (cube(b, key, types, cols), sup, False)
    rate_rows, gene_out = [], {}
    # common universe: genes count-supported on the canonical gate (direct and both gates read on the same genes)
    U = g.count_support.to_numpy()
    for rd, (V, sup, direct) in reads.items():
        dep = departures(V, direct)
        ok = expressed[None] & U[None, None, :] & np.isfinite(V).all(0)[None]
        for s, sp in enumerate(species):
            for t, k in enumerate(types):
                m = ok[0, t]
                rate_rows.append(dict(cohort=c, readout=rd, species=sp, cell_type=k, tested=int(m.sum()), departures=int(dep[s, t][m].sum())))
            tested = ok[0].sum(0)
            frac = np.where(tested > 0, (dep[s] & ok[0]).sum(0) / np.maximum(tested, 1), np.nan)
            gene_out[f"{rd}__{sp}"] = frac
    gd = pd.DataFrame(gene_out, index=g.index)
    gd.insert(0, "symbol", g.symbol)
    gd.insert(1, "hsapiens_gene_id", g.hsapiens_gene_id)
    gd["count_support"], gd["nameable"], gd["saturated"] = g.count_support, g.nameable, g.saturated_flag
    gd["nameable_free"], gd["saturated_free"] = z.nameable, z.saturated_flag
    gd["n_types_tested"] = (expressed & U[None, :]).sum(0)
    return pd.DataFrame(rate_rows), gd, types


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rates, br = [], []
    for c in R.BRANCH_MY:
        r, gd, types = one(c)
        gd.to_parquet(OUT / f"genes_{c}.parquet")
        rates.append(r)
        tot = r.groupby(["readout", "species"])[["tested", "departures"]].sum()
        tot["rate"] = tot.departures / tot.tested
        for rd, t in tot.groupby(level=0):
            t = t.droplevel(0)
            bl = pd.Series(R.BRANCH_MY[c]).reindex(t.index)
            rho = spearmanr(t.rate, bl)[0] if len(t) >= 3 else np.nan
            br.append(dict(cohort=c, readout=rd, n_types=len(types), rho_rate_branch=rho, **{f"rate_{k}": v for k, v in t.rate.items()}))
        print(c, len(types), "types", flush=True)
    pd.concat(rates).to_csv(OUT / "rates.csv", index=False)
    pd.DataFrame(br).to_csv(OUT / "branch.csv", index=False)
    print(pd.DataFrame(br).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
