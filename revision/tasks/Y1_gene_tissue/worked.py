"""Y1 (d): worked genes with raw counts and the species x cell type decomposition. Gene list given on the command line.

    python -I launch.py revision.tasks.Y1_gene_tissue.worked SYMBOL [SYMBOL ...]

Writes worked_counts.csv (tissue, species, type: cells, pooled counts, expressing cells, CPM) and worked_contrib.csv (species x type margin
S_gsk and weighted contribution from the y1half forward pass, seed 42) and worked_readouts.csv.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from revision.tasks.Y1_gene_tissue import common as Y
from revision.tasks.Y1_gene_tissue import prep as P
from revision.tasks.Y1_gene_tissue import rules as RU

MAT, ORDER = P.MAT, RU.TISSUES
OUT = RU.OUT
GOLDT = {"liver": "G_liver", "testis": "G_testis", "kidney": "G_kidney", "islet": "G_pancreas"}


def main(genes):
    sym = pd.read_parquet(MAT / "symbols.parquet")["symbol"]
    inv = {v: k for k, v in sym.items() if isinstance(v, str)}
    zz = {k: pd.read_parquet(MAT / f"z_{k}__ED.parquet") for k in ("S", "B1", "D", "Sdir_pattern", "Sdir_level")}
    sup, strict = pd.read_parquet(MAT / "support.parquet"), pd.read_parquet(MAT / "strict.parquet")
    gold = pd.read_parquet(MAT / "golds.parquet")
    cnts, ctr, rd = [], [], []
    for t in ORDER:
        tb = Y.tissue_table(t).set_index("hsapiens_gene_id")
        pr = np.load(Y.CACHE / f"profiles__{t}.npz")
        vn = {v: i for i, v in enumerate(pr["var_names"].astype(str))}
        types = pr["types"].astype(str)
        fw = dict(np.load(Y.FW / f"fw_{t}_s42.npz", allow_pickle=False))
        gr = fw["y1half__groups"]
        sidx, lab, n = np.asarray(gr["species_idx"], int), np.asarray(gr["label"]).astype(str), np.asarray(gr["n"], float)
        typ = np.array([l.rsplit("|", 1)[0] for l in lab])
        spf = [str(x) for x in fw["species"]]
        fvn = {v: i for i, v in enumerate(fw["var_names"].astype(str))}
        for g in genes:
            gid = inv.get(g)
            if gid is None or gid not in tb.index:
                continue
            v = tb.loc[gid, "var_name"]
            i = vn[v]
            for s in pr["species"].astype(str):
                for j, k in enumerate(types):
                    c, e, nc, ls = pr[f"counts_{s}"][j, i], pr[f"nexp_{s}"][j, i], pr[f"ncells_{s}"][j], pr[f"libsum_{s}"][j]
                    if nc == 0:
                        continue
                    cnts.append(dict(gene=g, tissue=t, species=s, cell_type=k, shared=k in pr["shared"].astype(str), n_cells=int(nc), counts=int(c), n_expressing=int(e),
                                     pct_expressing=100 * e / nc, cpm=1e6 * c / ls))
            fi = fvn[v]
            for si, s in enumerate(spf):
                if s.lower() not in ("human", "mouse"):
                    continue
                ns = n[sidx == si].sum()
                for k in np.unique(typ[sidx == si]):
                    mk = (sidx == si) & (typ == k)
                    ctr.append(dict(gene=g, tissue=t, species=s, cell_type=k, n_cells=int(n[mk].sum()), S_gsk=float(fw["y1half__sum_meanz"][mk, fi].sum() / n[mk].sum()),
                                    C_gsk=float(fw["y1half__sum_meanz"][mk, fi].sum() / ns / 2)))
            r = dict(gene=g, tissue=t, S=tb.at[gid, "S"], D=tb.at[gid, "D"], B1=tb.at[gid, "B1"], ppass=tb.at[gid, "ppass"], support=bool(sup.at[gid, t]), strict=bool(strict.at[gid, t]),
                     sign_stable=bool(tb.at[gid, "sign_stable_S"]), top_type=tb.at[gid, "top_type"], tau=tb.at[gid, "tau"], amp=tb.at[gid, "amp"], shift_mouse_minus_human=tb.at[gid, "shift"])
            for k, Z in zz.items():
                r[f"z_{k}"] = Z.at[gid, t]
            if t in GOLDT and gid in gold.index:
                r["G"] = gold.at[gid, GOLDT[t]]
            rd.append(r)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(cnts).to_csv(OUT / "worked_counts.csv", index=False)
    pd.DataFrame(ctr).to_csv(OUT / "worked_contrib.csv", index=False)
    pd.DataFrame(rd).to_csv(OUT / "worked_readouts.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 40):
        print(pd.DataFrame(rd).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
