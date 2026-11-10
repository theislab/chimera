"""Y6 use 1, PH6 (post hoc, see ledger): an annotation fault should appear in every tissue. Median across rows of the within-row percentile, per species, even autosomes.

    PYTHONPATH=$PWD python -m analysis.tasks.Y6_uses.use1_consensus
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from analysis.tasks.Y6_uses import rules as RU
from analysis.tasks.Y6_uses.use1 import OUT, ap_setup, block_weights, wap, wprf

ROWS = {"mmusculus": [("liv_4sp", "Mouse"), ("test_3sp", "Mouse"), ("bone_marrow", "mouse"), ("vz_full", "mouse"), ("kidney", "Mouse"), ("islet", "Mouse")],
        "mmulatta": [("liv_4sp", "Monkey"), ("test_3sp", "Macaque"), ("vz_full", "macaca_mulatta")]}
SCORES = ["S_row", "zero_gap", "zero_s", "id_neg"]
MIN_ROWS = {"mmusculus": 3, "mmulatta": 3}


def main():
    d = pd.read_parquet(OUT / "scored_rows.parquet")
    named = pd.read_csv(OUT / "named_else_pairs.csv")
    out = []
    keep = []
    for sp, rows in ROWS.items():
        parts = {s: [] for s in SCORES}
        for fit, lab in rows:
            x = d[(d.fit == fit) & (d.species_label == lab)].set_index("hsapiens_gene_id")
            x = x[~x.index.duplicated()]
            ok = x.det_other >= RU.DET_EXPR
            for s in SCORES:
                parts[s].append(x[s].where(ok).rank(pct=True).rename(fit))
        cons = {s: pd.concat(v, axis=1) for s, v in parts.items()}
        n_rows = cons["S_row"].notna().sum(axis=1)
        any_ = d[(d.species == sp)].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
        T = pd.DataFrame({f"{s}_cons": cons[s].median(axis=1) for s in SCORES})
        T = T[n_rows.reindex(T.index) >= MIN_ROWS[sp]].join(any_[["symbol", "gene_id", "split", "block", "T_lowconf", "T_later", "T_goc", "T_newasm", "T_no3utr", "identity_s", "goc", "wga", "homology_type_116"]])
        T["T_named_else"] = T.gene_id.isin(set(named.gene_id)).astype(float)
        T["species"] = sp
        keep.append(T)
        even = T[T.split == "even"]
        for label in ["T_lowconf", "T_later", "T_goc", "T_named_else"]:
            e = even[even[label].notna()]
            y = e[label].to_numpy(float)
            if y.sum() < 10:
                continue
            setups = {s: ap_setup(e[f"{s}_cons"].fillna(0).to_numpy()) for s in SCORES}
            flags = {s: e[f"{s}_cons"].fillna(0).to_numpy() >= np.quantile(e[f"{s}_cons"].fillna(0), 0.99) for s in SCORES}
            rng = np.random.default_rng(RU.SEED + zlib.crc32(f"{sp}|{label}".encode()) % 100000)
            B = {s: [] for s in SCORES}
            for w in block_weights(e.block.to_numpy(), rng, RU.DRAWS_PR):
                for s in SCORES:
                    B[s].append(wap(y, w, setups[s]))
            for s in SCORES:
                a = wap(y, np.ones(len(y)), setups[s])
                p, r, _ = wprf(y, np.ones(len(y)), flags[s])
                dl = np.array(B[s], float) - np.array(B["zero_gap"], float)
                out.append(dict(species=sp, label=label, score=s, n=len(e), n_pos=int(y.sum()), base_rate=y.mean(), ap=a, ap_lift=a / y.mean(), ap_lo=np.nanpercentile(B[s], 2.5), ap_hi=np.nanpercentile(B[s], 97.5),
                                top1_n=int(flags[s].sum()), top1_precision=p, top1_recall=r, d_zero_gap=a - wap(y, np.ones(len(y)), setups["zero_gap"]), d_lo=np.nanpercentile(dl, 2.5), d_hi=np.nanpercentile(dl, 97.5)))
    pd.DataFrame(out).to_csv(OUT / "posthoc_consensus.csv", index=False)
    pd.concat(keep).to_parquet(OUT / "posthoc_consensus_genes.parquet")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(pd.DataFrame(out).round(3).to_string())
    for sp, T in zip(ROWS, keep):
        print(sp)
        print(T.sort_values("S_row_cons", ascending=False).head(25)[["symbol", "gene_id", "identity_s", "S_row_cons", "zero_gap_cons", "goc", "wga", "homology_type_116", "T_lowconf", "T_later", "T_named_else"]].round(3).to_string())


if __name__ == "__main__":
    main()
