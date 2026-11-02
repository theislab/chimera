"""X4 (EXPLORATORY): independent per-type and per-donor tables from the gold atlases.

1. TS / TMS liver, per (atlas, type, donor) count sums (T4.3 ``h3_inputs``, same vocabulary and labels):
   - between-donor SD of hepatocyte log2(1 + CPM), donors with >= 30 hepatocytes;
   - TS vs TMS per-type level difference, for replication of the CAMEX gap;
   - donor-half reliability of the per-type level gold of T4.3 (20 splits).
2. Murat held-out species, per coarse type: level matching among species, mean over species pairs of
   -|rank_s1 - rank_s2| within the type (the testis analogue of the T4.3 driver gold), genes detected
   in >= 1% of the type's cells in at least 5 species.

Writes results/revision/X4/cache/{atlas_h3.parquet,atlas_h3_rel.csv,atlas_h4_types.parquet}.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from revision.tasks.T4_3_scoreboard import golds as GO
from revision.tasks.X4_where_differs import common as CM

MIN_DONOR_CELLS = 30


def donor_table(inp) -> pd.DataFrame:
    genes = pd.Index(inp["genes"], name="hsapiens_gene_id")
    types = list(inp["types"])
    out = {}
    for name in ("TS", "TMS"):
        d = inp[name]
        t_of = np.array([k.split("|")[0] for k in d["keys"]])
        for t in types:
            m = (t_of == t)
            S, Nc, Ne = d["S"][m], d["Nc"][m], d["Nexp"][m]
            pooled = S.sum(0)
            out[f"{name}_x_{t}"] = np.log2(1 + 1e6 * pooled / max(pooled.sum(), 1))
            out[f"{name}_f_{t}"] = Ne.sum(0) / max(Nc.sum(), 1)
            use = Nc >= MIN_DONOR_CELLS
            if use.sum() >= 3:
                X = np.log2(1 + 1e6 * S[use] / S[use].sum(1, keepdims=True))
                out[f"{name}_donorSD_{t}"] = X.std(0, ddof=1)
                out[f"{name}_ndonor_{t}"] = np.full(len(genes), int(use.sum()))
    return pd.DataFrame(out, index=genes)


def driver_rel(inp, n_splits=20) -> pd.DataFrame:
    """Split-half reliability of Gdrv per type, donors halved in each atlas; Spearman-Brown."""
    rng = np.random.default_rng(CM.useed("drv_rel"))
    donors = {n: sorted({k.split("|")[1] for k in inp[n]["keys"]}) for n in ("TS", "TMS")}
    n = len(inp["genes"])
    types = list(inp["types"])

    def gdrv(prof):
        o = {}
        for k, t in enumerate(types):
            rt = rankdata(prof["TS_x"][:, k]) / n
            rm = rankdata(prof["TMS_x"][:, k]) / n
            ok = (prof["TS_f"][:, k] >= 0.01) & (prof["TMS_f"][:, k] >= 0.01)
            o[t] = np.where(ok, -np.abs(rt - rm), np.nan)
        return o

    rows = []
    for i in range(n_splits):
        ha, hb = {}, {}
        for nm, ds in donors.items():
            p = rng.permutation(ds)
            ha[nm], hb[nm] = set(p[: len(ds) // 2]), set(p[len(ds) // 2:])
        _, a = GO.h3_gold(inp, ha, profiles=True)
        _, b = GO.h3_gold(inp, hb, profiles=True)
        ga, gb = gdrv(a), gdrv(b)
        for j, t in enumerate(types):
            ok = np.isfinite(ga[t]) & np.isfinite(gb[t])
            r = np.corrcoef(rankdata(ga[t][ok]), rankdata(gb[t][ok]))[0, 1]
            rows.append(dict(split=i, type=t, r=r, n=int(ok.sum()), min_cells_a=min(a["TS"][j], a["TMS"][j]),
                             min_cells_b=min(b["TS"][j], b["TMS"][j])))
    df = pd.DataFrame(rows)
    s = df.groupby("type").agg(r=("r", "mean"), n=("n", "mean"), min_cells=("min_cells_a", "min")).reset_index()
    s["rel"] = 2 * s.r / (1 + s.r)
    return s


def h4_types(inp) -> pd.DataFrame:
    res = GO.h4_species_profiles(inp)
    n = len(inp["genes"])
    genes = pd.Index(inp["genes"], name="hsapiens_gene_id")
    spp = GO.PRIMATES + GO.OUTGROUPS
    out = {}
    for j, t in enumerate(GO.H4_TYPES):
        R = {}
        for s in spp:
            d = inp["species"][s]
            prof, _ = res[s]
            x = prof[:, j]
            t_of = np.array([k.split("|")[0] for k in d["keys"]])
            m = (t_of == t) & (d["Nc"] >= GO.MIN_CELLS)
            f = np.full(n, np.nan)
            f[d["rows"]] = d["Nexp"][m].sum(0) / max(d["Nc"][m].sum(), 1)
            r = np.full(n, np.nan)
            ok = np.isfinite(x)
            r[ok] = rankdata(x[ok]) / ok.sum()
            r[~(f >= 0.01)] = np.nan
            R[s] = r
        diffs = [-np.abs(R[a] - R[b]) for a, b in itertools.combinations(spp, 2)]
        M = np.column_stack(diffs)
        nsp = np.sum([np.isfinite(R[s]) for s in spp], 0)
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                g = np.nanmean(M, 1)
        g[nsp < 5] = np.nan
        out[f"Gdrv4_{t}"] = g
        out[f"nsp_{t}"] = nsp
        P = np.column_stack([prof_ for prof_ in [res[s][0][:, j] for s in spp]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"H4_x_{t}"] = np.nanmean(P, 1)
    return pd.DataFrame(out, index=genes)


def main():
    CM.CACHE.mkdir(parents=True, exist_ok=True)
    inp = GO.h3_inputs()
    t = donor_table(inp)
    t.to_parquet(CM.CACHE / "atlas_h3.parquet")
    rel = driver_rel(inp)
    rel.to_csv(CM.CACHE / "atlas_h3_rel.csv", index=False)
    print(rel, flush=True)
    h = GO.h4_inputs()
    h4_types(h).to_parquet(CM.CACHE / "atlas_h4_types.parquet")
    print("done", flush=True)


if __name__ == "__main__":
    main()
