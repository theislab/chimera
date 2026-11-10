"""X8 step (e): named examples that pass every check.

Per (gene, cohort) with a call: in the universe (count-supported in both gates), call also at q < 0.01, not saturated in either
gate (alpha > 0.95 in any seed), per-gene-gate `nameable` (strict support, no zero or quiet species, no mapping flag, no reference
artefact), O1 table-wide flags clean (flags_by_cohort: not T2_table, R2_table, T3, TZ; covers the zebrafish pairs that enter liver
priors), every table pair of the gene at identity >= MIN_PAIR_IDENTITY. A gene is listed when it passes in >= 1 thread cohort; its
calls in all 26 cohorts and its recurrence over the thread are reported beside, with its five nearest ESM-2 neighbours.

    PYTHONPATH=$PWD python -m analysis.tasks.X8_gate_divergence.named
Output: results/analysis/X8/named_checks.csv (every call with every check), named.csv (passing genes, one row per gene)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.common import gene_ids
from analysis.tasks.X8_gate_divergence import rules as RU

OUT = RU.OUT
O1 = RU.RES / "O1" / "flags_by_cohort.parquet"


def table_identity(c: str, reg: dict, var_names: pd.Index, hs: pd.Series) -> pd.Series:
    """Minimum identity over the table's non-human pairs, per gene; checks the human ID against the X8 table."""
    for key in (c, reg[c].get("table")):
        try:
            t = gene_ids.load_table(key)
        except Exception:  # noqa: BLE001
            continue
        ids = pd.Index(var_names).str.replace("ortho_", "", regex=False).astype(int)
        if not ids.isin(t.index).all():
            continue
        tt = t.loc[ids]
        if (tt.hsapiens_gene_id.to_numpy() != hs.to_numpy()).mean() > 0.01:
            continue
        cols = [x for x in tt.columns if x.endswith("_identity") and x != "mean_identity"]
        return pd.Series(tt[cols].min(1).to_numpy(float), index=var_names)
    return pd.Series(np.nan, index=var_names)


def main() -> int:
    reg = RU.cohorts()
    th = RU.thread()
    tissue_of = {c: reg[c]["tissue"] for c in reg}
    fl = pd.read_parquet(O1)
    rec = pd.read_csv(OUT / "recurrence_genes.csv").set_index("hsapiens_gene_id")
    rows = []
    for c in reg:
        g = pd.read_parquet(OUT / "genes" / f"{c}.parquet")
        g = g[g.call != 0].copy()
        if not len(g):
            continue
        f = fl[fl.cohort == c].set_index("var_name").reindex(g.index)
        o1_bad = np.zeros(len(g), bool)
        for col in ("T2_table", "R2_table", "T3", "TZ"):
            if col in f:
                o1_bad |= f[col].fillna(False).astype(bool).to_numpy()
        o1_missing = f.cohort.isna().to_numpy()
        if c not in set(fl.cohort):  # C1 cohorts (joined by Ensembl ID, no table-wide audit): O1's core flags on the gene table
            o1_bad = g.o1_T2_free.fillna(True).astype(bool).to_numpy() | g.o1_zero_any_free.fillna(True).astype(bool).to_numpy()
            o1_missing = np.zeros(len(g), bool)
        ident = table_identity(c, reg, g.index, g.hsapiens_gene_id)
        g["thread"] = c in th.values()
        g["tissue"] = tissue_of[c]
        g["chk_strict_q01"] = g.call_strict == g.call
        g["chk_not_saturated"] = ~g.saturated_any.astype(bool)
        g["chk_nameable_free"] = g.nameable_free.fillna(False).astype(bool)
        g["chk_o1_table"] = ~o1_bad & ~o1_missing
        g["min_pair_identity"] = ident.to_numpy()
        g["chk_identity"] = g.min_pair_identity >= RU.MIN_PAIR_IDENTITY
        g["pass_all"] = g[[x for x in g.columns if x.startswith("chk_")]].all(1)
        rows.append(g.reset_index())
    A = pd.concat(rows, ignore_index=True)
    keep = ["var_name", "cohort", "tissue", "thread", "hsapiens_gene_id", "symbol", "call", "delta", "t", "z_free", "z_can", "D_free", "D_can", "S_free", "S_can", "pi_raw",
            "expr", "detection", "tau", "min_pair_identity", "nbr_symbols", "nbr_zfree5", "nbr_mean_free"] + [x for x in A.columns if x.startswith("chk_")] + ["pass_all"]
    A[keep].to_csv(OUT / "named_checks.csv", index=False)
    print(A.groupby("cohort")[["pass_all"]].agg(["sum", "count"]).to_string())
    P = A[A.pass_all]
    out = []
    for gid, g in P.groupby("hsapiens_gene_id"):
        thr = g[g.thread]
        if not len(thr):
            continue
        allc = A[A.hsapiens_gene_id == gid]
        r = rec.loc[gid] if gid in rec.index else None
        dirs = np.sign(allc.call)
        tis_sign = g.groupby("tissue").call.agg(lambda v: int(np.sign(v).mean()) if len(set(np.sign(v))) == 1 else 0)
        n_up_t, n_dn_t = int((tis_sign > 0).sum()), int((tis_sign < 0).sum())
        out.append(dict(hsapiens_gene_id=gid, symbol=g.symbol.iloc[0], direction="up" if thr.call.iloc[0] > 0 else "down",
                        n_tissues_pass_up=n_up_t, n_tissues_pass_down=n_dn_t, tissues_pass=";".join(f"{k}{'+' if v > 0 else '-' if v < 0 else '?'}" for k, v in tis_sign.items()),
                        thread_tissues_pass=";".join(sorted(thr.tissue.unique())), n_thread_tissues_pass=thr.tissue.nunique(),
                        tissues_called_any=";".join(sorted(allc.tissue.unique())), n_tissues_called_any=allc.tissue.nunique(),
                        cohorts_called=";".join(f"{c}{'+' if s > 0 else '-'}" for c, s in zip(allc.cohort, dirs)), sign_consistent=bool(len(set(dirs)) == 1),
                        R_thread=int(r.R) if r is not None else np.nan, q_rec=float(r.q_rec) if r is not None else np.nan,
                        mean_delta_thread=float(thr.delta.mean()), z_free=float(thr.z_free.mean()), z_can=float(thr.z_can.mean()), pi_raw=float(thr.pi_raw.mean()),
                        expr=float(thr.expr.mean()), min_pair_identity=float(thr.min_pair_identity.min()), nbr_symbols=thr.nbr_symbols.iloc[0], nbr_zfree5=thr.nbr_zfree5.iloc[0]))
    N = pd.DataFrame(out)
    N["n_tissues_pass_same"] = N[["n_tissues_pass_up", "n_tissues_pass_down"]].max(1)
    N["clean"] = N.n_tissues_pass_same.ge(2) & (N[["n_tissues_pass_up", "n_tissues_pass_down"]].min(1) == 0) & N.sign_consistent
    N = N.sort_values(["clean", "n_tissues_pass_same", "n_tissues_called_any"], ascending=False)
    N.to_csv(OUT / "named.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(N.head(60).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
