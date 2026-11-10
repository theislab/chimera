"""EXPLORATORY: one gene table per cohort from existing outputs. Nothing is trained or tested here.

Reads T4_3_scoreboard cache, T3_3_consensus consensus, per-seed final states (read-only), T4_3_scoreboard baselines and gold table, T2_5_celltype_margin
flags, tables of X1_testis_discovery, X2_liver_biology and X3_cross_cohort (stage peaks, hepatocyte enrichment, contributions, per-type
pseudobulk, classes).
Writes results/analysis/X4/genes/<cohort>.parquet.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.common import controls as K
from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.X4_where_differs import common as CM
from analysis.tasks.X4_where_differs import rules as RU

sys.path.insert(0, str(CM.REPO / "analysis/tasks/X3_cross_cohort"))
import annotations as AN  # noqa: E402

from analysis.tasks.X1_testis_discovery import leads as X1  # noqa: E402

SPECIES = {"liv_hmouse": ["Human", "Mouse"], "test_hmouse": ["Human", "Mouse"]}


def level_pattern(g: pd.DataFrame) -> pd.DataFrame:
    """Exact decomposition of the Human-Mouse per-type difference (rules.py)."""
    types = [c[len("x_Mouse_"):] for c in g.columns if c.startswith("x_Mouse_") and f"x_Human_{c[8:]}" in g.columns]
    H = g[[f"x_Human_{t}" for t in types]].to_numpy(float)
    M = g[[f"x_Mouse_{t}" for t in types]].to_numpy(float)
    fH = g[[f"f_Human_{t}" for t in types]].to_numpy(float)
    fM = g[[f"f_Mouse_{t}" for t in types]].to_numpy(float)
    D = M - H
    for j in range(D.shape[1]):
        ok = (fH[:, j] >= 0.01) & (fM[:, j] >= 0.01)
        D[:, j] -= np.nanmedian(D[ok, j])
    shift = D.mean(1)
    shape = D.std(1)
    msd = (D ** 2).mean(1)
    out = pd.DataFrame(dict(shift=shift, abs_shift=np.abs(shift), shape=shape, msd=msd,
                            frac_level=shift ** 2 / np.where(msd > 0, msd, np.nan),
                            spread=(H.std(1) + M.std(1)) / 2), index=g.index)
    for j, t in enumerate(types):
        out[f"lfc_{t}"] = D[:, j]
    return out


def build(c: str, gold: pd.DataFrame) -> pd.DataFrame:
    d = I.load(c, CM.RES / "T4_3/cache")
    t = d["tab"].copy()
    cons = pd.read_csv(CM.RES / f"T3_3/{c}__consensus.csv")
    assert (cons.var_name.to_numpy() == t.var_name.to_numpy()).all()
    for k in ("symbol", "prior", "alpha", "sign_consistency", "n_seeds"):
        t[k] = cons[k].to_numpy()
    t = t.rename(columns={"S_ema": "S"})
    runs = d["meta"]["seeds"]
    A, Sg = [], []
    for s, r in sorted(runs.items(), key=lambda kv: int(kv[0])):
        st = K.read_run(r)
        A.append(st["alpha"])
        Sg.append(st["S_g_ema"])
    A, Sg = np.column_stack(A), np.column_stack(Sg)
    Ds = A - t["prior_raw"].to_numpy()[:, None]
    Ds = Ds - np.median(Ds, 0)
    if not np.allclose(Sg.mean(1), t["S"].to_numpy(), atol=1e-5):
        raise AssertionError(f"{c}: seed S mean differs from the consensus")
    t["signfrac_D"] = (np.sign(Ds) == np.sign(t["D_raw"].to_numpy())[:, None]).mean(1)
    t["signfrac_S"] = (np.sign(Sg) == np.sign(t["S"].to_numpy())[:, None]).mean(1)
    t["masked"] = np.abs(t["S"].to_numpy()) < 1e-6
    b = pd.read_csv(CM.RES / f"T4_3/baselines_consensusD/baselines__{c}.csv").set_index("var_name")
    t["B3b"] = b["B3b"].reindex(t["var_name"]).to_numpy()
    for k in ("B1b_raw", "B1c_raw", "B1z_raw", "B2b", "B3a"):
        t[k] = b[k].reindex(t["var_name"]).to_numpy()
    t = t.drop(columns=["B1b", "B1c", "B1z"], errors="ignore")
    t = t.join(gold, on="hsapiens_gene_id")
    t["sign_stable"] = (t.signfrac_D == 1) & (t.signfrac_S == 1)
    t["nameable"] = (t.margin_support.eq(True) & t.sign_stable & (np.sign(t.D_raw) == np.sign(t.S)) & ~t.masked
                     & t.nonzero_all_species)
    if c.startswith("liv"):
        t.loc[t.symbol.astype(str).str.startswith("MT-"), "nameable"] = False
    x3 = pd.read_parquet(CM.RES / f"X3/genes/{c}.parquet").set_index("var_name")
    keep = [k for k in x3.columns if (k.startswith("C_") or k.startswith("Csp_")) and not k.startswith("C_Human")]
    t = t.join(x3[keep], on="var_name")
    if c.startswith("liv"):
        x2 = pd.read_parquet(CM.RES / f"X2/genes/{c}.parquet").set_index("var_name")
        ren = {k: k.replace("__", "_") for k in x2.columns if k.startswith(("x__", "f__"))}
        t = t.join(x2[list(ren) + ["hep_enrich"]].rename(columns=ren), on="var_name")
    else:
        x1 = pd.read_parquet(CM.RES / f"X1/genes/{c}.parquet").set_index("var_name")
        pb = [k for k in x1.columns if k.startswith(("x_", "f_")) and not k.startswith("f_Human_x")]
        t = t.join(x1[pb], on="var_name")
        t["peak"] = X1.peak_stage(t)
        t["peak_strict"] = X1.peak_stage(t, 2.0, 0.20)
    cl = AN.gene_classes(pd.Index(t["hsapiens_gene_id"]))
    for k in AN.CLASS_ORDER:
        t[f"cls_{k}"] = cl[k].to_numpy()
    if c in SPECIES:
        lp = level_pattern(t)
        t = pd.concat([t, lp], axis=1)
    return t


def main():
    CM.GDIR.mkdir(parents=True, exist_ok=True)
    gold = pd.read_csv(CM.RES / "T4_3/golds/gold_table.csv", index_col=0)
    for c in RU.COHORTS:
        t = build(c, gold)
        t.to_parquet(CM.GDIR / f"{c}.parquet")
        print(c, t.shape, "nameable", int(t.nameable.sum()), "seeds", int(t.n_seeds.max()),
              "median signfrac_D", t.signfrac_D.median(), flush=True)


if __name__ == "__main__":
    main()
