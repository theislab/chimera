"""Y5 F1: protein properties against S, D_raw, GIN protein-only and expression-only responses, with comparators.

Usage: python -m revision.tasks.Y5_aspects.f1 [scan|ridge]
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from revision.tasks.T4_3_scoreboard import stats as ST
from revision.tasks.Y5_aspects import common as K, rules as RU

COHORTS = RU.PRIMARY + RU.SENS_RAWTAIL
READ = RU.F1_READOUTS + ["B1b", "B1c", "Sdir"]


def universe(g: pd.DataFrame, prop: str, half: str, covs) -> pd.DataFrame:
    m = (~g.masked.astype(bool)) & (g.half == half)
    cols = [prop] + covs + READ + ["block"]
    return g[m & g[cols].replace([np.inf, -np.inf], np.nan).notna().all(1)]


def scan():
    rows = []
    for c in COHORTS:
        g = K.genes(c)
        for prop in RU.PROPERTIES:
            covs = K.C3 if prop == "log_length" else K.C4
            for half in ("odd", "even"):
                T = universe(g, prop, half, covs)
                r = K.scan(T, prop, READ, covs, seed=K.useed(c, prop, half))
                r["cohort"], r["half"], r["property"], r["cov"] = c, half, prop, "C4"
                rows.append(r)
                if "tau_shared" in g and prop != "log_length":
                    Tp = T[T.tau_shared.notna()]
                    r2 = K.scan(Tp, prop, READ, K.CPAT, seed=K.useed(c, prop, half, "p"), draws=200)
                    r2["cohort"], r2["half"], r2["property"], r2["cov"] = c, half, prop, "Cpat"
                    rows.append(r2)
        print(c, flush=True)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(K.OUT / "f1_scan.csv", index=False)
    # external tissues, S and D beside
    ext = []
    ann = pd.read_parquet(K.OUT / "annot.parquet").drop(columns=["chrom"])
    chrom = pd.read_parquet(K.OUT / "chromosomes.parquet").set_index("hsapiens_gene_id").chrom
    for nm, f in RU.EXTERNAL_TISSUE.items():
        x = pd.read_parquet(K.REPO / f)
        Sname = "S_ema"
        x = x.rename(columns={Sname: "S"}).merge(ann, on="hsapiens_gene_id", how="left")
        x["half"] = np.where(x.hsapiens_gene_id.map(chrom).isin(RU.ODD), "odd", np.where(x.hsapiens_gene_id.map(chrom).isin(RU.EVEN), "even", "other"))
        x = x[x.S != 0]
        for prop in RU.PROPERTIES:
            covs = K.C3 if prop == "log_length" else K.C4
            for half in ("odd", "even"):
                T = x[(x.half == half) & x[[prop] + covs + ["S", "D_raw", "B1b", "block"]].notna().all(1)]
                r = K.scan(T, prop, ["S", "D_raw", "B1b"], covs, seed=K.useed(nm, prop, half), draws=200)
                r["cohort"], r["half"], r["property"], r["cov"] = nm, half, prop, "C4"
                ext.append(r)
    pd.concat(ext, ignore_index=True).to_csv(K.OUT / "f1_external.csv", index=False)
    return t


def ridge():
    """Protein-language regressor fitted on odd genes, tested on even genes; within tissue and across tissues."""
    out = []
    tabs = {}
    for c in COHORTS + ["liv_hmonkey", "test_hmac"]:
        g = K.genes(c)
        x3 = pd.read_parquet(K.REPO / f"results/revision/X3/genes/{c}.parquet")
        E = x3[[f"esm_{i}" for i in range(64)]].to_numpy(float)
        tabs[c] = (g, E)
    bm = {c: tabs[c][0].set_index("hsapiens_gene_id") for c in tabs}
    for c in COHORTS:
        g, E = tabs[c]
        for y in ["D_raw", "S", "B1b", "B1c", "Sdir"]:
            m = (~g.masked.astype(bool)) & g[[y] + K.C4 + ["block"]].notna().all(1)
            if m.sum() < 500:
                continue
            res = pd.Series(np.nan, index=g.index)
            T = g[m]
            res[m] = ST.resid(ST.midrank(T[y].to_numpy(float)), ST.design(T, K.C4))
            tr = m & (g.half == "odd")
            fit = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(E[tr], res[tr])
            g2 = g.copy()
            g2[f"pred_{y}"] = fit.predict(E)
            for guard, te in (("all", m & (g.half == "even")), ("no_odd_paralog", m & (g.half == "even") & ~g.has_odd_paralog)):
                Tt = g2[te]
                r = K.scan(Tt, f"pred_{y}", [y], K.C4, seed=K.useed(c, y, guard))
                r["train"], r["target_cohort"], r["guard"] = c, c, guard
                out.append(r)
            # across tissues: same regressor, the other tissue's readout on even genes
            for d in COHORTS:
                if d == c:
                    continue
                gd = bm[d]
                ids = g2.hsapiens_gene_id
                tgt = ids.map(gd[y].to_dict()) if y in gd else None
                if tgt is None:
                    continue
                Tt = g2[(g.half == "even")].copy()
                for k in K.C4 + ["block", "masked"]:
                    Tt[k] = Tt.hsapiens_gene_id.map(gd[k].to_dict())
                Tt["tgt"] = Tt.hsapiens_gene_id.map(gd[y].to_dict())
                Tt = Tt[Tt[K.C4 + ["block", "tgt"]].notna().all(1) & ~Tt.masked.astype(bool)]
                for guard, sel in (("all", np.ones(len(Tt), bool)), ("no_odd_paralog", ~Tt.has_odd_paralog.to_numpy())):
                    r = K.scan(Tt[sel], f"pred_{y}", ["tgt"], K.C4, seed=K.useed(c, d, y, guard))
                    r["y"], r["train"], r["target_cohort"], r["guard"] = y, c, d, guard
                    out.append(r)
        print(c, flush=True)
    pd.concat(out, ignore_index=True).to_csv(K.OUT / "f1_ridge.csv", index=False)


if __name__ == "__main__":
    dict(scan=scan, ridge=ridge)[sys.argv[1] if len(sys.argv) > 1 else "scan"]()
