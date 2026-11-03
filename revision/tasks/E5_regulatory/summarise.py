"""E5 summaries: generality across tissues on both gates, sensitivities, species-pair groups, regulons, joint model.

Usage (from chimera/):  mamba run -n chimera_env python -m revision.tasks.E5_regulatory.summarise
Reads results/revision/E5/{scan,regulon,joint}.csv. Writes scan_summary.csv, scan_primary_q.csv, scan_tissue_values.csv,
regulon_summary.csv, regulon_tissue_values.csv, joint_summary.csv beside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from revision.common import gene_ids
from revision.tasks.E2_protein.summarise import classify, tissue_table
from revision.tasks.E5_regulatory import rules as RU
from revision.tasks.T4_3_scoreboard import stats as ST

OUT = gene_ids.REPO_ROOT / "results" / "revision" / "E5"
GATES = ["D_can", "D_free"]


def group(c: str) -> str:
    return "primate" if c in RU.PRIMATE else ("hmouse" if c in RU.HMOUSE else "other")


def verdict(out: pd.DataFrame) -> np.ndarray:
    sgn = np.sign(out.D_can_median) == np.sign(out.D_free_median)
    return np.where(out.D_can_general & out.D_free_general & sgn, "ROBUST",
                    np.where(out.D_free_general, "ONE-GATE free", np.where(out.D_can_general, "ONE-GATE canonical", "NEGATIVE")))


def scan_summary() -> pd.DataFrame:
    s = pd.read_csv(OUT / "scan.csv")
    p = s[s["cov"] == "C"].copy()
    allp = pd.concat([p[f"{y}_p"] for y in GATES + ["S"]])
    q = pd.Series(ST.bh(allp.fillna(1).tolist()), index=allp.index)
    for k, y in enumerate(GATES + ["S"]):
        p[f"{y}_q"] = q.iloc[k * len(p):(k + 1) * len(p)].to_numpy()
    p.to_csv(OUT / "scan_primary_q.csv", index=False)
    p["key"] = p.property
    res = {}
    for y in GATES + ["S"]:
        sig = p.assign(sig=p[f"{y}_q"] < 0.05).groupby(["key", "tissue"]).sig.any().groupby("key").sum()
        res[y] = classify(tissue_table(p, y), sig)
    out = pd.concat(res, axis=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    for cov in ("Ctau", "Cbrd", "Cgc", "Cpub", "Ccls"):
        sc = s[s["cov"] == cov].assign(key=lambda d: d.property)
        for y in GATES + (["B1"] if cov in ("Ctau", "Cbrd") else []):
            tm = tissue_table(sc, y)
            out[f"{cov}_{y}_median"] = tm.median(1)
            out[f"{cov}_{y}_same"] = np.sign(tm).eq(np.sign(tm.median(1)), axis=0).sum(1)
    tb = tissue_table(p, "B1")
    out["B1_median"] = tb.median(1)
    out["B1_same"] = np.sign(tb).eq(np.sign(out["B1_median"]), axis=0).sum(1)
    p["grp"] = p.cohort.map(group)
    for y in GATES + ["B1"]:
        g = p.groupby(["key", "grp"])[y].median().unstack()
        for k in g.columns:
            out[f"{y}_{k}"] = g[k]
    out["n_pos_median"] = p.groupby("key").n_pos.median()
    out["verdict"] = verdict(out)
    out.index.name = "property"
    out = out.sort_values("D_free_median")
    out.to_csv(OUT / "scan_summary.csv")
    tv = [tissue_table(p, y).stack().rename("rho_p").reset_index().assign(readout=y) for y in GATES + ["S", "B1"]]
    for cov in ("Ctau", "Cbrd"):
        sc = s[s["cov"] == cov].assign(key=lambda d: d.property)
        for y in GATES:
            tv.append(tissue_table(sc, y).stack().rename("rho_p").reset_index().assign(readout=f"{y}_{cov}"))
    pd.concat(tv).rename(columns={"key": "property"}).to_csv(OUT / "scan_tissue_values.csv", index=False)
    return out


def regulon_summary() -> pd.DataFrame:
    f = pd.read_csv(OUT / "regulon.csv")
    f["key"] = f.tf
    ys = {"D_can": "r_D_can", "D_free": "r_D_free", "S": "r_S", "tau_D_can": "rt_D_can", "tau_D_free": "rt_D_free",
          "B1": "r_B1", "pub_D_can": "rp_D_can", "pub_D_free": "rp_D_free"}
    allp = pd.concat([f[f"{ys[y]}_p"] for y in GATES])
    q = pd.Series(ST.bh(allp.fillna(1).tolist()), index=allp.index)
    f["D_can_q"] = q.iloc[:len(f)].to_numpy()
    f["D_free_q"] = q.iloc[len(f):].to_numpy()
    res = {}
    for y, col in ys.items():
        sig = f.assign(sig=f[f"{y}_q"] < 0.05).groupby(["key", "tissue"]).sig.any().groupby("key").sum() if y in GATES else None
        res[y] = classify(tissue_table(f, f"{col}_mean"), sig)
    out = pd.concat(res, axis=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["verdict"] = verdict(out)
    out["n_targets_median"] = f.groupby("key").n.median()
    out.index.name = "tf"
    out = out.sort_values("D_free_median")
    out.to_csv(OUT / "regulon_summary.csv")
    tv = [tissue_table(f, f"{col}_mean").stack().rename("mean_resid").reset_index().assign(readout=y) for y, col in ys.items()]
    pd.concat(tv).rename(columns={"key": "tf"}).to_csv(OUT / "regulon_tissue_values.csv", index=False)
    return out


def joint_summary() -> pd.DataFrame:
    j = pd.read_csv(OUT / "joint.csv")
    j["key"] = j.readout + ":" + j.property
    tm = tissue_table(j, "beta")
    out = pd.DataFrame({"median_beta": tm.median(1), "same_sign": np.sign(tm).eq(np.sign(tm.median(1)), axis=0).sum(1),
                        "n_tissues": tm.notna().sum(1)})
    out.index.name = "readout_property"
    out.to_csv(OUT / "joint_summary.csv")
    return out


def main():
    pd.set_option("display.width", 250)
    sc = scan_summary()
    cols = ["D_can_median", "D_can_same_sign", "D_can_n_tissue_sig", "D_free_median", "D_free_same_sign",
            "D_free_n_tissue_sig", "Ctau_D_free_median", "Cbrd_D_free_median", "Cpub_D_free_median", "Ccls_D_free_median",
            "B1_median", "verdict"]
    print(sc[cols].round(3).to_string())
    rg = regulon_summary()
    print(rg.verdict.value_counts())
    js = joint_summary()
    print(js.round(3).to_string())


if __name__ == "__main__":
    main()
