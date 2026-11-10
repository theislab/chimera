"""Y6: one tidy table of (fit, species, gene) rows from stored forward passes and per-gene tables. No label is read here.

    PYTHONPATH=$PWD python -m analysis.tasks.Y6_uses.build_rows
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from analysis.tasks.Y6_uses import rules as RU

RES = RU.RES
TABLE_FILES = {"liver": RU.ROOT / "data/processed/ortholog_table.csv", "testis": RU.ROOT / "data/processed/ortholog_table_testis.csv",
               "van_zyl": RU.ROOT / "data/analysis/van_zyl/ortholog_table_van_zyl.csv",
               "bone_marrow": RU.ROOT / "data/analysis/bone_marrow/ortholog_table_bone_marrow.csv",
               "kidney": RU.ROOT / "data/analysis/disease_kidney/ortholog_table_kidney.csv"}


def forward_files(fit: str, d: dict) -> list[str]:
    if d["kind"] == "x7":
        return [str(RES / "X7/forward" / (f"{fit}.npz" if s is None else f"{fit}_s{s}.npz")) for s in RU.SEEDS_X7[fit]]
    if d["kind"] == "p12":
        return [str(RES / f"P12/drivers/fw_s{s}.npz") for s in (42, 46)]
    if d["kind"] == "t11":
        return sorted(glob.glob(str(RES / f"T11/drivers/fw_{d['t11']}_s*.npz")))
    if d["kind"] == "y1":
        return sorted(glob.glob(str(RES / f"Y1/forward/fw_{d['y1']}_s*.npz")))
    raise KeyError(d["kind"])


def species_margin(files: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Mean and SD over seeds of S_gs, species in columns, var_name in the index."""
    arrs = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        key = "sp_l" if "sp_l" in z.files else "sp_sum_meanz"
        sp = [str(x) for x in z["species"]]
        S = z[key] / np.asarray(z["n_species"], float)[:, None]
        arrs.append(pd.DataFrame(S.T, index=[str(v) for v in z["var_names"]], columns=sp))
    st = np.stack([a.reindex(index=arrs[0].index, columns=arrs[0].columns).to_numpy() for a in arrs])
    m = pd.DataFrame(st.mean(0), index=arrs[0].index, columns=arrs[0].columns)
    sd = pd.DataFrame(st.std(0), index=arrs[0].index, columns=arrs[0].columns)
    return m, sd, len(files)


def gene_table(fit: str, d: dict) -> pd.DataFrame:
    if d["kind"] == "t11" and d["t11"] in ("hpig", "hmfas"):
        g = pd.read_parquet(RES / f"T11/readout/genes_{d['t11']}.parquet").set_index("var_name", drop=False)
        g = g.rename(columns={"prior_raw": "pi_raw"})
        return g
    g = pd.read_parquet(RES / f"H1/genes/{fit}.parquet")
    if "var_name" in g:
        g = g.set_index("var_name", drop=False)
    else:
        g.index.name = "var_name"
    return g


def pair_b1(fit: str, d: dict, label: str, g: pd.DataFrame) -> pd.Series:
    h = d["human"]
    if d["kind"] == "x7":
        e = pd.read_parquet(RES / f"X7/baselines/{fit}__pair_E.parquet").set_index("var_name")
        return e[f"E_{h}_{label}"].reindex(g.index)
    if d["kind"] == "p12":
        e = pd.read_parquet(RES / "P12/readout/bm_genes.parquet")
        e = e.set_index("var_name") if "var_name" in e else e
        return e[f"E_human_{label}"].reindex(g.index)
    if d["kind"] == "t11":
        e = pd.read_parquet(RES / f"T11/readout/genes_{d['t11']}.parquet").set_index("var_name")
        c = f"E_human_{label}"
        return e[c].reindex(g.index) if c in e else e["B1_raw"].reindex(g.index)
    return g["B1"]


def main():
    RU.OUT.mkdir(parents=True, exist_ok=True)
    chrom = pd.read_csv(RES / "X1/genes/chromosome_ensembl110.csv", dtype={"chrom": str}).set_index("hsapiens_gene_id")["chrom"]
    blocks = pd.read_parquet(RES / "Y1/matrices/blocks.parquet")["block"]
    rows, wides = [], {}
    tables = {k: pd.read_csv(v) for k, v in TABLE_FILES.items()}
    for fit, d in RU.FITS.items():
        table = "testis" if fit == "islet" else d["table"]
        tab = tables[table].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
        g = gene_table(fit, d)
        m, sd, nseed = species_margin(forward_files(fit, d))
        m, sd = m.reindex(g.index), sd.reindex(g.index)
        dets = {c.replace("detection_", ""): g[c] for c in g.columns if c.startswith("detection_") and c not in ("detection_spmean", "detection_sprange")}
        wide = pd.DataFrame({"hsapiens_gene_id": g["hsapiens_gene_id"], "symbol": g["symbol"]}, index=g.index)
        wide["fit"] = fit
        for lab, pre in d["species"].items():
            det = dets[lab]
            others = pd.concat([v for k, v in dets.items() if k != lab], axis=1)
            r = pd.DataFrame({"fit": fit, "species_label": lab, "species": pre, "var_name": g.index,
                              "hsapiens_gene_id": g["hsapiens_gene_id"].values, "symbol": g["symbol"].values})
            r["gene_id"] = r.hsapiens_gene_id.map(tab[f"{pre}_gene_id"]) if f"{pre}_gene_id" in tab else np.nan
            r["identity_s"] = r.hsapiens_gene_id.map(tab[f"{pre}_identity"]).values
            r["S_gs"] = m[lab].values
            r["S_gs_sd"] = sd[lab].values
            r["S_human"] = m[d["human"]].values
            r["S_ema"] = g["S_ema"].values
            r["D"] = g["D"].values
            r["det_s"] = det.values
            r["det_human"] = dets[d["human"]].values
            r["det_other"] = others.max(axis=1).values
            r["E_hs"] = pair_b1(fit, d, lab, g).values
            r["log_mean"] = g["log_mean"].values
            r["tau"] = g["tau"].values
            r["pi_raw"] = g["pi_raw"].values
            r["n_seeds"] = nseed
            rows.append(r)
            wide[f"S_{pre}"] = m[lab].values
            wide[f"id_{pre}"] = r["identity_s"].values
            wide[f"E_{pre}"] = r["E_hs"].values
            wide[f"det_{pre}"] = det.values
        wide["S_ema"] = g["S_ema"].values
        wide["S_human"] = m[d["human"]].values
        wide["D"] = g["D"].values
        wide["count_support"] = g["count_support"].values if "count_support" in g else np.nan
        wides[fit] = wide
        print(fit, g.shape, nseed, flush=True)
    out = pd.concat(rows, ignore_index=True)
    out["chrom"] = out.hsapiens_gene_id.map(chrom)
    out["block"] = out.hsapiens_gene_id.map(blocks)
    fallback = (out.hsapiens_gene_id.astype("category").cat.codes.astype(np.int64) % 500) + 100000
    out["block"] = out["block"].fillna(fallback)
    auto = pd.to_numeric(out.chrom, errors="coerce")
    out["split"] = np.where(auto.isin(range(1, 23, 2)), "odd", np.where(auto.isin(range(2, 23, 2)), "even", "none"))
    out.to_parquet(RU.OUT / "rows.parquet")
    pd.concat(wides.values()).to_parquet(RU.OUT / "wide.parquet")
    print(out.groupby(["fit", "species_label"]).agg(n=("S_gs", "size"), odd=("split", lambda x: (x == "odd").sum()), even=("split", lambda x: (x == "even").sum()),
                                                  gene=("gene_id", lambda x: x.notna().sum()), seeds=("n_seeds", "first")))


if __name__ == "__main__":
    main()
