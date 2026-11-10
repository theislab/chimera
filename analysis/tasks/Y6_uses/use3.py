"""Y6 use 3: drug targets, rules.py USE 3.

    PYTHONPATH=$PWD python -m analysis.tasks.Y6_uses.use3
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from statsmodels.stats.multitest import multipletests

from analysis.tasks.X3_cross_cohort import annotations as AN
from analysis.tasks.X4_where_differs import common as CM
from analysis.tasks.Y6_uses import rules as RU

CM.N_JOBS = 12
OUT = RU.OUT / "use3"
MX = RU.RES / "Y1/matrices"
M = RU.DATA / "minikel2024"
OT = RU.DATA / "opentargets"
L_C1 = ["expr", "det", "log_cds_length"]
LADDER = {"C0": [], "C1": L_C1, "C2": L_C1 + ["tau"], "C3": L_C1 + ["tau", "prior_raw"], "C4": L_C1 + ["tau", "prior_raw", "dNdS", "LOEUF"],
          "C5": L_C1 + ["tau", "prior_raw", "dNdS", "LOEUF", "log_pubmed_h", "log_pubmed_m"]}
DRUG_CLASSES = ["receptor", "GPCR", "ion_channel", "protein_kinase", "SLC_transporter", "TF", "enzyme"]


def labels() -> pd.DataFrame:
    pp = pd.read_csv(M / "pp.tsv", sep="\t")
    sym = pd.read_parquet(MX / "symbols.parquet").symbol
    s2e = {s: e for e, s in sym.dropna().items()}
    indic = pd.read_csv(M / "indic.tsv", sep="\t").set_index("indication_mesh_id")
    areas = indic.areas.fillna("").str.split(",").map(lambda v: {x.strip() for x in v if x.strip()})
    pp["areas"] = pp.indication_mesh_id.map(areas).map(lambda x: x if isinstance(x, set) else set())
    pp["onco"] = pp.areas.map(lambda x: "oncology" in x)

    def appr(df):
        g = df.groupby("gene")
        launched = g.ccat.apply(lambda x: (x == "Launched").any())
        active = g.acat.apply(lambda x: x.notna().any())
        mx = g.ccatnum.max()
        y = pd.Series(np.nan, index=launched.index)
        y[launched] = 1.0
        y[(~launched) & (~active) & mx.between(2, 4)] = 0.0
        return y

    out = pd.DataFrame({"y_appr": appr(pp)})
    out["y_appr_nononc"] = appr(pp[~pp.onco])
    clinical = pp[pp.ccatnum >= 2]
    m = pd.read_csv(M / "merge2.tsv.gz", sep="\t", low_memory=False, usecols=["ti_uid", "gene", "ccatnum", "comb_norm", "assoc_source", "l2g_share"])
    supp = m[(m.comb_norm >= 0.8) & (m.ccatnum >= 2)]
    supp2 = supp[(supp.assoc_source != "OTG") | (supp.l2g_share >= 0.5)]
    onc_ti = set(pp.ti_uid[pp.onco])
    out["y_gen"] = pd.Series(np.nan, index=out.index)
    genes_clin = set(clinical.gene)
    idx = pd.Index(sorted(genes_clin))
    yg = pd.Series(idx.isin(set(supp.gene)).astype(float), index=idx)
    yg2 = pd.Series(idx.isin(set(supp2.gene)).astype(float), index=idx)
    clin_no = clinical[~clinical.ti_uid.isin(onc_ti)]
    idx3 = pd.Index(sorted(set(clin_no.gene)))
    yg3 = pd.Series(idx3.isin(set(supp2[~supp2.ti_uid.isin(onc_ti)].gene)).astype(float), index=idx3)
    tab = pd.DataFrame(index=pd.Index(sym.index, name="hsapiens_gene_id"))
    tab["symbol"] = sym
    for c, ser in {"y_appr": out.y_appr, "y_appr_nononc": out.y_appr_nononc, "y_gen": yg, "y_gen_l2g": yg2, "y_gen_nononc": yg3}.items():
        tab[c] = tab.symbol.map(ser)
    tab["y_tgt"] = tab.symbol.isin(genes_clin).astype(float)
    # Therapy area labels: within the area's indications.
    for area, tissue in RU.AREA_TISSUE.items():
        sub = pp[pp.areas.map(lambda x: area in x)]
        if len(sub):
            tab[f"y_appr_{area}"] = tab.symbol.map(appr(sub))
    # Open Targets.
    ct = pd.concat([pd.read_parquet(f, columns=["targetId", "maxClinicalStage"]) for f in glob.glob(str(OT / "clinical_target/*.parquet"))])
    approved = set(ct.targetId[ct.maxClinicalStage.isin(["APPROVAL", "PHASE_4"])])
    trial = set(ct.targetId[ct.maxClinicalStage.isin(["PHASE_1", "PHASE_1_2", "PHASE_2", "PHASE_2_3", "PHASE_3", "EARLY_PHASE_1"])])
    tab["y_appr_ot"] = np.nan
    tab.loc[tab.index.isin(trial), "y_appr_ot"] = 0.0
    tab.loc[tab.index.isin(approved), "y_appr_ot"] = 1.0
    parts = []
    for f in glob.glob(str(OT / "association_by_datatype_direct/*.parquet")):
        t = pq.read_table(f, columns=["targetId", "aggregationValue", "associationScore"], filters=[("aggregationValue", "=", "genetic_association")]).to_pandas()
        parts.append(t)
    gen = pd.concat(parts).groupby("targetId").associationScore.max()
    clin_ot = approved | trial
    tab["y_gen_ot"] = np.nan
    ix = tab.index.isin(clin_ot)
    tab.loc[ix, "y_gen_ot"] = (tab.index[ix].map(gen).to_numpy(float) >= 0.5).astype(float)
    tab.to_parquet(OUT / "labels.parquet")
    print(tab.drop(columns="symbol").agg(["count", "sum"]).T.astype(int), flush=True)
    return tab


def covariates() -> dict:
    cov = {c: pd.read_parquet(MX / f"cov_{c}.parquet") for c in ["expr", "det", "tau", "log_cds_length", "prior_raw"]}
    a = pd.read_parquet(RU.RES / "Y2/annot/pheno.parquet")
    ext = pd.DataFrame(index=cov["expr"].index)
    ext["log_pubmed_h"], ext["log_pubmed_m"] = a.log_pubmed_h.reindex(ext.index), a.log_pubmed_m.reindex(ext.index)
    con = []
    for t in ["liver", "testis", "bone_marrow", "kidney"]:
        g = pd.read_parquet(RU.RES / f"Y2/genes_{t}.parquet")[["hsapiens_gene_id", "dNdS", "LOEUF"]].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
        con.append(g)
    con = pd.concat(con)
    con = con[~con.index.duplicated()]
    ext["dNdS"], ext["LOEUF"] = con.dNdS.reindex(ext.index), con.LOEUF.reindex(ext.index)
    ext["block"] = pd.read_parquet(MX / "blocks.parquet").block.reindex(ext.index)
    return dict(cov=cov, ext=ext)


def frame(tissue: str, base: dict, tab: pd.DataFrame, scores: dict) -> pd.DataFrame:
    g = base["ext"].copy()
    if tissue == "summary":
        for c, v in base["cov"].items():
            g[c] = v.mean(axis=1)
    else:
        for c, v in base["cov"].items():
            g[c] = v[tissue]
    for k, v in scores.items():
        g[k] = v if isinstance(v, pd.Series) else v[tissue]
    return g.join(tab.drop(columns="symbol"))


def rank_design(g, covs):
    out = {}
    for c in covs:
        out[f"r_{c}"] = g[c].rank(pct=True)
    return pd.DataFrame(out, index=g.index)


def run_unit(g, y, covs, scores, draws, rows, label, ref="S"):
    T = g[rows].copy()
    R = rank_design(T, covs)
    T = pd.concat([T, R], axis=1)
    rs = CM.rho_unit(T, y, [], scores, extra_linear=list(R.columns), draws=draws, ref=ref if ref in scores else None, label=label)
    return rs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab = labels()
    base = covariates()
    # Identity human-mouse, not a covariate in its own row.
    ident = pd.read_csv(RU.ROOT / "data/analysis/disease_kidney/ortholog_table_kidney.csv").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id").mmusculus_identity
    support = pd.read_parquet(MX / "support.parquet")
    Sraw, Draw, Braw = (pd.read_parquet(MX / f"raw_{k}.parquet") for k in ("S_fw", "D", "B1"))
    Zs, Zd, Zb = (pd.read_parquet(MX / f"z_{k}__ED.parquet") for k in ("S", "D", "B1"))
    n_sup = support.sum(axis=1)
    sm = {"S": Zs.where(support).mean(axis=1).where(n_sup >= 3), "D": Zd.where(support).mean(axis=1).where(n_sup >= 3), "B1": Zb.where(support).mean(axis=1).where(n_sup >= 3)}
    rng = np.random.default_rng(RU.SEED)
    units = []
    y_all = ["y_appr", "y_gen", "y_appr_ot", "y_gen_ot"]
    # Summary ladder and tissue rows.
    rows_out = []
    for tissue in ["summary"] + RU.TISSUES:
        scores = sm if tissue == "summary" else {"S": Sraw, "D": Draw, "B1": Braw}
        g = frame(tissue, base, tab, scores)
        g["identity"] = ident.reindex(g.index)
        g["random"] = rng.random(len(g))
        sup = (n_sup >= 3) if tissue == "summary" else support[tissue]
        for y in y_all:
            for lad, covs in LADDER.items():
                rows = (sup & g[y].notna()).reindex(g.index, fill_value=False)
                c2 = [c for c in covs]
                units.append(dict(g=g, y=y, covs=c2, scores=["S", "D", "B1", "random"], rows=rows, lad=lad, tissue=tissue))
                covs_id = [c for c in covs if c != "prior_raw"]
                units.append(dict(g=g, y=y, covs=covs_id, scores=["identity"], rows=rows, lad=lad, tissue=tissue))
    res = []
    for u in units:
        res += run_unit(u["g"], u["y"], u["covs"], u["scores"], RU.DRAWS_RHO if u["lad"] in ("C0", "C5") else 500, u["rows"], dict(tissue=u["tissue"], y=u["y"], ladder=u["lad"]))
    R = pd.DataFrame(res)
    # Primary family BH, block p from the bootstrap SE is not stored: use a normal approximation from the CI, as an exploratory q.
    p = R[(R.ladder == "C5") & R.y.isin(["y_appr", "y_gen"]) & R.score.isin(["S", "D"])].copy()
    se = (p.ci_hi - p.ci_lo) / 3.92
    from scipy.stats import norm

    p["p"] = 2 * norm.sf(np.abs(p.rho_p / se))
    p["q"] = multipletests(p.p, method="fdr_bh")[1]
    R = R.merge(p[["tissue", "y", "score", "ladder", "p", "q"]], how="left", on=["tissue", "y", "score", "ladder"])
    R.to_csv(OUT / "ladder.csv", index=False)

    # Sensitivities on the summary at C5.
    g = frame("summary", base, tab, sm)
    sup = n_sup >= 3
    # The ladder on one set of genes (those complete at C5), so the steps differ only in what is held; y_tgt beside it.
    same = []
    complete = sup & g[LADDER["C5"]].notna().all(axis=1)
    for y in ["y_appr", "y_gen", "y_appr_ot", "y_gen_ot", "y_tgt"]:
        for lad, covs in LADDER.items():
            same += run_unit(g, y, covs, ["S", "D", "B1"], 1000, complete & g[y].notna(), dict(y=y, ladder=lad))
    pd.DataFrame(same).to_csv(OUT / "ladder_same_genes.csv", index=False)
    cl = AN.gene_classes(pd.Index(g.index))
    cl = cl.reindex(g.index)
    class_cols = []
    for k in [c for c in DRUG_CLASSES if c in cl]:
        g[f"cl_{k}"] = cl[k].astype(float)
        class_cols.append(f"cl_{k}")
    sens = []
    for y, extra, tag in (("y_appr_nononc", [], "non-oncology"), ("y_gen_nononc", [], "non-oncology"), ("y_gen_l2g", [], "L2G share >= 0.5"), ("y_appr", class_cols, "with class"), ("y_gen", class_cols, "with class"), ("y_tgt", [], "whole vocabulary, no PubMed held")):
        covs = LADDER["C5"] if y != "y_tgt" else LADDER["C4"]
        rows = sup & g[y].notna()
        T = g[rows].copy()
        Rk = rank_design(T, covs)
        T = pd.concat([T, Rk], axis=1)
        sens += CM.rho_unit(T, y, [], ["S", "D", "B1"], extra_linear=list(Rk.columns) + extra, draws=RU.DRAWS_RHO, ref="S", label=dict(tag=tag))
    rows = sup & g.y_tgt.notna()
    T = g[rows].copy()
    Rk = rank_design(T, LADDER["C5"])
    T = pd.concat([T, Rk], axis=1)
    sens += CM.rho_unit(T, "y_tgt", [], ["S", "D", "B1"], extra_linear=list(Rk.columns), draws=RU.DRAWS_RHO, ref="S", label=dict(tag="whole vocabulary, PubMed held"))
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)

    # Tissue match.
    tm = []
    for area, tissue in RU.AREA_TISSUE.items():
        y = f"y_appr_{area}"
        for t2 in RU.TISSUES:
            gt = frame(t2, base, tab, {"S": Sraw, "D": Draw, "B1": Braw})
            rows = support[t2] & gt[y].notna()
            T = gt[rows].copy()
            if len(T) < 60:
                continue
            Rk = rank_design(T, LADDER["C5"])
            T = pd.concat([T, Rk], axis=1)
            tm += CM.rho_unit(T, y, [], ["S", "D", "B1"], extra_linear=list(Rk.columns), draws=1000, label=dict(area=area, matched=tissue, tissue=t2, n_pos=int(T[y].sum())))
    M_ = pd.DataFrame(tm)
    M_.to_csv(OUT / "tissue_match_raw.csv", index=False)
    out = []
    for (a, m, s), x in M_.groupby(["area", "matched", "score"]):
        own = x[x.tissue == m].rho_p
        oth = x[x.tissue != m].rho_p
        if len(own) and len(oth):
            out.append(dict(area=a, tissue=m, score=s, own=own.iloc[0], others=oth.mean(), delta=own.iloc[0] - oth.mean(), own_lo=x[x.tissue == m].ci_lo.iloc[0], own_hi=x[x.tissue == m].ci_hi.iloc[0], n=x[x.tissue == m].n.iloc[0], n_pos=x[x.tissue == m].n_pos.iloc[0]))
    pd.DataFrame(out).to_csv(OUT / "tissue_match.csv", index=False)


if __name__ == "__main__":
    main()
