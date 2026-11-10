"""Step 3: named genes (robust calls, rule of E1_atlas) for drug targets and disease genes per tissue, with mouse-model status,
replication in other cohorts, and the disease-cohort response (Q4, descriptive).

Writes results/analysis/E4/{names_targets.csv, names_disease.csv, q4_disease_response.csv, q4_fisher.csv}.

    PYTHONPATH=$PWD mamba run -n chimera_env python -m analysis.tasks.E4_translational.names
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from analysis.tasks.E4_translational import rules as RU

TAB = RU.OUT / "tables"
COLS = ["cohort", "tissue_e4", "role", "hsapiens_gene_id", "symbol", "mouse_symbol", "call", "call_can", "call_free", "int_can", "int_free", "S", "B1",
        "pi_raw", "pi_pct", "B1_pct", "expr", "tau", "det_human", "det_partner_min", "det_partner_max", "pubmed_h"]


def base() -> pd.DataFrame:
    T = pd.read_parquet(TAB / "cohorts.parquet")
    T["pi_pct"] = T.groupby("cohort").pi_raw.rank(pct=True)
    T["B1_pct"] = T.groupby("cohort").B1.rank(pct=True)
    return T


def replication(T: pd.DataFrame, ids: set, tissue: str) -> pd.DataFrame:
    """Calls of the same gene in the other cohorts of the same tissue (primate pairs) and in the other human-mouse thread tissues."""
    x = T[T.hsapiens_gene_id.isin(ids)]
    prim = x[(x.tissue_e4 == tissue) & (x.role == "primate")].groupby("hsapiens_gene_id").call.agg(
        lambda s: ",".join(str(int(v)) for v in s)).rename("call_primate_same_tissue")
    oth = x[(x.tissue_e4 != tissue) & (x.role == "thread")].groupby("hsapiens_gene_id").call.agg(
        n_other_below=lambda s: int((s == -1).sum()), n_other_above=lambda s: int((s == 1).sum()), n_other_tested="size")
    return pd.concat([prim, oth], axis=1)


def main() -> int:
    T = base()
    tg = pd.read_parquet(RU.OUT / "annot" / "targets.parquet").set_index("hsapiens_gene_id")
    dg = pd.read_parquet(RU.OUT / "annot" / "disease_genes.parquet").set_index("hsapiens_gene_id")
    sets = pd.read_parquet(RU.OUT / "annot" / "sets.parquet")
    hm = {**RU.HM_THREAD, **RU.HM_EXTRA}
    rows_t, rows_d = [], []
    for t, c in hm.items():
        u = T[T.cohort == c]
        # targets: approved in the universe, tissue-indicated flag
        a = u[u.hsapiens_gene_id.isin(tg.index[tg.approved])].copy()
        indcol = f"ind_{t}" if f"ind_{t}" in tg.columns else None
        a["indicated_here"] = a.hsapiens_gene_id.map(tg[indcol]) if indcol else False
        a["indicated_drugs"] = a.hsapiens_gene_id.map(tg[f"{indcol}_drugs"]) if indcol else ""
        a["approved_drugs"] = a.hsapiens_gene_id.map(tg.approved_drugs)
        a = a[a.call != 0]
        if len(a):
            a = a[COLS + ["indicated_here", "indicated_drugs", "approved_drugs"]].join(replication(T, set(a.hsapiens_gene_id), t), on="hsapiens_gene_id")
            rows_t.append(a)
        # disease genes: own-organ PanelApp set (and HPO for testis), with mouse-model status
        own = set(sets.hsapiens_gene_id[sets.set == f"own:{t}"])
        dos = set(sets.hsapiens_gene_id[sets.set == f"do:{t}"])
        d = u[u.hsapiens_gene_id.isin(own | dos) & (u.call != 0)].copy()
        if len(d):
            d["in_panel"] = d.hsapiens_gene_id.isin(own)
            d["in_do_organ"] = d.hsapiens_gene_id.isin(dos)
            d["mouse_tested"] = d.hsapiens_gene_id.map(dg.mouse_tested)
            d["mouse_model_same_disease"] = d.hsapiens_gene_id.map(dg.Y_mgi)
            d["mouse_not_model"] = d.hsapiens_gene_id.map(dg.Y_not)
            d["approved_target"] = d.hsapiens_gene_id.map(tg.approved).fillna(False)
            d = d[COLS + ["in_panel", "in_do_organ", "mouse_tested", "mouse_model_same_disease", "mouse_not_model", "approved_target"]].join(
                replication(T, set(d.hsapiens_gene_id), t), on="hsapiens_gene_id")
            rows_d.append(d)
    NT = pd.concat(rows_t, ignore_index=True)
    ND = pd.concat(rows_d, ignore_index=True)
    NT = NT[~NT.symbol.astype(str).str.startswith("MT-")]
    ND = ND[~ND.symbol.astype(str).str.startswith("MT-")]
    NT.sort_values(["tissue_e4", "indicated_here", "call", "int_free"], ascending=[True, False, True, True]).to_csv(RU.OUT / "names_targets.csv", index=False)
    ND.sort_values(["tissue_e4", "call", "int_free"]).to_csv(RU.OUT / "names_disease.csv", index=False)
    print(NT.groupby(["tissue_e4", "call"]).size().unstack(fill_value=0))
    print(ND.groupby(["tissue_e4", "call"]).size().unstack(fill_value=0))
    q4(T, tg, sets)
    return 0


def read_de(path, human: bool) -> pd.DataFrame:
    d = pd.read_csv(RU.RES / path, sep="\t")
    idc = "gene_id" if "gene_id" in d.columns else d.columns[0]
    lfc = "log2fc_shrunk" if "log2fc_shrunk" in d.columns else ("log2fc_fixed" if "log2fc_fixed" in d.columns else "log2FoldChange")
    pa = "padj" if "padj" in d.columns else "FDR"
    return d.rename(columns={idc: "gid", lfc: "lfc", pa: "padj"})[["gid", "lfc", "padj"]].dropna(subset=["gid"])


def q4(T, tg, sets) -> None:
    g = pd.read_parquet(RU.OUT / "annot" / "genes.parquet")
    ph = pd.read_parquet(RU.REUSED["pheno"]).reset_index()[["hsapiens_gene_id", "mouse_ens"]].dropna()
    ok = set(g.index[g.mouse_1to1_agree])
    m2h = dict(zip(ph.mouse_ens, ph.hsapiens_gene_id))
    rows, fis = [], []
    for t, spec in RU.DISEASE_DE.items():
        c = RU.HM_THREAD[t]
        u = T[T.cohort == c].set_index("hsapiens_gene_id")
        own = set(sets.hsapiens_gene_id[sets.set == f"own:{t}"]) | set(sets.hsapiens_gene_id[sets.set == f"do:{t}"])
        ind = set(tg.index[tg.get(f"ind_{t}", pd.Series(False, index=tg.index))]) | set(tg.index[tg.approved])
        for ty in spec["types"]:
            try:
                h = read_de(spec["human"].format(t=ty), True)
                m = read_de(spec["mouse"].format(t=ty), False)
            except FileNotFoundError:
                continue
            m["gid"] = m.gid.map(m2h)
            m = m.dropna(subset=["gid"]).drop_duplicates("gid").set_index("gid")
            h = h.drop_duplicates("gid").set_index("gid")
            j = h.join(m, lsuffix="_h", rsuffix="_m", how="inner")
            j = j[j.index.isin(ok) & j.index.isin(u.index)]
            j["human_de"] = j.padj_h < RU.PADJ
            j["transfer"] = j.human_de & (np.sign(j.lfc_h) == np.sign(j.lfc_m)) & (j.padj_m < 0.5)
            j["call"] = u.call.reindex(j.index)
            j["symbol"] = u.symbol.reindex(j.index)
            j["int_free"], j["int_can"], j["pi_raw"] = u.int_free.reindex(j.index), u.int_can.reindex(j.index), u.pi_raw.reindex(j.index)
            j["disease_gene"], j["target"] = j.index.isin(own), j.index.isin(ind)
            j["tissue"], j["type"] = t, ty
            sel = j[(j.disease_gene | j.target) & j.human_de]
            rows.append(sel.reset_index().rename(columns={"index": "hsapiens_gene_id", "gid": "hsapiens_gene_id"}))
            for name, sub in [("disease_or_target", sel), ("all_human_de", j[j.human_de])]:
                a = int(((sub.call == -1) & sub.transfer).sum()); b = int(((sub.call == -1) & ~sub.transfer).sum())
                cc = int(((sub.call != -1) & sub.transfer).sum()); dd = int(((sub.call != -1) & ~sub.transfer).sum())
                orr, p = fisher_exact([[a, b], [cc, dd]]) if min(a + b, cc + dd) > 0 else (np.nan, np.nan)
                fis.append(dict(tissue=t, type=ty, genes=name, below_transfer=a, below_not=b, rest_transfer=cc, rest_not=dd, OR=orr, p=p))
    pd.concat(rows, ignore_index=True).to_csv(RU.OUT / "q4_disease_response.csv", index=False)
    F = pd.DataFrame(fis)
    F.to_csv(RU.OUT / "q4_fisher.csv", index=False)
    print(F.to_string())


if __name__ == "__main__":
    raise SystemExit(main())
