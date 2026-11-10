"""E4 step 1: gene annotations from the downloads (definitions in rules.py). No read-out is touched here.

Writes results/analysis/E4/annot/{genes.parquet, sets.parquet, disease_genes.parquet, target_drugs.parquet, checks.txt}.
Downloads are read as data only; run with python -I through the module path:

    PYTHONPATH=$PWD mamba run -n chimera_env python -m analysis.tasks.E4_translational.annot
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from analysis.tasks.E4_translational import rules as RU

OUTA = RU.OUT / "annot"
D = RU.DATA


def obo_parents(path, rels=("is_a",)) -> tuple[dict, dict]:
    par, name = defaultdict(set), {}
    cur, obs = None, False
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if ln == "[Term]":
            cur, obs = None, False
        elif ln.startswith("id: "):
            cur = ln[4:]
        elif ln.startswith("name: ") and cur:
            name[cur] = ln[6:]
        elif ln.startswith("is_obsolete: true"):
            obs = True
        elif cur and ln.startswith("is_a: ") and "is_a" in rels:
            par[cur].add(ln[6:].split(" ")[0])
        elif cur and ln.startswith("relationship: "):
            r = ln.split(" ")
            if r[1] in rels:
                par[cur].add(r[2])
    return par, name


def closure(par: dict) -> dict:
    memo = {}

    def anc(t):
        if t in memo:
            return memo[t]
        memo[t] = set()
        out = set()
        for p in par.get(t, ()):
            out.add(p)
            out |= anc(p)
        memo[t] = out
        return out

    return {t: anc(t) for t in list(par)}


def split_ids(s) -> list[str]:
    if not isinstance(s, str) or not s:
        return []
    return [x.strip() for x in s.replace(",", "|").split("|") if x.strip()]


def gene_base() -> pd.DataFrame:
    """Human Ensembl -> Entrez, Y2's mouse MGI and PubMed counts; MGI one-to-one homology must agree."""
    ph = pd.read_parquet(RU.REUSED["pheno"]).reset_index()
    base = ph[["hsapiens_gene_id", "ncbi", "mgi", "mouse_ens", "pubmed_h", "pubmed_m"]].copy()
    hom = pd.read_csv(D / "mgi" / "HOM_MouseHumanSequence.rpt", sep="\t", dtype=str)
    h = hom[hom["NCBI Taxon ID"] == "9606"]
    m = hom[hom["NCBI Taxon ID"] == "10090"]
    nh, nm = h.groupby("DB Class Key").size(), m.groupby("DB Class Key").size()
    one = nh.index[(nh == 1)].intersection(nm.index[nm == 1])
    mp = h[h["DB Class Key"].isin(one)][["DB Class Key", "EntrezGene ID", "Symbol"]].merge(
        m[m["DB Class Key"].isin(one)][["DB Class Key", "Mouse MGI ID", "Symbol"]], on="DB Class Key", suffixes=("_h", "_m"))
    mp = mp.rename(columns={"EntrezGene ID": "ncbi_s", "Mouse MGI ID": "mgi_hom", "Symbol_h": "symbol_mgi_h", "Symbol_m": "mouse_symbol"})
    base["ncbi_s"] = base.ncbi.dropna().astype(int).astype(str).reindex(base.index)
    base = base.merge(mp[["ncbi_s", "mgi_hom", "symbol_mgi_h", "mouse_symbol"]], on="ncbi_s", how="left")
    base["mouse_1to1_agree"] = base.mgi_hom.notna() & (base.mgi_hom == base.mgi)
    return base


def mgi_disease(base: pd.DataFrame, doanc: dict, checks: list) -> pd.DataFrame:
    do = pd.read_csv(D / "mgi" / "MGI_DO.rpt", sep="\t", dtype=str)
    hu = do[do["NCBI Taxon ID"] == "9606"]
    dis = hu.groupby("EntrezGene ID").agg(doids=("DO Disease ID", lambda s: set(s)),
                                          omims=("OMIM IDs", lambda s: set(x for v in s.dropna() for x in split_ids(v))))
    tested = set()
    gp = pd.read_csv(D / "mgi" / "MGI_GenePheno.rpt", sep="\t", header=None, dtype=str)
    for v in gp[6].dropna():
        tested.update(split_ids(v))
    def model_table(fn):
        g = pd.read_csv(D / "mgi" / fn, sep="\t", header=None, dtype=str)
        g.columns = ["genotype", "alleles", "allele_ids", "background", "mp", "pmid", "markers", "doid", "omim", "genotype_id"][: g.shape[1]]
        g = g.drop_duplicates(["genotype_id", "markers", "doid", "omim", "alleles"])
        g["mk"] = g.markers.map(split_ids)
        g["single"] = (g.mk.map(len) == 1) & ~g.alleles.fillna("").str.contains(r"Tg\(", regex=True)
        rows = []
        for r in g.itertuples():
            for mk in r.mk:
                rows.append((mk, r.doid, r.omim, r.single, r.genotype_id))
        return pd.DataFrame(rows, columns=["mgi", "doid", "omim", "single", "genotype_id"])
    pos, neg = model_table("MGI_Geno_DiseaseDO.rpt"), model_table("MGI_Geno_NotDiseaseDO.rpt")
    pos_by = pos.groupby("mgi")
    neg_by = neg.groupby("mgi")
    out = []
    b = base[base.mouse_1to1_agree].drop_duplicates("hsapiens_gene_id")
    for r in b.itertuples():
        e = r.ncbi_s
        if e not in dis.index:
            continue
        dd, oo = dis.loc[e, "doids"], dis.loc[e, "omims"]
        y = ys = yn = False
        if r.mgi in pos_by.groups:
            x = pos_by.get_group(r.mgi)
            hit = x.doid.isin(dd) | x.omim.fillna("").map(lambda v: bool(set(split_ids(v)) & oo))
            y, ys = bool(hit.any()), bool((hit & x.single).any())
        if r.mgi in neg_by.groups:
            x = neg_by.get_group(r.mgi)
            yn = bool((x.doid.isin(dd) | x.omim.fillna("").map(lambda v: bool(set(split_ids(v)) & oo))).any())
        anc = set().union(*[doanc.get(t, set()) | {t} for t in dd])
        out.append(dict(hsapiens_gene_id=r.hsapiens_gene_id, n_disease=len(dd), mouse_tested=r.mgi in tested, Y_mgi=y, Y_mgi_single=ys,
                        Y_not=(yn and not y), **{f"organ_{t}": bool(anc & set(RU.ORGAN_DO[t])) for t in RU.ORGAN_DO}))
    out = pd.DataFrame(out)
    checks.append(f"MGI_DO human disease genes (1:1 mouse agreeing): {len(out)}; mouse phenotyped {int(out.mouse_tested.sum())}; "
                  f"Y_mgi {int(out.Y_mgi.sum())}; Y_mgi_single {int(out.Y_mgi_single.sum())}; Y_not {int(out.Y_not.sum())}")
    return out


def ot_tables(checks: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    OT = D / "opentargets"
    dis = pd.read_parquet(OT / "disease" / "00000000.parquet", columns=["id", "name", "descendants"])
    desc = {i: set(d) if d is not None else set() for i, d in zip(dis.id, dis.descendants)}
    nm = dict(zip(dis.id, dis.name))
    for t, ids in RU.TA_OT.items():
        for i, n in ids.items():
            assert nm.get(i, "").lower() == n.lower(), (i, nm.get(i), n)
    onc = set().union(*[desc[i] | {i} for i in RU.ONCOLOGY_OT])
    ta = {t: set().union(*[desc[i] | {i} for i in ids]) - onc for t, ids in RU.TA_OT.items()}
    ct = pd.read_parquet(OT / "clinical_target" / "00000000.parquet")
    ci_path = OT / "clinical_indication" / "00000000.parquet"
    drugs = pd.read_parquet(OT / "drug_molecule" / "00000000.parquet", columns=["id", "name"])
    dname = dict(zip(drugs.id, drugs.name))
    rows = []
    for r in ct.itertuples():
        ds = [x["diseaseId"] for x in (r.diseases if r.diseases is not None else []) if x is not None and x.get("diseaseId")]
        rows.append((r.targetId, r.drugId, r.maxClinicalStage, ds))
    T = pd.DataFrame(rows, columns=["hsapiens_gene_id", "drug", "stage", "diseases"])
    # per-indication stage where clinical_indication exists: approved indications only count for tissue indication
    appr_ind = None
    if ci_path.exists():
        ci = pd.read_parquet(ci_path)
        checks.append(f"clinical_indication columns: {list(ci.columns)}")
        dcol = next((c for c in ("drugId", "drug_id", "chemblId") if c in ci.columns), None)
        scol = next((c for c in ("maxClinicalStage", "clinicalStage", "phase") if c in ci.columns), None)
        icol = next((c for c in ("diseaseId", "indicationId", "efoId") if c in ci.columns), None)
        if dcol and scol and icol:
            a = ci[ci[scol].isin(RU.APPROVED_STAGES)]
            appr_ind = a.groupby(dcol)[icol].agg(lambda s: set(s)).to_dict()
    tgt = []
    for g, x in T.groupby("hsapiens_gene_id"):
        st = set(x.stage)
        appr = bool(st & RU.APPROVED_STAGES)
        clin = (not appr) and bool(st & RU.CLINICAL_STAGES)
        d = dict(hsapiens_gene_id=g, approved=appr, clinical_only=clin, withdrawn_only=st <= {"WITHDRAWAL"}, any_clinical=appr or bool(st & RU.CLINICAL_STAGES))
        xa = x[x.stage.isin(RU.APPROVED_STAGES)]
        for t, S in ta.items():
            hit = []
            for r in xa.itertuples():
                ind = appr_ind.get(r.drug, set()) if appr_ind is not None else set(r.diseases)
                if ind & S:
                    hit.append(r.drug)
            d[f"ind_{t}"] = bool(hit)
            d[f"ind_{t}_drugs"] = ";".join(sorted({dname.get(h, h) for h in hit})[:6])
        d["approved_drugs"] = ";".join(sorted({dname.get(h, h) for h in xa.drug})[:6])
        tgt.append(d)
    tgt = pd.DataFrame(tgt)
    checks.append(f"Open Targets targets: {len(tgt)}; approved {int(tgt.approved.sum())}; clinical only {int(tgt.clinical_only.sum())}; "
                  + "; ".join(f"ind_{t} {int(tgt[f'ind_{t}'].sum())}" for t in ta))
    # Mendelian diseases and animal-model evidence
    men = []
    for src in ("evidence_orphanet", "evidence_gene2phenotype", "evidence_clingen", "evidence_genomics_england"):
        fs = sorted(glob.glob(str(OT / src / "*.parquet")))
        if not fs:
            checks.append(f"missing {src}")
            continue
        e = pd.concat([pd.read_parquet(f, columns=["targetId", "diseaseId", "confidence"]) for f in fs])
        checks.append(f"{src}: rows {len(e)}, confidence {e.confidence.value_counts().head(8).to_dict()}")
        keep = ~e.confidence.astype(str).str.lower().isin({"limited", "disputed", "refuted", "no known disease relationship", "amber", "red", "low"})
        men.append(e[keep][["targetId", "diseaseId"]])
    men = pd.concat(men).drop_duplicates() if men else pd.DataFrame(columns=["targetId", "diseaseId"])
    fs = sorted(glob.glob(str(OT / "evidence_impc" / "*.parquet")))
    imp = []
    if fs:
        for f in fs:
            x = pd.read_parquet(f, columns=["targetId", "diseaseId", "score"])
            imp.append(x.groupby(["targetId", "diseaseId"], as_index=False).score.max())
        imp = pd.concat(imp).groupby(["targetId", "diseaseId"], as_index=False).score.max()
    else:
        imp = pd.DataFrame(columns=["targetId", "diseaseId", "score"])
        checks.append("missing evidence_impc")
    mset = men.groupby("targetId").diseaseId.agg(set)
    iset = imp.groupby("targetId").diseaseId.agg(set)
    rows = []
    for g in mset.index:
        if g not in iset.index:
            continue
        rows.append(dict(hsapiens_gene_id=g, n_mendelian=len(mset[g]), Y_impc=bool(mset[g] & iset[g])))
    ip = pd.DataFrame(rows)
    checks.append(f"IMPC/Phenodigm: Mendelian targets {len(mset)}, with animal-model rows {len(ip)}, own disease matched {int(ip.Y_impc.sum()) if len(ip) else 0}")
    return tgt, ip


def panel_sets(checks: list) -> pd.DataFrame:
    rows = []
    allp = {**{(t, k): v for t, d in RU.PANELS.items() for k, v in d.items()}, **{(t, k): v for t, d in RU.EXTRA_PANELS.items() for k, v in d.items()}}
    for (t, pid), name in allp.items():
        js = json.load(open(D / "panelapp" / f"panel_{pid}.json"))
        assert js["name"].strip().lower() == name.strip().lower(), (pid, js["name"], name)
        n = 0
        for g in js["genes"]:
            if str(g.get("confidence_level")) != "3":
                continue
            ens = (g["gene_data"].get("ensembl_genes") or {}).get("GRch38") or {}
            ids = {v.get("ensembl_id") for v in ens.values() if isinstance(v, dict)}
            for e in ids:
                if e:
                    rows.append(("panelapp", t, f"panel:{pid}:{name}", e, g["gene_data"].get("hgnc_symbol")))
                    n += 1
        checks.append(f"PanelApp {pid} {name} v{js['version']}: green {n}")
    P = pd.DataFrame(rows, columns=["source", "tissue", "set", "hsapiens_gene_id", "symbol"])
    own = P[P.apply(lambda r: int(r.set.split(":")[1]) in RU.PANELS.get(r.tissue, {}), axis=1)].copy()
    own["set"] = "own:" + own.tissue
    return pd.concat([P, own.drop_duplicates(["tissue", "hsapiens_gene_id"])], ignore_index=True)


def hpo_male_infertility(base: pd.DataFrame, checks: list) -> pd.DataFrame:
    par, _ = obo_parents(RU.REUSED["hp_obo"])
    anc = closure(par)
    keep = {t for t, a in anc.items() if RU.HPO_MALE_INFERTILITY in a} | {RU.HPO_MALE_INFERTILITY}
    g = pd.read_csv(RU.REUSED["hpo_g2p"], sep="\t", dtype=str)
    e = set(g[g.hpo_id.isin(keep)].ncbi_gene_id.dropna())
    b = base.dropna(subset=["ncbi_s"])
    ids = b.hsapiens_gene_id[b.ncbi_s.isin(e)].unique()
    checks.append(f"HPO male infertility terms {len(keep)}, genes {len(ids)}")
    return pd.DataFrame(dict(source="hpo", tissue="testis", set="hpo:male infertility", hsapiens_gene_id=ids, symbol=None))


def main() -> int:
    OUTA.mkdir(parents=True, exist_ok=True)
    checks = []
    par, nm = obo_parents(D / "do" / "doid.obo")
    for t, ids in RU.ORGAN_DO.items():
        for i, n in ids.items():
            assert nm.get(i) == n, (i, nm.get(i), n)
    doanc = closure(par)
    base = gene_base()
    checks.append(f"genes in Y2 base {base.hsapiens_gene_id.nunique()}; MGI 1:1 agreeing {int(base.mouse_1to1_agree.sum())}")
    dg = mgi_disease(base, doanc, checks)
    dg.to_parquet(OUTA / "disease_genes.parquet")
    tgt, ip = ot_tables(checks)
    tgt.to_parquet(OUTA / "targets.parquet")
    ip.to_parquet(OUTA / "impc.parquet")
    P = panel_sets(checks)
    H = hpo_male_infertility(base, checks)
    own_t = H.assign(set="own:testis")
    # DO organ sets (second source)
    DO = []
    for t in RU.ORGAN_DO:
        ids = dg.hsapiens_gene_id[dg[f"organ_{t}"]]
        DO.append(pd.DataFrame(dict(source="do", tissue=t, set=f"do:{t}", hsapiens_gene_id=ids, symbol=None)))
    TG = []
    for col in ["approved", "clinical_only"] + [f"ind_{t}" for t in RU.TA_OT]:
        ids = tgt.hsapiens_gene_id[tgt[col]]
        TG.append(pd.DataFrame(dict(source="target", tissue=col.replace("ind_", "") if col.startswith("ind_") else "all",
                                    set=f"target:{col}", hsapiens_gene_id=ids, symbol=None)))
    S = pd.concat([P, H, own_t, *DO, *TG], ignore_index=True).drop_duplicates(["set", "hsapiens_gene_id"])
    S.to_parquet(OUTA / "sets.parquet")
    g = base.drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")[["ncbi_s", "mgi", "mouse_symbol", "mouse_1to1_agree", "pubmed_h", "pubmed_m"]]
    g.to_parquet(OUTA / "genes.parquet")
    checks.append("sets: " + "; ".join(f"{k} {v}" for k, v in S.groupby("set").size().items()))
    (OUTA / "checks.txt").write_text("\n".join(checks) + "\n")
    print("\n".join(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
