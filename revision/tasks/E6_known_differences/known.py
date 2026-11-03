"""E6 analysis: literature species-difference lists against Chimera readouts (definitions in rules.py).

Usage: python -I known.py [assign_id ...]      (default: every row of assign.tsv)
Outputs in results/revision/E6/: mapping.tsv, auroc.tsv, genes.tsv (every list gene with its within-stratum
percentiles and naming flags), named.tsv.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata as _rank

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules as RU  # noqa: E402

TASK = Path(__file__).resolve().parent
NAME_KEYS = {"human": "human", "mouse": r"^detection_mouse$|_mouse$|^detection_Mouse$", "macaque": "macaque|monkey|mulatta|fascicularis",
             "chimpanzee": "chimp", "marmoset": "marmoset", "lemur": "lemur", "pig": "pig", "zebrafish": "zebrafish"}

# ---------------------------------------------------------------- symbol mapping
_ORTHO_TABLES = [
    "data/processed/ortholog_table.csv", "data/processed/ortholog_table_testis.csv",
    "data/revision/bone_marrow/ortholog_table_bone_marrow.csv", "data/revision/disease_kidney/ortholog_table_kidney.csv",
    "data/revision/van_zyl/ortholog_table_van_zyl.csv",
] + [f"data/revision/new_usecases/c1_build/objects/ortholog_table_{c}.csv" for c in
     ["brain_dlpfc", "brain_m1", "brain_m1_hmouse", "lung_healthy", "muscle_ageing", "tumour_zilionis"]]
SP_COL = {"mouse": "mmusculus", "macaque": "mmulatta", "chimpanzee": "ptroglodytes", "marmoset": "cjacchus", "lemur": "mmurinus"}


def hgnc_maps():
    h = pd.read_csv(RU.HGNC, sep="\t", dtype=str, usecols=["symbol", "status", "alias_symbol", "prev_symbol", "ensembl_gene_id"])
    h = h[h.ensembl_gene_id.notna()]
    approved = dict(zip(h.symbol.str.upper(), h.ensembl_gene_id))
    other = {}
    for col in ["prev_symbol", "alias_symbol"]:
        for s, e in zip(h[col], h.ensembl_gene_id):
            if isinstance(s, str):
                for x in s.split("|"):
                    other.setdefault(x.strip().upper(), set()).add(e)
    other = {k: next(iter(v)) for k, v in other.items() if len(v) == 1 and k not in approved}
    return approved, other


def species_maps():
    m = {}
    for t in _ORTHO_TABLES:
        p = RU.REPO / t
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype=str)
        for sp, pre in SP_COL.items():
            nc = f"{pre}_gene_name"
            if nc in df.columns:
                for n, e in zip(df[nc], df.hsapiens_gene_id):
                    if isinstance(n, str) and isinstance(e, str):
                        m.setdefault(sp, {}).setdefault(n.upper(), set()).add(e)
    return {sp: {k: next(iter(v)) for k, v in d.items() if len(v) == 1} for sp, d in m.items()}


def map_symbols(lst, approved, other, spmaps):
    out, how = [], []
    for s, sp in zip(lst.gene_symbol.astype(str), lst.symbol_species.astype(str).str.lower()):
        sp = next((k for k in SP_COL if k in sp), "human")  # "mouse_uppercased", "mouse-style symbol ..." -> mouse
        u = s.strip().upper()
        e, h = None, "unmapped"
        if u.startswith("ENSG"):
            e, h = u.split(".")[0], "ensembl_id"
        elif sp in SP_COL and u in spmaps.get(sp, {}):
            e, h = spmaps[sp][u], f"{sp}_1to1_table"
        elif u in approved:
            e, h = approved[u], "hgnc_approved" if sp == "human" else "hgnc_approved_case"
        elif u in other:
            e, h = other[u], "hgnc_prev_alias" if sp == "human" else "hgnc_prev_alias_case"
        out.append(e)
        how.append(h)
    lst = lst.copy()
    lst["hsapiens_gene_id"], lst["map_how"] = out, how
    return lst


def read_list(a):
    f = RU.LISTS / f"{a['source_id']}.tsv"
    lst = pd.read_csv(f if f.exists() else Path(str(f) + ".gz"), sep="\t", dtype=str).fillna("")
    if a["sets"]:
        lst = lst[lst.set.isin(a["sets"].split(","))]
    if a.get("ct_regex"):
        lst = lst[lst.cell_type.str.contains(a["ct_regex"], regex=True)]
    if a.get("contrast_regex"):
        lst = lst[lst.contrast.str.contains(a["contrast_regex"], regex=True)]
    return lst


_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
NO_CORE = {"shami2020"}  # cluster labels only, no gene-level strength: full sets are the reading


def _first_float(x):
    """Strength of a row: max |log fold change| over fields named log_fold/log2FC/lnFC/logFC; |ln FC| for plain FC fields;
    else the first number after '='."""
    vals = []
    for k, v in re.findall(r"([^=;]+)=\s*(" + _NUM + ")", str(x)):
        k = k.strip()
        if re.search(r"log_fold|log2FC|lnFC|logFC", k):
            vals.append(abs(float(v)))
        elif re.match(r"FC\(", k) and float(v) > 0:
            vals.append(abs(np.log(float(v))))
    if vals:
        return max(vals)
    m = re.search(_NUM, str(x).split("=", 1)[-1])
    return abs(float(m.group(0))) if m else 0.0


def add_core(lst):
    """rules: a set with more than CORE_N unique mapped genes also enters as <set>__core: the CORE_N genes called in the most
    source cell types, ties by the largest |effect| (first number after '=' in the effect field), then by symbol. Hodge 2019 has one
    gene-level score and no cell types: its core is the paper's own top-3% tier (note field `tier=top3pct`)."""
    extra = []
    for st, sub in lst.dropna(subset=["hsapiens_gene_id"]).groupby("set"):
        if "conserved" in st or sub.hsapiens_gene_id.nunique() <= RU.CORE_N or (sub.source_id.isin(NO_CORE)).all():
            continue
        if (sub.source_id == "hodge2019").all():
            core = sub[sub.note.str.contains("tier=top3pct")]
        else:
            agg = sub.assign(eff=sub.effect.map(_first_float)).groupby("hsapiens_gene_id").agg(
                n_ct=("cell_type", "nunique"), eff=("eff", "max"), sym=("gene_symbol", "first"))
            keep = agg.sort_values(["n_ct", "eff", "sym"], ascending=[False, False, True]).head(RU.CORE_N).index
            core = sub[sub.hsapiens_gene_id.isin(keep)]
        extra.append(core.assign(set=st + "__core"))
    return pd.concat([lst] + extra, ignore_index=True) if extra else lst


# ---------------------------------------------------------------- cohort table
def det_col(cols, species):
    pat = re.compile(NAME_KEYS[species], re.I)
    return [c for c in cols if c.startswith("detection_") and pat.search(c) and not (species == "mouse" and "lemur" in c.lower())]


def load_cohort(c):
    h = pd.read_parquet(RU.H1_GENES / f"{c}.parquet")
    z = pd.read_parquet(RU.Z4_GENES / f"{c}.parquet", columns=["hsapiens_gene_id", "D", "S_ema", "count_support",
                                                                 "sign_D_stable", "sign_S_stable", "saturated_flag", "nameable"])
    z = z.rename(columns={k: f"{k}_free" for k in z.columns if k != "hsapiens_gene_id"})
    g = h.merge(z, on="hsapiens_gene_id", how="left")
    g = g.rename(columns={"D": "D_can", "S_ema": "S_can", "S_ema_free": "S_free"})
    g["supp_both"] = g.count_support.astype(bool) & g.count_support_free.fillna(False).astype(bool)
    return g


def add_lvl(g, other_species):
    """-|logit det_human - logit det_other|, max gap over the other species named in the contrast and present."""
    cols = list(g.columns)
    hc = det_col(cols, "human")
    oc = sorted({x for sp in other_species if sp in NAME_KEYS for x in det_col(cols, sp)})
    if not hc or not oc:
        g["LVL"] = np.nan
        return g, None
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
    gaps = np.column_stack([np.abs(lg(g[hc[0]].to_numpy(float)) - lg(g[o].to_numpy(float))) for o in oc])
    g["LVL"] = -gaps.max(1)
    return g, f"{hc[0]} vs {','.join(oc)}"


def strata(g, tau=True):
    e = pd.qcut(g.expr.rank(method="first"), RU.Q_EXPR, labels=False).to_numpy()
    p = pd.qcut(g.pi_raw.rank(method="first"), RU.Q_PI, labels=False).to_numpy()
    s = e * 100 + p
    if tau:
        t = pd.qcut(g.tau.rank(method="first"), RU.Q_TAU, labels=False).to_numpy()
        s = s * 10 + t
    return s


# ---------------------------------------------------------------- stratified AUROC
def strat_auroc(score, member, strat, rng, n_boot=RU.N_BOOT, n_perm=RU.N_PERM):
    """List-weighted stratified AUROC, bootstrap CI, one-sided lower permutation p. score NaN rows dropped."""
    ok = np.isfinite(score)
    score, member, strat = score[ok], member[ok], strat[ok]
    pre = []
    for s in np.unique(strat):
        ix = np.where(strat == s)[0]
        m = member[ix]
        n1, n0 = int(m.sum()), int((~m).sum())
        if n1 >= 1 and n0 >= RU.MIN_STRATUM_BG:
            x = score[ix]
            pre.append((x[m], x[~m], _rank(x), m, n1, n0))
    if not pre:
        return dict(auroc=np.nan, lo=np.nan, hi=np.nan, p_lower=np.nan, n_list=0, n_bg=0, n_strata=0)
    w = np.array([b[4] for b in pre], float)
    W = w.sum()
    obs = sum(wb * (r[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) for (_, _, r, m, n1, n0), wb in zip(pre, w)) / W
    boots = np.empty(n_boot)
    for i in range(n_boot):
        acc = 0.0
        for (x1, x0, _, _, n1, n0), wb in zip(pre, w):
            r = _rank(np.concatenate([x1[rng.integers(0, n1, n1)], x0[rng.integers(0, n0, n0)]]))
            acc += wb * (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        boots[i] = acc / W
    perms = np.empty(n_perm)
    for i in range(n_perm):
        acc = 0.0
        for (_, _, r, _, n1, n0), wb in zip(pre, w):
            acc += wb * (r[rng.choice(len(r), n1, replace=False)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        perms[i] = acc / W
    return dict(auroc=float(obs), lo=float(np.quantile(boots, 0.025)), hi=float(np.quantile(boots, 0.975)),
                p_lower=float((1 + (perms <= obs).sum()) / (n_perm + 1)), n_list=int(W),
                n_bg=int(sum(b[5] for b in pre)), n_strata=len(pre))


def within_stratum_pct(g, col, strat):
    return pd.Series(g[col].to_numpy(float)).groupby(strat).rank(pct=True).to_numpy()


# ---------------------------------------------------------------- one (assignment, cohort)
_CTX = {}


def _init():
    _CTX["approved"], _CTX["other"] = hgnc_maps()
    _CTX["spmaps"] = species_maps()


def _tru(v):
    return bool(v) if pd.notna(v) else False


def run_one(job):
    a, c, k = job
    rng = np.random.default_rng(RU.SEED + k)
    rows, genes = [], []
    lst = add_core(map_symbols(read_list(a), _CTX["approved"], _CTX["other"], _CTX["spmaps"]))
    g, lvl_desc = add_lvl(load_cohort(c), a["other_species"].split(","))
    universe_all = g.D_can.notna().to_numpy()
    universe_sup = (g.supp_both & g.B1.notna()).to_numpy()
    st_tau = strata(g, True)
    ctrl_all = set(lst[lst.set.str.contains("conserved")].hsapiens_gene_id.dropna())
    for setname, sub in lst.groupby("set"):
        ids = set(sub.hsapiens_gene_id.dropna())
        member = g.hsapiens_gene_id.isin(ids).to_numpy()
        ctrl_ids = ctrl_all if "conserved" not in setname else set()
        for uname, uni in [("supported", universe_sup), ("all", universe_all)]:
            gu = g[uni].reset_index(drop=True)
            mu = member[uni]
            if mu.sum() < RU.MIN_LIST or (~mu).sum() < 50:
                continue
            stt, stn = strata(gu, True), strata(gu, False)
            scores = RU.SCORES if uname == "supported" else ["D_can", "pi_raw"]
            base = dict(assign_id=a["assign_id"], source_id=a["source_id"], circular=a["circular"], primary=a["primary"],
                        cohort=c, set=setname, universe=uname, lvl=lvl_desc)
            for sc in scores:
                x = gu[sc].to_numpy(float)
                if not np.isfinite(x).any():
                    continue
                for strat_name, stv in [("expr_tau_pi", stt), ("expr_pi", stn)]:
                    rows.append(dict(base, score=sc, strata=strat_name, comparison="vs_matched", **strat_auroc(x, mu, stv, rng)))
                if ctrl_ids and uname == "supported":
                    gb = gu[gu.hsapiens_gene_id.isin(ids | ctrl_ids).to_numpy()].reset_index(drop=True)
                    mb = gb.hsapiens_gene_id.isin(ids - ctrl_ids).to_numpy()
                    if mb.sum() >= RU.MIN_LIST and (~mb).sum() >= RU.MIN_LIST:
                        stb = (pd.qcut(gb.expr.rank(method="first"), 3, labels=False).to_numpy() * 10
                               + pd.qcut(gb.tau.rank(method="first"), 3, labels=False).to_numpy())
                        # control strata need >= 1 control gene; MIN_STRATUM_BG relaxed to 1 by passing through a copy of the rule
                        old = RU.MIN_STRATUM_BG
                        RU.MIN_STRATUM_BG = 1
                        rows.append(dict(base, score=sc, strata="expr3_tau3", comparison="vs_conserved_control",
                                         **strat_auroc(gb[sc].to_numpy(float), mb, stb, rng)))
                        RU.MIN_STRATUM_BG = old
    pct = {sc: within_stratum_pct(g, sc, st_tau) for sc in ["S_can", "D_can", "D_free", "B1", "LVL"]}
    dec = {sc: g[sc].rank(pct=True).to_numpy() for sc in ["S_can", "D_can", "D_free", "B1"]}
    gid = g.hsapiens_gene_id.to_numpy()
    for _, r in lst.dropna(subset=["hsapiens_gene_id"]).drop_duplicates(["hsapiens_gene_id", "set"]).iterrows():
        ix = np.where(gid == r.hsapiens_gene_id)[0]
        if len(ix) == 0:
            continue
        i = ix[0]
        gr = g.iloc[i]
        genes.append(dict(assign_id=a["assign_id"], source_id=a["source_id"], circular=a["circular"], cohort=c, set=r.set,
                          symbol=gr.symbol, gene_symbol_list=r.gene_symbol, hsapiens_gene_id=r.hsapiens_gene_id,
                          cell_type_list=r.cell_type, direction=r.direction, effect=r.effect,
                          D_can=gr.D_can, S_can=gr.S_can, D_free=gr.D_free, S_free=gr.S_free, B1=gr.B1, LVL=gr.LVL,
                          pi_raw=gr.pi_raw, expr=gr.expr, tau=gr.tau,
                          **{f"pct_{kk}": v[i] for kk, v in pct.items()}, **{f"cohort_pct_{kk}": v[i] for kk, v in dec.items()},
                          supp_both=_tru(gr.supp_both), nameable=_tru(gr.nameable), nameable_free=_tru(gr.nameable_free),
                          stable=_tru(gr.sign_D_stable) and _tru(gr.sign_S_stable) and _tru(gr.sign_D_stable_free) and _tru(gr.sign_S_stable_free),
                          saturated=_tru(gr.saturated_flag) or _tru(gr.saturated_flag_free),
                          reference_artefact=_tru(gr.get("reference_artefact", False)),
                          o1_T2=_tru(gr.o1_T2), quiet=_tru(gr.quiet_in_species),
                          **{c2: gr[c2] for c2 in g.columns if c2.startswith("detection_")}))
    return rows, genes, lst.assign(assign_id=a["assign_id"], cohort=c)


def main():
    from multiprocessing import Pool
    RU.OUT.mkdir(parents=True, exist_ok=True)
    assign = pd.read_csv(TASK / "assign.tsv", sep="\t", comment="#", dtype=str).fillna("")
    only = sys.argv[1:]
    jobs = []
    for _, a in assign.iterrows():
        if only and a.assign_id not in only:
            continue
        for c in a.cohorts.split(","):
            jobs.append((a.to_dict(), c, len(jobs)))
    rows, genes, maps = [], [], []
    with Pool(min(18, len(jobs)), initializer=_init) as pool:
        for r, gl, m in pool.imap_unordered(run_one, jobs):
            rows += r
            genes += gl
            maps.append(m)
            print(f"done {m.assign_id.iloc[0]} {m.cohort.iloc[0]}", flush=True)
    tag = "" if not only else "_" + "_".join(only)
    pd.concat(maps).to_csv(RU.OUT / f"mapping{tag}.tsv.gz", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(RU.OUT / f"auroc{tag}.tsv", sep="\t", index=False)
    gt = pd.DataFrame(genes)
    gt.to_csv(RU.OUT / f"genes{tag}.tsv.gz", sep="\t", index=False)
    named = gt[gt.nameable & gt.stable & ~gt.saturated & gt.supp_both & ((gt.cohort_pct_S_can <= 0.10) | (gt.cohort_pct_S_can >= 0.90))]
    named.to_csv(RU.OUT / f"named{tag}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
