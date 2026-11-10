"""EXPLORATORY: attempts to explain the results of analyses.py another way. All of this was written after reading them; none of it is a search for a
Chimera positive and none is reported as one.

F1  what removes the C-only tissue-matched signal: add tau, then dN/dS and LOEUF, then PubMed, one at a time.
F2  B3a's tissue-matched signal: does it survive gene class (X3_cross_cohort classes) and 20 or 64 ESM PCs? Same for S, B1b.
F3  testis Delta-alpha against promoter phyloP, CpG class and essentiality: given CpG class, given S, given 64 ESM PCs; D_raw and B3b beside.
F4  bone-marrow S on hematopoietic concordance: given log variance, given B1b; human-mouse only profile correlation beside.
F5  discordant classes on phyloP: rematch on the region's GC and log length; calibration on random sets of the same size from the same pi band.
F6  testis PD-high on reproductive concordance: matched also on spread and pleiotropy; rank partial correlation inside pi tertiles.
F7  calibration of all class excesses in (c) on random sets from the class's pi band, because nearest-neighbor matching leaves part of pi in the excess.
F4b replication diagonals with IMPC-only mouse side and ClinGen genes.
Usage (from chimera/): PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.Y2_functional.followup
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analysis.tasks.X4_where_differs import common as CM
from analysis.tasks.Y2_functional import analyses as A
from analysis.tasks.Y2_functional import rules as RU

OUT = A.OUT
N = A.N_JOBS
CLS = ["cls_enzyme", "cls_TF", "cls_kinase_any", "cls_receptor", "cls_GPCR", "cls_ion_channel", "cls_SLC_transporter", "cls_chromatin_reg", "cls_mitochondrial"]
C4 = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]


def with_x4(t: pd.DataFrame, c: str) -> pd.DataFrame:
    g = pd.read_parquet(A.RES / f"X4/genes/{c}.parquet")
    keep = ["var_name"] + [k for k in g.columns if k.startswith("esm")] + CLS
    x = t.merge(g[keep], on="var_name", how="left")
    for k in CLS:
        x[k] = x[k].astype(float)
    return x


def lin_design(T: pd.DataFrame, covs: list[str], linear: list[str]) -> np.ndarray:
    Z = A.z_design(T, covs)
    if linear:
        L = T[linear].to_numpy(float)
        L = (L - L.mean(0)) / (L.std(0) + 1e-12)
        Z = np.column_stack([Z, L])
    return Z


def cell2(t, rows, y, covs, linear, scores, block):
    c = A.make_cell(t, rows & t[linear].notna().all(axis=1).to_numpy() if linear else rows, y, covs, scores, block)
    if c is None:
        return None
    cols = [y] + covs + scores
    m = t[cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy() & (rows & t[linear].notna().all(axis=1).to_numpy() if linear else rows)
    c["Z"] = lin_design(t.loc[m], covs, linear)
    return c


def f1_f2(tabs, out):
    L, T = with_x4(tabs["liver"], "liv_hmouse"), with_x4(tabs["testis"], "test_hmouse")
    D = A.Draws(list(L["gblock"].astype(str)) + list(T["gblock"].astype(str)), RU.DRAWS, RU.SEED + 21)
    base = C4
    esm20 = [f"esm{i}" for i in range(20)]
    esm64 = [f"esm{i}" for i in range(64)]
    specs = {
        "C": (base, []), "C+tau": (base + ["tau_shared"], []), "C+tau+dNdS+LOEUF": (base + ["tau_shared", "dNdS", "LOEUF"], []),
        "C+dNdS+LOEUF": (base + ["dNdS", "LOEUF"], []), "C+PubMed": (base + A.PUB, []),
        "C_Y2": (RU.C_Y2, []), "C_Y2+class": (RU.C_Y2, CLS), "C_Y2+ESM20": (RU.C_Y2, esm20), "C_Y2+ESM64": (RU.C_Y2, esm64), "C_Y2+class+ESM64": (RU.C_Y2, CLS + esm64),
    }
    scores = ["D", "S", "Sdir_pattern", "B1b_raw", "B1c_raw", "B1z_raw", "B2b", "B3a", "random"]
    cells, keys = {}, []
    for name, (cov, lin) in specs.items():
        for u in ("ms", "un"):
            for tissue, tt in (("liver", L), ("testis", T)):
                for s in ("hepmet", "repro"):
                    c = cell2(tt, tt[u].to_numpy(bool) & tt[f"U_{s}"].to_numpy(bool), f"y_{s}", cov, lin, scores, "gblock")
                    cells[(name, u, tissue, s)] = c
                    keys.append((name, u, tissue, s))
    res = Parallel(n_jobs=N, backend="loky")(delayed(A.eval_cell)(cells[k], D) for k in keys)
    E = dict(zip(keys, res))
    rows = []
    for name in specs:
        for u in ("ms", "un"):
            a, b, c_, d = E[(name, u, "liver", "hepmet")], E[(name, u, "liver", "repro")], E[(name, u, "testis", "repro")], E[(name, u, "testis", "hepmet")]
            e = 0.5 * ((a[0] - b[0]) + (c_[0] - d[0]))
            B = 0.5 * ((a[1] - b[1]) + (c_[1] - d[1]))
            for j, s in enumerate(scores):
                rows.append(dict(adj=name, universe=u, score=s, stat="Delta_match", n="/".join(str(cells[(name, u, x, y)]["n"]) for x, y in (("liver", "hepmet"), ("liver", "repro"), ("testis", "repro"), ("testis", "hepmet")))) | A.summar(e[j], B[:, j]))
            for tissue, sys_ in (("liver", "hepmet"), ("liver", "repro"), ("testis", "repro"), ("testis", "hepmet")):
                ee, BB = E[(name, u, tissue, sys_)]
                for j, s in enumerate(scores):
                    rows.append(dict(adj=name, universe=u, score=s, stat=f"{tissue}_{sys_}") | A.summar(ee[j], BB[:, j]))
    out["F1_F2_adjustment"] = pd.DataFrame(rows)


def f3(tabs, out):
    rows = []
    for tissue, c in (("testis", "test_hmouse"), ("liver", "liv_hmouse")):
        t = with_x4(tabs[tissue], c)
        t["log_abc"] = np.log1p(t["abc_links_mean"])
        t["promoter_cgi_f"] = t["promoter_cgi"].astype(float)
        cs = t["depmap_common"].astype(float)
        leth = t["lethal_known"]
        t["U_E"] = leth.notna() & cs.notna() & ((leth == 1) | (cs == 1))
        t["y_E"] = ((leth == 1) & (cs == 1)).astype(float).where(t["U_E"])
        t["y_common"] = cs
        D = A.Draws(t["block"].astype(str).tolist(), RU.DRAWS, RU.SEED + 22)
        esm = [f"esm{i}" for i in range(64)]
        reg = [c_ for c_ in RU.C_Y2 if c_ not in A.PUB] + ["gc_promoter", "log_length_promoter"]
        reg2 = [c_ for c_ in RU.C_Y2 if c_ not in A.PUB]
        specs = [("phylop_promoter", reg, [], "C_REG"), ("phylop_promoter", reg, ["promoter_cgi_f"], "C_REG+CGI"), ("phylop_promoter", reg, esm, "C_REG+ESM64"), ("phylop_promoter", reg + ["S"], [], "C_REG+S"),
                 ("phylop_utr5", reg2 + ["gc_utr5", "log_length_utr5"], [], "C_REG"), ("phylop_utr5", reg2 + ["gc_utr5", "log_length_utr5"], ["promoter_cgi_f"], "C_REG+CGI"), ("phylop_utr5", reg2 + ["gc_utr5", "log_length_utr5"], esm, "C_REG+ESM64"),
                 ("promoter_cgi", reg2, [], "C_REG"), ("promoter_cgi", reg2, esm, "C_REG+ESM64"), ("promoter_cgi", reg2 + ["S"], [], "C_REG+S"), ("promoter_cgi", reg2 + ["spread"], [], "C_REG+spread"),
                 ("y_E", RU.C_Y2, [], "C_Y2"), ("y_E", RU.C_Y2, esm, "C_Y2+ESM64"), ("y_E", RU.C_Y2 + ["S"], [], "C_Y2+S"), ("y_common", RU.C_Y2, [], "C_Y2"), ("y_common", RU.C_Y2, esm, "C_Y2+ESM64")]
        for y, cov, lin, name in specs:
            for u in ("un", "ms"):
                rowsm = t[u].to_numpy(bool)
                scores = [s for s in ("D", "D_raw", "S", "B1b_raw", "B3a", "B3b") if s not in cov]
                c_ = cell2(t, rowsm, y, cov, lin, scores, "block")
                if c_ is None:
                    continue
                e, B = A.eval_cell(c_, D)
                for j, s in enumerate(scores):
                    rows.append(dict(tissue=tissue, universe=u, outcome=y, adj=name, score=s, n=c_["n"]) | A.summar(e[j], B[:, j]))
    out["F3_D_regulatory"] = pd.DataFrame(rows)


def f4(tabs, out):
    t = tabs["bone_marrow"].copy()
    D = A.Draws(t["block"].astype(str).tolist(), RU.DRAWS, RU.SEED + 23)
    rows = []
    t["B1b_hm"] = t["E_human_mouse"]
    scores = ["D", "S", "B1b_raw", "B1b_hm", "random"]
    for u in ("un", "ms"):
        for name, cov in (("C_Y2", RU.C_Y2), ("C_Y2+log_var", RU.C_Y2 + ["log_var"]), ("C_Y2+det_range", RU.C_Y2 + ["detection_sprange"]), ("no_tau", [c for c in RU.C_Y2 if c != "tau_shared"])):
            est, draws = {}, {}
            for s in ("heme", "hepmet", "repro", "renal"):
                c = A.make_cell(t, t[u].to_numpy(bool) & t[f"U_{s}"].to_numpy(bool), f"y_{s}", cov, scores, "block")
                est[s], draws[s] = A.eval_cell(c, D)
                for j, x in enumerate(scores):
                    rows.append(dict(universe=u, adj=name, system=s, score=x, stat="cell", n=c["n"]) | A.summar(est[s][j], draws[s][:, j]))
            for j, x in enumerate(scores):
                e = est["heme"][j] - np.mean([est[k][j] for k in ("hepmet", "repro", "renal")])
                B = draws["heme"][:, j] - np.mean([draws[k][:, j] for k in ("hepmet", "repro", "renal")], axis=0)
                rows.append(dict(universe=u, adj=name, system="heme", score=x, stat="diag_minus_offdiag") | A.summar(e, B))
        # S given B1b, B1b given S
        for x, given in (("S", "B1b_raw"), ("B1b_raw", "S"), ("S", "D"), ("D", "S")):
            c = A.make_cell(t, t[u].to_numpy(bool) & t["U_heme"].to_numpy(bool), "y_heme", RU.C_Y2 + [given], [x], "block")
            e, B = A.eval_cell(c, D)
            rows.append(dict(universe=u, adj=f"C_Y2+{given}", system="heme", score=x, stat="cell", n=c["n"]) | A.summar(e[0], B[:, 0]))
    out["F4_bone_marrow"] = pd.DataFrame(rows)


def _rand_matched(t, pool, band, n, y, match, seed):
    rng = np.random.default_rng(seed)
    cand = np.where(band & pool)[0]
    f = np.zeros(len(t), bool)
    f[rng.choice(cand, min(n, len(cand)), replace=False)] = True
    return CM.matched(t, f, pool, y, match, key=f"rand{seed}")["excess_d"]


def f5(tabs, out):
    rows, cal = [], []
    for tissue in ("liver", "testis"):
        t = tabs[tissue].copy()
        cl = A.classes(t)
        match = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared", "log_pubmed_h"] + (["hep_enrich"] if tissue == "liver" else [])
        pi = CM.tertile(t["prior_raw"])
        for name, (flag, pool) in cl.items():
            for y, gc, ln in (("phylop_promoter", "gc_promoter", "log_length_promoter"), ("phylop_utr5", "gc_utr5", "log_length_utr5"), ("phylop_utr3", "gc_utr3", "log_length_utr3")):
                r0 = CM.matched(t, flag, pool, y, match, key=f"Y2|{tissue}|{name}")
                r1 = CM.matched(t, flag, pool, y, match + [gc, ln], key=f"Y2|{tissue}|{name}")
                t["_cgi"] = t["promoter_cgi"].astype(float)
                r2 = CM.matched(t, flag, pool, y, match + [gc, ln, "_cgi"], key=f"Y2|{tissue}|{name}")
                rows.append(dict(tissue=tissue, cls=name, outcome=y, match="base") | r0)
                rows.append(dict(tissue=tissue, cls=name, outcome=y, match="+GC+length") | r1)
                rows.append(dict(tissue=tissue, cls=name, outcome=y, match="+GC+length+CGI") | r2)
                band = (pi == 3).to_numpy() if name.startswith("PClow") else (pi == 1).to_numpy()
                if flag.sum() >= 20 and name in ("PClow_D_ms", "PClow_S_un", "PDhigh_S_un", "PClow_B1b_un", "PDhigh_D_ms"):
                    ds = Parallel(n_jobs=N, backend="loky")(delayed(_rand_matched)(t, pool, band, int(flag.sum()), y, match + [gc, ln], 1000 + i) for i in range(200))
                    ds = np.array(ds, dtype=float)
                    obs = r1["excess_d"]
                    cal.append(dict(tissue=tissue, cls=name, outcome=y, obs=obs, null_mean=np.nanmean(ds), null_sd=np.nanstd(ds), frac_ge=(np.nansum(np.abs(ds - np.nanmean(ds)) >= abs(obs - np.nanmean(ds))) + 1) / (np.isfinite(ds).sum() + 1)))
    out["F5_discordant_phylop"] = pd.DataFrame(rows)
    out["F5_calibration"] = pd.DataFrame(cal)


def f6(tabs, out):
    t = tabs["testis"].copy()
    cl = A.classes(t)
    rows = []
    base = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared", "log_pubmed_h"]
    t["_n_mp"], t["_n_hp"] = t["n_mp_sys"].astype(float), t["n_hp_sys"].astype(float)
    u = t["U_repro"].to_numpy(bool)
    for name in ("PDhigh_D_ms", "PDhigh_S_un", "PDhigh_B1b_un", "PDhigh_B1c_un"):
        f, p = cl[name]
        for lab, m in (("base", base), ("+spread", base + ["spread"]), ("+pleiotropy", base + ["_n_mp", "_n_hp"]), ("+PubMed mouse", base + ["log_pubmed_m"]), ("+dNdS+LOEUF", base + ["dNdS", "LOEUF"])):
            rows.append(dict(cls=name, outcome="y_repro", match=lab) | CM.matched(t, f & u, p & u, "y_repro", m, key=f"F6|{name}"))
    pi = CM.tertile(t["prior_raw"])
    D = A.Draws(t["block"].astype(str).tolist(), RU.DRAWS, RU.SEED + 24)
    scores = ["D", "D_raw", "S", "Sdir_pattern", "Sdir_level", "B1b_raw", "B1c_raw", "B1z_raw", "B2b", "B3a"]
    for uni in ("un", "ms"):
        for pt in (1, 2, 3):
            for sys_ in ("repro", "hepmet"):
                for name, tab_ in (("testis", t),):
                    rows_ = tab_[uni].to_numpy(bool) & tab_[f"U_{sys_}"].to_numpy(bool) & (pi == pt).to_numpy()
                    c = A.make_cell(tab_, rows_, f"y_{sys_}", [x for x in RU.C_Y2], scores, "block")
                    if c is None:
                        continue
                    e, B = A.eval_cell(c, D)
                    for j, s in enumerate(scores):
                        rows.append(dict(cls="pi_tertile", pi_tertile=pt, universe=uni, outcome=f"y_{sys_}", score=s, n=c["n"], n_pos=c["n_pos"]) | A.summar(e[j], B[:, j]))
    out["F6_testis_PDhigh"] = pd.DataFrame(rows)


def _rand_excess(t, pool, band, n, y, match, seed):
    rng = np.random.default_rng(seed)
    cand = np.where(band & pool)[0]
    if len(cand) < n or n < 5:
        return np.nan
    f = np.zeros(len(t), bool)
    f[rng.choice(cand, n, replace=False)] = True
    return CM.matched(t, f, pool, y, match, key=f"rand{seed}")["excess_d"]


def f7(tabs, out):
    """Calibration of (c): the same matched excess on 200 random sets of the same size from the class's pi band, for every outcome."""
    rows = []
    jobs = []
    for tissue in ("liver", "testis"):
        t = tabs[tissue].copy()
        t["log_abc"] = np.log1p(t["abc_links_mean"])
        for k in ("M_any", "H_any", "mouse_lethal", "promoter_cgi", "tata"):
            t[k] = t[k].astype(float)
        cl = A.classes(t)
        match = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared", "log_pubmed_h"] + (["hep_enrich"] if tissue == "liver" else [])
        pi = CM.tertile(t["prior_raw"])
        for name, (flag, pool) in cl.items():
            band = (pi == 3).to_numpy() if name.startswith("PClow") else (pi == 1).to_numpy()
            if name.startswith("PClow"):
                band_flag = (t["S"].to_numpy(float) < 0) if name.endswith("_ms") else np.ones(len(t), bool)
            else:
                band_flag = (t["S"].to_numpy(float) > 0) if name.endswith("_ms") else np.ones(len(t), bool)
            for y in ("y_hepmet", "y_repro", "M_any", "H_any", "mouse_lethal", "promoter_cgi", "log_abc"):
                f, p = flag, pool
                if y.startswith("y_"):
                    u = t[f"U_{y[2:]}"].to_numpy(bool)
                    f, p = flag & u, pool & u
                r = CM.matched(t, f, p, y, match, key=f"Y2|{tissue}|{name}")
                jobs.append((tissue, name, y, r, t, p, band & band_flag, int(f.sum()), match))
    for tissue, name, y, r, t, p, band, n, match in jobs:
        ds = np.array(Parallel(n_jobs=N, backend="loky")(delayed(_rand_excess)(t, p, band, n, y, match, 2000 + i) for i in range(200)), dtype=float)
        obs = r["excess_d"]
        ok = np.isfinite(ds)
        rows.append(dict(tissue=tissue, cls=name, outcome=y, n=n, obs=obs, ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_neighbour=r["p"], null_mean=np.nanmean(ds) if ok.any() else np.nan, null_sd=np.nanstd(ds) if ok.any() else np.nan,
                         excess_over_null=obs - np.nanmean(ds) if ok.any() else np.nan,
                         p_random=(np.sum(np.abs(ds[ok] - ds[ok].mean()) >= abs(obs - ds[ok].mean())) + 1) / (ok.sum() + 1) if ok.any() else np.nan))
    r = pd.DataFrame(rows)
    r["q_random"] = np.nan
    pp = r["p_random"].dropna()
    r.loc[pp.index, "q_random"] = A.S.bh(pp.tolist())
    out["F7_discordant_calibrated"] = r


def f4b(tabs, out):
    """Bone marrow and kidney, IMPC-only mouse side and ClinGen genes; diagonal minus off-diagonal."""
    rows = []
    for tissue, own in (("bone_marrow", "heme"), ("kidney", "renal"), ("testis", "repro"), ("liver", "hepmet")):
        t = tabs[tissue].copy()
        D = A.Draws(t["block"].astype(str).tolist(), RU.DRAWS, RU.SEED + 25)
        scores = [s for s in ("D", "S", "B1b_raw", "random") if s in t]
        others = [k for k in ("heme", "renal", "repro", "hepmet") if k != own]
        for u in ("un", "ms"):
            for mode in ("impc", "clingen"):
                est, draws = {}, {}
                for k in [own] + others:
                    if mode == "impc":
                        c = A.make_cell(t, t[u].to_numpy(bool) & t[f"Ui_{k}"].to_numpy(bool), f"yi_{k}", RU.C_Y2, scores, "block")
                    else:
                        c = A.make_cell(t, t[u].to_numpy(bool) & t[f"U_{k}"].to_numpy(bool) & t["clingen_ok"].fillna(False).to_numpy(bool), f"y_{k}", RU.C_Y2, scores, "block")
                    if c is None:
                        continue
                    est[k], draws[k] = A.eval_cell(c, D)
                off = [k for k in others if k in est]
                if own not in est or not off:
                    continue
                for j, s in enumerate(scores):
                    rows.append(dict(tissue=tissue, universe=u, mode=mode, score=s) | A.summar(est[own][j] - np.mean([est[k][j] for k in off]), draws[own][:, j] - np.mean([draws[k][:, j] for k in off], axis=0)))
    out["F4b_replication_sens"] = pd.DataFrame(rows)


def f4c(tabs, out):
    """Bone marrow: is S's hematopoietic link carried by human and mouse rows, or by the mouse lemur rows (phenotypes are human and mouse)?"""
    t = tabs["bone_marrow"].copy()
    g = pd.read_parquet(A.RES / "P12/drivers/drivers_genes.parquet")
    for sp, cols in (("human", [k for k in g if k.startswith("R_human_")]), ("mouse", [k for k in g if k.startswith("R_mouse_") and not k.startswith("R_mouse_lemur_")]), ("lemur", [k for k in g if k.startswith("R_mouse_lemur_")])):
        g["S_" + sp] = g[cols].sum(axis=1)
    g["S_hm"] = g["S_human"] + g["S_mouse"]
    print("S_total vs sum of rows, max abs difference", float((g["S_total"] - g["S_hm"] - g["S_lemur"]).abs().max()), flush=True)
    t = t.merge(g[["var_name", "S_total", "S_human", "S_mouse", "S_lemur", "S_hm"]], on="var_name", how="left")
    D = A.Draws(t["block"].astype(str).tolist(), RU.DRAWS, RU.SEED + 26)
    scores = ["S", "S_total", "S_hm", "S_human", "S_mouse", "S_lemur", "B1b_raw"]
    t["B1b_hm"] = t["E_human_mouse"]
    scores.append("B1b_hm")
    rows = []
    for u in ("un", "ms"):
        est, draws = {}, {}
        for k in ("heme", "hepmet", "repro", "renal"):
            c = A.make_cell(t, t[u].to_numpy(bool) & t[f"U_{k}"].to_numpy(bool), f"y_{k}", RU.C_Y2, scores, "block")
            est[k], draws[k] = A.eval_cell(c, D)
            for j, x in enumerate(scores):
                rows.append(dict(universe=u, system=k, score=x, stat="cell", n=c["n"]) | A.summar(est[k][j], draws[k][:, j]))
        for j, x in enumerate(scores):
            rows.append(dict(universe=u, system="heme", score=x, stat="diag_minus_offdiag") | A.summar(est["heme"][j] - np.mean([est[k][j] for k in ("hepmet", "repro", "renal")]), draws["heme"][:, j] - np.mean([draws[k][:, j] for k in ("hepmet", "repro", "renal")], axis=0)))
    out["F4c_bone_marrow_species"] = pd.DataFrame(rows)


def main():
    import sys

    tabs = A.load(False)
    fns = dict(f1=f1_f2, f3=f3, f4=f4, f4b=f4b, f4c=f4c, f5=f5, f6=f6, f7=f7)
    for fn in [fns[a] for a in (sys.argv[1:] or list(fns))]:
        res: dict = {}
        fn(tabs, res)
        for k, v in res.items():
            v.to_csv(OUT / f"{k}.csv", index=False)
            print(k, v.shape, flush=True)


if __name__ == "__main__":
    main()
