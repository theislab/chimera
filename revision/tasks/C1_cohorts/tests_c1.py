"""C1 tests of plan section 13, one file, same statistics for every cohort. Exploratory, Family 3.

Inputs: results/revision/C1/h1/genes/<cohort>.parquet (H1 schema), drivers fw files, golds, DE and H-D4 of the cases, references.tsv.

    python revision/tasks/C1_cohorts/tests_c1.py brain | ad | lung | tumour  [--n-perm 10000] [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.common import gene_ids
from revision.tasks.C1_cohorts import configs as CF
from revision.tasks.T2_5_celltype_margin import margin as MG
from revision.tasks.T4_3_scoreboard import inputs as I
from revision.tasks.T4_3_scoreboard import stats as S

C_COLS = ["pi_raw", "expr", "detection", "log_cds_length"]
H1 = __import__("pathlib").Path(__import__("os").environ.get("C1_H1_OUT", str(CF.RES / "h1")))
GOLD = CF.RES / "golds"
REF = pd.read_csv(CF.TASK / "references.tsv", sep="\t", keep_default_na=False)
SEED = 20261102


def genes(c: str) -> pd.DataFrame:
    g = pd.read_parquet(H1 / "genes" / f"{c}.parquet").reset_index()
    a = ad.read_h5ad(CF.h5ad(c), backed="r")
    F = np.asarray(a.varm["gene_features"], dtype=float)
    a.file.close()
    g["block"], _ = I.esm_blocks(F[:, :I.N_PCS])
    return g


def rho_table(u: pd.DataFrame, target: str, scores: list[str], n_perm: int, n_boot: int, rng, label: dict, extra_nuis: tuple = ()) -> list[dict]:
    """rho_p(score, target | C) on one universe (all scores defined); permutation within pi x expression; ESM-block bootstrap; Delta to the first score."""
    u = u.dropna(subset=[target, *scores, *C_COLS, *extra_nuis]).reset_index(drop=True)
    Z = S.design(u, C_COLS + list(extra_nuis))
    R = np.column_stack([S.midrank(u[target])] + [S.midrank(u[s]) for s in scores])
    E = S.resid(R, Z)
    lab = S.strata(u["pi_raw"].to_numpy(), u["expr"].to_numpy())
    draws = S.bootstrap(R, Z, n_boot, rng, u["block"].to_numpy())
    obs = S.rho_p_matrix(R, Z)
    rows = []
    for j, s in enumerate(scores, start=1):
        null = S.perm_null(E[:, j], E[:, 0], lab, n_perm, rng)
        lo, hi = S.pct_ci(draws[:, j])
        rows.append(dict(**label, score=s, n=len(u), rho_raw=float(spearmanr(u[s], u[target])[0]), rho_p=float(obs[j]), ci_lo=lo, ci_hi=hi, p_two=S.p_two_sided(float(obs[j]), null),
                         null_sd=float(null.std()), delta_to_first=float(obs[j] - obs[1]), delta_lo=S.pct_ci(draws[:, j] - draws[:, 1])[0], delta_hi=S.pct_ci(draws[:, j] - draws[:, 1])[1]))
    return rows


def boot_spear(a: np.ndarray, b: np.ndarray, block: np.ndarray, rng, n: int = 500) -> tuple[float, float, float]:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, block = a[ok], b[ok], block[ok]
    rho = float(spearmanr(a, b)[0])
    ids = np.unique(block)
    groups = {i: np.flatnonzero(block == i) for i in ids}
    out = np.empty(n)
    for k in range(n):
        idx = np.concatenate([groups[i] for i in rng.choice(ids, len(ids))])
        out[k] = spearmanr(a[idx], b[idx])[0]
    return rho, *S.pct_ci(out)


def tert(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average", pct=True).to_numpy()
    return np.minimum((r * 3).astype(int), 2)


def auroc_row(g: pd.DataFrame, members: set, scores: list[str], rng, label: dict, n_perm: int = 10000, tau: bool = False, background: set | None = None) -> list[dict]:
    """Stratified AUROC (pi tertile x expression tertile; with tau: x tau tertile): set vs background (all other genes, or the given set)."""
    u = g.dropna(subset=["pi_raw", "expr"]).reset_index(drop=True)
    inset = u["symbol"].isin(members).to_numpy()
    if background is not None:
        keep = inset | u["symbol"].isin(background).to_numpy()
        u, inset = u[keep].reset_index(drop=True), inset[keep]
    lab = tert(u["pi_raw"].to_numpy()) * 10 + tert(u["expr"].to_numpy())
    if tau:
        lab = lab * 10 + tert(u["tau"].fillna(u["tau"].median()).to_numpy())
    rows = []
    for s in scores:
        ok = u[s].notna().to_numpy()
        keep = S.kept_strata(inset[ok], lab[ok], ratio=1, absolute=True)
        v, i2, l2 = u[s].to_numpy(float)[ok][keep], inset[ok][keep], lab[ok][keep]
        if i2.sum() < 3:
            rows.append(dict(**label, score=s, n_set=int(i2.sum()), tau_matched=tau, auroc=np.nan))
            continue
        a = S.strat_auroc(v, i2, l2)
        null = S.auroc_perm(v, i2, l2, n_perm, rng)
        rows.append(dict(**label, score=s, n_set=int(i2.sum()), n_bg=int((~i2).sum()), tau_matched=tau, auroc=float(a), null_mean=float(null.mean()), p_two=float((1 + (np.abs(null - null.mean()) >= abs(a - null.mean())).sum()) / (len(null) + 1))))
    return rows


def fw_tables(c: str) -> dict:
    """Seed mean over kept seeds of S_gsk (species x type rows), weights, labels, species."""
    runs = pd.read_csv(H1 / "runs.tsv", sep="\t")
    seeds = sorted(runs[(runs.cohort == c) & runs.kept].seed)
    S_, dec = None, None
    n = 0
    for s in seeds:
        f = CF.RES / c / "drivers" / f"fw_s{s}.npz"
        if not f.exists():
            continue
        fw = dict(np.load(f, allow_pickle=False))
        dec = MG.decompose(fw, "cell_ontology_class")
        S_ = dec["S_gsk_meanz"] if S_ is None else S_ + dec["S_gsk_meanz"]
        n += 1
        vn = fw["var_names"].astype(str)
    return dict(S=S_ / n, w=dec["w"], labels=dec["labels"], sp=dec["species_idx"], species=dec["species"], var_names=vn, n_seeds=n)


def poisson_binomial_p(ps: np.ndarray, k: int) -> float:
    d = np.array([1.0])
    for p in ps:
        d = np.convolve(d, [1 - p, p])
    return float(d[k:].sum())


def named_pairs(c: str, g: pd.DataFrame, cohort_set: str, fw: dict, row_species: str, out_label: dict, base_col: str = "E") -> tuple[dict, pd.DataFrame]:
    """E-B4: named (gene, class): top class of |row contribution| in species row; chance = base rate of that class being top over supported genes."""
    idx = pd.Index(fw["var_names"])
    sidx = fw["species"].index(row_species)
    rows = np.flatnonzero(fw["sp"] == sidx)
    types = fw["labels"][rows]
    C = fw["S"][rows] * fw["w"][rows, None]
    top = np.array(types)[np.abs(C).argmax(0)]
    gi = g.set_index("var_name")
    sup = gi["count_support"].reindex(idx).fillna(False).to_numpy()
    base = pd.Series(top[sup]).value_counts(normalize=True)
    r = REF[(REF.cohort == cohort_set) & (REF.cell_type != "") & REF.set.isin(["human_lineage", "hominini"])]
    sym = gi["symbol"].reindex(idx).to_numpy()
    rec, ps = [], []
    for x in r.itertuples():
        m = np.flatnonzero(sym == x.gene)
        if len(m) != 1 or x.cell_type not in set(types):
            continue
        hit = top[m[0]] == x.cell_type
        ps.append(float(base.get(x.cell_type, 0.0)))
        rec.append(dict(gene=x.gene, named_type=x.cell_type, top_type=top[m[0]], hit=bool(hit), supported=bool(sup[m[0]]), chance=ps[-1], S_ema=float(gi.loc[idx[m[0]], "S_ema"]), D=float(gi.loc[idx[m[0]], "D"])))
    t = pd.DataFrame(rec)
    k = int(t.hit.sum()) if len(t) else 0
    return dict(**out_label, n_pairs=len(t), hits=k, expected=float(sum(ps)), p=poisson_binomial_p(np.array(ps), k) if len(t) else np.nan), t


# ------------------------------------------------------------------------------------------------ cortex
BRAIN_GOLDS = {"brain_dlpfc": ["M1_marmoset", "MTG_marmoset", "MTG_macaque"], "brain_dlpfc_hmac": ["MTG_macaque"], "brain_dlpfc_hmarm": ["M1_marmoset", "MTG_marmoset"],
               "brain_dlpfc_hchimp": ["MTG_macaque"], "brain_m1": ["DLPFC_marmoset", "MTG_marmoset"]}
PARTNER = {"brain_dlpfc_hchimp": "ptroglodytes", "brain_dlpfc_hmac": "mmulatta", "brain_dlpfc_hmarm": "cjacchus", "brain_dlpfc": "ptroglodytes", "brain_m1": "cjacchus", "brain_m1_hmouse": "mmusculus"}
ROWS = {"brain_dlpfc": ["Human", "Chimpanzee", "Macaque", "Marmoset"], "brain_m1": ["Human", "Marmoset", "Mouse"]}


def brain(n_perm: int, n_boot: int) -> None:
    rng = np.random.default_rng(SEED)
    out = CF.RES / "cortex"
    out.mkdir(exist_ok=True)
    e1, e2, e3, e4, e4d, e5 = [], [], [], [], [], []
    tabs = {}
    for c, gl in BRAIN_GOLDS.items():
        if not (H1 / "genes" / f"{c}.parquet").exists():
            continue
        g = genes(c)
        tabs[c] = g
        scores = ["D", "S_ema", "B1", "pi_raw"]
        part = PARTNER[c]
        g["identity_partner"] = g[f"{part}_identity"].astype(float)
        g["log_dnds_partner"] = np.log10((g[f"{part}_dn"].astype(float) / g[f"{part}_ds"].astype(float)).where(lambda x: x > 0).clip(lower=1e-3))
        g["log_dnds_mouse"] = np.log10(g["dNdS"].where(g["dNdS"] > 0).clip(lower=1e-3))
        for gname in gl:
            gg = pd.read_parquet(GOLD / f"{gname}.parquet").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
            u = g.assign(gold=gg["gold"].reindex(g["hsapiens_gene_id"]).to_numpy())
            lab = dict(cohort=c, gold=gname)
            e1 += rho_table(u, "gold", scores, n_perm, n_boot, rng, lab)
            for name, mask in (("identity_ge_98", u["identity_partner"] >= 98), ("top_pi_tertile", u["pi_raw"].rank(pct=True) > 2 / 3), ("identity_ge_99_5", u["identity_partner"] >= 99.5)):
                v = u[mask.to_numpy()]
                for sc in ("identity_partner", "log_dnds_mouse", "D", "S_ema", "B1"):
                    if v[sc].notna().sum() < 50 or v[sc].nunique() < 3:
                        continue
                    a = v[["gold", sc, "block"]].dropna()
                    rho, lo, hi = boot_spear(a[sc].to_numpy(float), a["gold"].to_numpy(float), a["block"].to_numpy(), rng)
                    e2.append(dict(**lab, stratum=name, score=sc, n=len(a), spearman=rho, ci_lo=lo, ci_hi=hi))
        lab = dict(cohort=c)
        for st in ("human_lineage", "hominini", "primate_vs_mouse", "conserved_marker"):
            mem = set(REF[(REF.cohort == "brain") & (REF.set == st)].gene)
            e3 += auroc_row(g, mem, ["S_ema", "D", "B1"], rng, {**lab, "set": st}, tau=False) + auroc_row(g, mem, ["S_ema", "D", "B1"], rng, {**lab, "set": st}, tau=True)
        mem_c = set(REF[(REF.cohort == "brain") & (REF.set == "conserved_marker")].gene)
        mem_h = set(REF[(REF.cohort == "brain") & (REF.set == "human_lineage")].gene)
        e3 += auroc_row(g, mem_c, ["S_ema", "D", "B1"], rng, {**lab, "set": "conserved_vs_lineage"}, background=mem_h)
        fwf = CF.RES / c / "drivers"
        if c in ROWS and fwf.exists() and list(fwf.glob("fw_s*.npz")):
            fw = fw_tables(c)
            r, t = named_pairs(c, g, "brain", fw, "Human", {"cohort": c})
            e4.append(r)
            e4d.append(t.assign(cohort=c))
            sup = g.set_index("var_name")["count_support"].reindex(fw["var_names"]).fillna(False).to_numpy()
            for sp_name in fw["species"]:
                rows = np.flatnonzero(fw["sp"] == fw["species"].index(sp_name))
                Cr = (fw["S"][rows] * fw["w"][rows, None])[:, sup]
                total_neg = np.where(Cr < 0, -Cr, 0).sum(0)
                total_all = np.abs(Cr).sum(0)
                e5.append(dict(cohort=c, species=sp_name, n_genes=int(sup.sum()), mean_negative_mass=float(total_neg.mean()), mean_S_row=float(Cr.sum(0).mean()), median_negative_share=float(np.median(total_neg / np.maximum(total_all, 1e-12)))))
            gs = g.set_index("var_name")
            for pair in [k for k in gs.columns if k.startswith("E_")]:
                e5.append(dict(cohort=c, species=pair, n_genes=int(gs[pair].notna().sum()), median_E=float(gs[pair].median())))
    pd.DataFrame(e1).to_csv(out / "e_b1.tsv", sep="\t", index=False)
    pd.DataFrame(e2).to_csv(out / "e_b2.tsv", sep="\t", index=False)
    pd.DataFrame(e3).to_csv(out / "e_b3.tsv", sep="\t", index=False)
    pd.DataFrame(e4).to_csv(out / "e_b4.tsv", sep="\t", index=False)
    if e4d:
        pd.concat(e4d).to_csv(out / "e_b4_pairs.tsv", sep="\t", index=False)
    pd.DataFrame(e5).to_csv(out / "e_b5.tsv", sep="\t", index=False)
    rows = []
    cs = list(tabs)
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            x = tabs[a].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
            y = tabs[b].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
            ix = x.index.intersection(y.index)
            sup = x.loc[ix, "count_support"].to_numpy() & y.loc[ix, "count_support"].to_numpy()
            rows.append(dict(a=a, b=b, n=len(ix), n_support=int(sup.sum()), rho_D=float(spearmanr(x.loc[ix, "D"], y.loc[ix, "D"])[0]), rho_D_support=float(spearmanr(x.loc[ix, "D"][sup], y.loc[ix, "D"][sup])[0]), rho_S_support=float(spearmanr(x.loc[ix, "S_ema"][sup], y.loc[ix, "S_ema"][sup])[0])))
    pd.DataFrame(rows).to_csv(out / "e_b6.tsv", sep="\t", index=False)
    ase_test(n_perm, n_boot)


def ase_test(n_perm: int, n_boot: int) -> None:
    """E-B7, plan section 13a. Hybrid allele-specific reference (Wang, Starr, Fraser 2024; Agoglia 2021; Y6 gold table): cis level change human vs chimpanzee, non-atlas.
    Targets, oriented so that positive means a smaller cis change: -abs(a_CS) (cortical spheroid), -abs(a_MN) (motor neuron), and P_cis. Join by human symbol."""
    rng = np.random.default_rng(SEED + 7)
    out = CF.RES / "cortex"
    ase = pd.read_parquet(gene_gold_path()).reset_index()
    lvl = pd.read_parquet(GOLD / "DLPFC_chimpanzee_level.parquet").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    rows, signed = [], []
    for c in ("brain_dlpfc_hchimp", "brain_dlpfc"):
        if not (H1 / "genes" / f"{c}.parquet").exists():
            continue
        g = genes(c)
        u = g[g["symbol"].notna() & ~g["symbol"].duplicated(keep=False)].merge(ase.drop_duplicates("symbol"), on="symbol", how="left")
        u["level_gap"] = -lvl["level_gap"].reindex(u["hsapiens_gene_id"]).to_numpy()
        u["signed_gap"] = lvl["signed_gap"].reindex(u["hsapiens_gene_id"]).to_numpy()
        for tname, tv in (("abs_a_CS", -u["a_CS"].abs()), ("abs_a_MN", -u["a_MN"].abs()), ("P_cis", u["P_cis"])):
            v = u.assign(_t=tv)
            rows += rho_table(v, "_t", ["D", "S_ema", "B1", "level_gap", "pi_raw"], n_perm, n_boot, rng, dict(cohort=c, target=tname))
        for col in ("a_CS", "a_MN"):
            a = u.dropna(subset=[col, "signed_gap"])
            signed.append(dict(cohort=c, context=col, n=len(a), spearman_signed_gap=float(spearmanr(a[col], a["signed_gap"])[0])))
    pd.DataFrame(rows).to_csv(out / "e_b7.tsv", sep="\t", index=False)
    pd.DataFrame(signed).to_csv(out / "e_b7_signed.tsv", sep="\t", index=False)


def gene_gold_path():
    return gene_ids.REPO_ROOT / "results" / "revision" / "Y6" / "gold" / "ase_reference.parquet"


# ------------------------------------------------------------------------------------------------ disease cases
def h_d1(case: str, model: str, types: list[str], n_perm: int, n_boot: int) -> pd.DataFrame:
    """rho_p(score, Q | C, abs l^h, SE(m)); l^h = human MLE, m = mouse fixed-scale, Q = sign(l^h) m; genes detected >= 5% in both species."""
    rng = np.random.default_rng(SEED + 1)
    de = CF.RES / case / "de"
    g = genes(model)
    tab = gene_ids.load_table(model)
    ids = pd.Index(g["var_name"]).str.replace("ortho_", "", regex=False).astype(int)
    g["mouse_id"] = tab.loc[ids, "mmusculus_gene_id"].to_numpy()
    rows = []
    for t in types:
        fh, fm = de / f"human_primary_{t}.tsv", de / f"mouse_primary_{t}.tsv"
        if not (fh.exists() and fm.exists()):
            continue
        h = pd.read_csv(fh, sep="\t").dropna(subset=["log2fc_mle", "gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
        m = pd.read_csv(fm, sep="\t").dropna(subset=["log2fc_fixed", "gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
        u = g.copy()
        u["l"] = h["log2fc_mle"].reindex(u["hsapiens_gene_id"]).to_numpy()
        u["det_h"] = h["det_frac"].reindex(u["hsapiens_gene_id"]).to_numpy()
        u["padj_h"] = h["padj"].reindex(u["hsapiens_gene_id"]).to_numpy()
        u["m"] = m["log2fc_fixed"].reindex(u["mouse_id"]).to_numpy()
        u["se_m"] = m["lfcse_fixed"].reindex(u["mouse_id"]).to_numpy()
        u["det_m"] = m["det_frac"].reindex(u["mouse_id"]).to_numpy()
        u = u[(u.det_h >= 0.05) & (u.det_m >= 0.05)].dropna(subset=["l", "m", "se_m"]).copy()
        u["abs_l"] = u["l"].abs()
        u["Q"] = np.sign(u["l"]) * u["m"]
        r = rho_table(u, "Q", ["D", "S_ema", "B1", "pi_raw"], n_perm, n_boot, rng, dict(case=case, cell_type=t, n_human_de=int((u.padj_h < 0.1).sum()), spearman_l_m=float(spearmanr(u.l, u.m)[0]), sign_agree=float((np.sign(u.l) == np.sign(u.m)).mean())), extra_nuis=("abs_l", "se_m"))
        rows += r
    t = pd.DataFrame(rows)
    t.to_csv(CF.RES / case / "h_d1.tsv", sep="\t", index=False)
    return t


def ref_de(case: str, ref_key: str, types_map: dict) -> pd.DataFrame:
    """Named genes in the DE of each species, no Chimera: sign, log2FC, padj."""
    de = CF.RES / case / "de"
    tab = pd.read_csv(CF.BUILD / "objects" / "ortholog_table_hm_all.csv")
    h2m = dict(zip(tab["hsapiens_gene_name"], tab["mmusculus_gene_id"]))
    h2h = dict(zip(tab["hsapiens_gene_name"], tab["hsapiens_gene_id"]))
    rows = []
    for x in REF[REF.cohort == ref_key].itertuples():
        t = types_map.get(x.cell_type)
        if t is None:
            continue
        h = pd.read_csv(de / f"human_primary_{t}.tsv", sep="\t").drop_duplicates("gene_id").set_index("gene_id")
        m = pd.read_csv(de / f"mouse_primary_{t}.tsv", sep="\t").drop_duplicates("gene_id").set_index("gene_id")
        hid, mid = h2h.get(x.gene), h2m.get(x.gene)
        rows.append(dict(set=x.set, gene=x.gene, cell_type=t, direction=x.direction, l_h=h["log2fc_mle"].get(hid, np.nan), padj_h=h["padj"].get(hid, np.nan), det_h=h["det_frac"].get(hid, np.nan), m=m["log2fc_fixed"].get(mid, np.nan), m_mle=m["log2fc_mle"].get(mid, np.nan), padj_m=m["padj"].get(mid, np.nan), det_m=m["det_frac"].get(mid, np.nan)))
    t = pd.DataFrame(rows)
    t.to_csv(CF.RES / case / "reference_de.tsv", sep="\t", index=False)
    return t


def ad_case(n_perm: int, n_boot: int) -> None:
    h_d1("ad", "brain_m1_hmouse", ["microglia", "astrocyte", "oligodendrocyte"], n_perm, n_boot)
    ref_de("ad", "ad", {"Micro": "microglia", "Oligo": "oligodendrocyte", "Astro": "astrocyte"})
    rng = np.random.default_rng(SEED + 2)
    g = genes("brain_m1_hmouse")
    r = REF[REF.cohort == "ad"]
    div = set(r[r.set.isin(["homeostatic_opposite", "dam_mouse_up"])].gene)
    sh = set(r[r.set == "dam_shared"].gene)
    rows = auroc_row(g, div, ["D", "S_ema", "B1"], rng, {"contrast": "divergent_vs_vocabulary"}) + auroc_row(g, div, ["D", "S_ema", "B1"], rng, {"contrast": "divergent_vs_vocabulary"}, tau=True)
    rows += auroc_row(g, div, ["D", "S_ema", "B1"], rng, {"contrast": "divergent_vs_dam_shared"}, background=sh)
    t = pd.DataFrame(rows)
    fw = fw_tables("brain_m1_hmouse")
    idx = pd.Index(fw["var_names"])
    gi = g.set_index("var_name")
    sym = gi["symbol"].reindex(idx).to_numpy()
    rr = []
    labs = fw["labels"]
    for x in r[r.cell_type == "Micro"].itertuples():
        m = np.flatnonzero(sym == x.gene)
        if len(m) != 1:
            continue
        row = dict(set=x.set, gene=x.gene, D=float(gi.loc[idx[m[0]], "D"]), S_ema=float(gi.loc[idx[m[0]], "S_ema"]), strict=bool(gi.loc[idx[m[0]], "strict_support"]), B1=float(gi.loc[idx[m[0]], "B1"]))
        for s, nm in enumerate(fw["species"]):
            k = np.flatnonzero((fw["sp"] == s) & (labs == "Micro"))
            row[f"S_micro_{nm}"] = float(fw["S"][k[0], m[0]]) if len(k) else np.nan
        rr.append(row)
    pd.DataFrame(rr).to_csv(CF.RES / "ad" / "e_a4_genes.tsv", sep="\t", index=False)
    t.to_csv(CF.RES / "ad" / "e_a4.tsv", sep="\t", index=False)


def muscle_case(n_perm: int, n_boot: int) -> None:
    h_d1("muscle", "muscle_ageing", ["fb", "musc", "smc", "pericyte", "capec", "artec", "venec", "macrophage"], n_perm, n_boot)
    ref_de("muscle", "muscle", {"FB": "fb", "MuSC": "musc", "Macrophage": "macrophage", "CapEC": "capec"})


def lung_case(n_perm: int, n_boot: int) -> None:
    h_d1("lung", "lung_healthy", ["macrophage", "at2", "endothelial", "t", "dc"], n_perm, n_boot)
    ref_de("lung", "lung", {"AT2": "at2", "Macrophage": "macrophage"})


# ------------------------------------------------------------------------------------------------ tumour
def tumour(n_boot: int) -> None:
    rng = np.random.default_rng(SEED + 3)
    out = CF.RES / "tumour_zilionis"
    g = genes("tumour_zilionis")
    fw = fw_tables("tumour_zilionis")
    idx = pd.Index(fw["var_names"])
    gi = g.set_index("var_name").reindex(idx)
    sup = gi["count_support"].to_numpy(bool)
    labs = fw["labels"]
    types = sorted(set(labs))
    # A_k: mean over supported genes of the contribution per cell in k, pooled over species = mean over species rows of S_gsk.
    A = {k: fw["S"][labs == k].mean(0) for k in types}
    Am = pd.DataFrame(A, index=idx)
    rows = []
    genes_idx = np.flatnonzero(sup)
    blocks = gi["hsapiens_gene_id"].to_numpy()
    for a, b in (("DC", "Macrophage"), ("Monocyte", "Macrophage"), ("Neutrophil", "Macrophage")):
        d = (Am[a] - Am[b]).to_numpy()[genes_idx]
        draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
        rows.append(dict(contrast=f"{a}-{b}", mean_diff=float(d.mean()), ci_lo=S.pct_ci(draws)[0], ci_hi=S.pct_ci(draws)[1], A_a=float(Am[a].to_numpy()[genes_idx].mean()), A_b=float(Am[b].to_numpy()[genes_idx].mean()), n_genes=len(d)))
    pd.DataFrame(rows).to_csv(out / "e_t1.tsv", sep="\t", index=False)
    Am.mean().rename("A_k").to_csv(out / "e_t1_A_k.tsv", sep="\t")
    # Baseline: class-level human-mouse Spearman over genes detected in 1%, from the training object; same object for latent mixing.
    a = ad.read_h5ad(CF.h5ad("tumour_zilionis"))
    X = a.X.tocsr().astype(np.float64)
    sp_ = a.obs["species"].astype(str).to_numpy()
    ct = a.obs["cell_ontology_class"].astype(str).to_numpy()
    lib = np.asarray(X.sum(1)).ravel()
    base = []
    for k in types:
        mh, mm = (sp_ == "Human") & (ct == k), (sp_ == "Mouse") & (ct == k)
        if mh.sum() < 20 or mm.sum() < 20:
            continue
        ph = np.asarray(X[mh].sum(0)).ravel() / lib[mh].sum()
        pm = np.asarray(X[mm].sum(0)).ravel() / lib[mm].sum()
        dh, dm = np.asarray((X[mh] > 0).mean(0)).ravel(), np.asarray((X[mm] > 0).mean(0)).ravel()
        ok = (dh >= 0.01) & (dm >= 0.01)
        base.append(dict(cell_type=k, n_human=int(mh.sum()), n_mouse=int(mm.sum()), n_genes=int(ok.sum()), spearman_class_profile=float(spearmanr(ph[ok], pm[ok])[0])))
    pd.DataFrame(base).to_csv(out / "e_t1_baseline.tsv", sep="\t", index=False)
    # E-T2: within-class species mixing in z: kNN entropy per class over species, kept seeds; scVI beside when cached.
    from sklearn.neighbors import NearestNeighbors

    runs = pd.read_csv(H1 / "runs.tsv", sep="\t")
    kept = runs[(runs.cohort == "tumour_zilionis") & runs.kept].run.tolist()
    codes = pd.factorize(sp_, sort=True)[0]
    mix = []
    lat = {}
    for run in kept:
        with np.load(gene_ids.REPO_ROOT / "training" / "runs" / run / "final_state.npz", allow_pickle=True) as z:
            lat[run] = np.asarray(z["z_mean"], float)
    sc = CF.RES / "tumour_zilionis" / "scvi_latent.npy"
    if sc.exists():
        lat["scVI"] = np.load(sc)
    for name, z in lat.items():
        for k in types:
            m = ct == k
            if m.sum() < 30 or len(set(codes[m])) < 2:
                continue
            idxs = NearestNeighbors(n_neighbors=16).fit(z[m]).kneighbors(z[m])[1][:, 1:]
            p = np.stack([(codes[m][idxs] == c).mean(1) for c in (0, 1)], 1)
            h = -(p * np.log(np.where(p > 0, p, 1.0))).sum(1) / np.log(2)
            mix.append(dict(run=name, cell_type=k, n=int(m.sum()), mean_entropy=float(h.mean())))
    pd.DataFrame(mix).to_csv(out / "e_t2.tsv", sep="\t", index=False)
    # E-T3: named sets, stratified AUROC on D, S_ema, B1 and on the class-row contribution for the named class.
    r = REF[REF.cohort == "tumour"]
    con = set(r[r.set.str.startswith("congruent")].gene)
    dv = set(r[r.set == "divergent_mac"].gene)
    rows = auroc_row(g, con, ["D", "S_ema", "B1"], rng, {"contrast": "congruent_vs_divergent"}, background=dv) + auroc_row(g, con, ["D", "S_ema", "B1"], rng, {"contrast": "congruent_vs_vocabulary"}) + auroc_row(g, dv, ["D", "S_ema", "B1"], rng, {"contrast": "divergent_vs_vocabulary"})
    pd.DataFrame(rows).to_csv(out / "e_t3.tsv", sep="\t", index=False)
    named = []
    for x in r.itertuples():
        m = gi[gi["symbol"] == x.gene]
        if len(m) != 1:
            continue
        v = m.index[0]
        row = dict(set=x.set, gene=x.gene, named_type=x.cell_type, D=float(m.D.iloc[0]), S_ema=float(m.S_ema.iloc[0]), strict=bool(m.strict_support.iloc[0]), B1=float(m.B1.iloc[0]))
        for s, nm in enumerate(fw["species"]):
            k = np.flatnonzero((fw["sp"] == s) & (labs == {"Mono": "Monocyte", "cDC": "DC"}.get(x.cell_type, x.cell_type)))
            row[f"S_class_{nm}"] = float(fw["S"][k[0], idx.get_loc(v)]) if len(k) else np.nan
        named.append(row)
    pd.DataFrame(named).to_csv(out / "e_t3_genes.tsv", sep="\t", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["brain", "ad", "lung", "tumour", "muscle"])
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    {"brain": lambda: brain(a.n_perm, a.n_boot), "ad": lambda: ad_case(a.n_perm, a.n_boot), "lung": lambda: lung_case(a.n_perm, a.n_boot), "tumour": lambda: tumour(a.n_boot), "muscle": lambda: muscle_case(a.n_perm, a.n_boot)}[a.which]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
