"""Use 2: choosing a model species, rules.py USE 2.

    PYTHONPATH=$PWD python -m analysis.tasks.Y6_uses.use2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.tasks.X4_where_differs import common as CM
from analysis.tasks.X4_where_differs import rules as XR
from analysis.tasks.Y6_uses import rules as RU
from analysis.tasks.Y6_uses.build_rows import forward_files, species_margin

CM.N_JOBS = 8
OUT = RU.OUT / "use2"
HALLMARK = RU.ROOT / "data/analysis/functional/Enrichr.MSigDB_Hallmark_2020.server_response.txt"


def wide(fit):
    w = pd.read_parquet(RU.OUT / "wide.parquet")
    return w[w.fit == fit].copy()


def splits(g):
    ch = pd.read_csv(RU.RES / "X1/genes/chromosome_ensembl110.csv", dtype={"chrom": str}).set_index("hsapiens_gene_id")["chrom"]
    a = pd.to_numeric(g.hsapiens_gene_id.map(ch), errors="coerce")
    return a.isin(range(1, 23, 2)).to_numpy(), a.isin(range(2, 23, 2)).to_numpy()


def sets():
    """Hallmark sets as gene symbols (Enrichr gmt: ``name, empty, gene, gene, ...``)."""
    out = {}
    for line in HALLMARK.read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            out[p[0]] = {x.split(",")[0] for x in p[2:] if x}
    return out


def pick(g, gold, scores, seed, draws=1000):
    """Pick accuracy on clear cases, all scores on the same genes, block CI. Median-centered in the fit."""
    m = g[[gold] + scores + ["block"]].replace([np.inf, -np.inf], np.nan).dropna()
    gc = m[gold] - m[gold].median()
    clear = gc.abs() > gc.abs().median()
    m, gc = m[clear], gc[clear]
    out = []
    blocks = m.block.to_numpy()
    ub, inv = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = [rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)))[inv].astype(float) for _ in range(draws)]
    point = {}
    B = {}
    for s in scores:
        sc = m[s] - g[s].median()
        hit = (np.sign(sc) == np.sign(gc)).to_numpy(float)
        point[s] = hit.mean()
        B[s] = np.array([(w * hit).sum() / w.sum() for w in W])
    for s in scores:
        lo, hi = np.percentile(B[s], [2.5, 97.5])
        out.append(dict(score=s, n_clear=len(m), pick_acc=point[s], lo=lo, hi=hi))
    return pd.DataFrame(out), B


def pathway(g, gold, scores, covs, seed, symbols):
    rs = {c: CM.residual_rank(g, c, covs) for c in [gold] + scores}
    H = sets()
    rows = []
    for n, mem in H.items():
        ix = symbols.isin(mem).to_numpy()
        vals = {c: rs[c][ix] for c in rs}
        ok = pd.concat(vals, axis=1).notna().all(axis=1)
        if ok.sum() < 15:
            continue
        rows.append(dict(set=n, n=int(ok.sum()), **{c: vals[c][ok].mean() for c in vals}))
    T = pd.DataFrame(rows)
    out = []
    if len(T) < 10:
        return T, pd.DataFrame()
    rng = np.random.default_rng(seed)
    for s in scores:
        est = spearmanr(T[s], T[gold])[0]
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(T), len(T))
            bs.append(spearmanr(T[s].to_numpy()[i], T[gold].to_numpy()[i])[0])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out.append(dict(score=s, n_sets=len(T), rho_sets=est, lo=lo, hi=hi))
    return T, pd.DataFrame(out)


def seeds_rho(fit, gcontrast, a, b, g, gold, covs, rows):
    """Chimera contrast on each stored seed, 200 draws, seed range."""
    out = []
    spec = RU.FITS[fit]
    for f in forward_files(fit, spec):
        m, _, _ = species_margin([f])
        gg = g.copy()
        gg["seed_c"] = (m[a] - m[b]).reindex(gg.var_name).to_numpy()
        r = CM.rho_unit(gg, gold, covs, ["seed_c"], rows=rows, draws=200, label=dict(file=str(f).split("/")[-1]))
        out.append(r[0])
    return pd.DataFrame(out)


def run(name, g, gold, covs, scores, ref, rows, tag):
    res = []
    odd, even = splits(g)
    for split, mask in (("all", rows), ("odd", rows & odd), ("even", rows & even)):
        res += CM.rho_unit(g, gold, covs, scores, rows=mask, ref=ref, draws=RU.DRAWS_RHO, label=dict(test=name, split=split, tag=tag))
    return pd.DataFrame(res)


def testis():
    g = CM.load("test_3sp").copy()
    mt = pd.read_parquet(RU.RES / "X7/golds/murat_tiers.parquet")
    g["Gt"] = g.hsapiens_gene_id.map(mt.Gt)
    pe = pd.read_parquet(RU.RES / "X7/baselines/test_3sp__pair_E.parquet").set_index("var_name")
    own = pd.read_parquet(RU.RES / "X7/baselines/test_3sp__own_margin.parquet").set_index("var_name")
    sd = pd.read_parquet(CM.CACHE / "Sdir__test_3sp.parquet").set_index("var_name")
    tab = pd.read_csv(RU.ROOT / "data/processed/ortholog_table_testis.csv").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    W = wide("test_3sp")
    g["S_mac"] = W["S_mmulatta"].reindex(g.var_name).to_numpy()
    g["S_mouse"] = W["S_mmusculus"].reindex(g.var_name).to_numpy()
    g["C_chimera"] = g.S_mac - g.S_mouse
    g["C_B1"] = (pe.E_Human_Macaque - pe.E_Human_Mouse).reindex(g.var_name).to_numpy()
    g["C_id"] = (g.hsapiens_gene_id.map(tab.mmulatta_identity) - g.hsapiens_gene_id.map(tab.mmusculus_identity)).to_numpy()
    g["C_own"] = (own.Sown_Macaque - own.Sown_Mouse).reindex(g.var_name).to_numpy()
    g["C_dir"] = (sd.Cspdir_Macaque - sd.Cspdir_Mouse).reindex(g.var_name).to_numpy()
    Hm = wide("test_hmac").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    Ho = wide("test_hmouse").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["C_pairfits"] = (g.hsapiens_gene_id.map(Hm.S_mmulatta) - g.hsapiens_gene_id.map(Ho.S_mmusculus)).to_numpy()
    g["C_Dpair"] = (g.hsapiens_gene_id.map(Hm.D) - g.hsapiens_gene_id.map(Ho.D)).to_numpy()
    u = (np.abs(g.S.to_numpy()) >= 1e-6) & g.nonzero_all_species.eq(True).to_numpy()
    covs = XR.C_PAT + ["H4_nsp"]
    scores = ["C_chimera", "C_pairfits", "C_Dpair", "C_B1", "C_id", "C_own", "C_dir"]
    res = run("testis_mac_vs_mouse", g, "Gt", covs, scores, "C_chimera", u, "two species, held-out species gold")
    pk, _ = pick(g[u], "Gt", scores, RU.SEED)
    pk["test"] = "testis_mac_vs_mouse"
    T, pw = pathway(g[u], "Gt", scores, covs, RU.SEED, g.symbol[u])
    pw["test"] = "testis_mac_vs_mouse"
    sr = seeds_rho("test_3sp", None, "Macaque", "Mouse", g, "Gt", covs, u)
    sr["test"] = "testis_mac_vs_mouse"
    g[u][["hsapiens_gene_id", "symbol", "block", "Gt"] + scores].to_parquet(OUT / "genes_testis.parquet")
    return res, pk, pw, sr, T


def liver():
    g = CM.load("liv_4sp").copy()
    gold = pd.read_parquet(RU.OUT / "gold/liver_primate_gold.parquet")
    g["G_lemur"] = g.hsapiens_gene_id.map(gold.G_lemur)
    g["G_H3_5"] = g.hsapiens_gene_id.map(gold.G_H3_5)
    g["G_prim"] = g.G_lemur - g.G_H3_5
    pe = pd.read_parquet(RU.RES / "X7/baselines/liv_4sp__pair_E.parquet").set_index("var_name")
    own = pd.read_parquet(RU.RES / "X7/baselines/liv_4sp__own_margin.parquet").set_index("var_name")
    sd = pd.read_parquet(CM.CACHE / "Sdir__liv_4sp.parquet").set_index("var_name")
    tab = pd.read_csv(RU.ROOT / "data/processed/ortholog_table.csv").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    W = wide("liv_4sp")
    g["S_mac"] = W["S_mmulatta"].reindex(g.var_name).to_numpy()
    g["S_mouse"] = W["S_mmusculus"].reindex(g.var_name).to_numpy()
    g["C_chimera"] = g.S_mac - g.S_mouse
    g["C_B1"] = (pe.E_Human_Monkey - pe.E_Human_Mouse).reindex(g.var_name).to_numpy()
    g["E_mouse"] = pe.E_Human_Mouse.reindex(g.var_name).to_numpy()
    g["id_mouse"] = g.hsapiens_gene_id.map(tab.mmusculus_identity).to_numpy()
    g["C_id"] = (g.hsapiens_gene_id.map(tab.mmulatta_identity) - g.hsapiens_gene_id.map(tab.mmusculus_identity)).to_numpy()
    g["C_own"] = (own.Sown_Monkey - own.Sown_Mouse).reindex(g.var_name).to_numpy()
    g["C_dir"] = (sd.Cspdir_Monkey - sd.Cspdir_Mouse).reindex(g.var_name).to_numpy()
    Hm = wide("liv_hmonkey").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    Ho = wide("liv_hmouse").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["C_pairfits"] = (g.hsapiens_gene_id.map(Hm.S_mmulatta) - g.hsapiens_gene_id.map(Ho.S_mmusculus)).to_numpy()
    g["C_Dpair"] = (g.hsapiens_gene_id.map(Hm.D) - g.hsapiens_gene_id.map(Ho.D)).to_numpy()
    u = (np.abs(g.S.to_numpy()) >= 1e-6) & g.nonzero_all_species.eq(True).to_numpy()
    covs = XR.C_PAT
    scores = ["C_chimera", "C_pairfits", "C_Dpair", "C_B1", "C_id", "C_own", "C_dir"]
    res = [run("liver_mac_vs_mouse", g, "G_prim", covs, scores, "C_chimera", u, "two species, held-out primate")]
    res.append(run("liver_mouse_row", g, "G_H3_5", covs, ["S_mouse", "E_mouse", "id_mouse"], "S_mouse", u, "mouse row against TS vs TMS, 5 types"))
    res.append(run("liver_mouse_row_H3", g, "G_H3", covs, ["S_mouse", "E_mouse", "id_mouse"], "S_mouse", u, "mouse row against G_H3, X7 reproduction"))
    pk, _ = pick(g[u], "G_prim", scores, RU.SEED)
    pk["test"] = "liver_mac_vs_mouse"
    T, pw = pathway(g[u], "G_prim", scores, covs, RU.SEED, g.symbol[u])
    pw["test"] = "liver_mac_vs_mouse"
    sr = seeds_rho("liv_4sp", None, "Monkey", "Mouse", g, "G_prim", covs, u)
    sr["test"] = "liver_mac_vs_mouse"
    g[u][["hsapiens_gene_id", "symbol", "block", "G_prim", "G_lemur", "G_H3_5"] + scores].to_parquet(OUT / "genes_liver.parquet")
    return pd.concat(res), pk, pw, sr, T


def marrow():
    g = pd.read_parquet(RU.RES / "P12/readout/bm_genes.parquet")
    W = wide("bone_marrow")
    g["S_mouse"] = W["S_mmusculus"].reindex(g.var_name).to_numpy()
    g["S_lemur"] = W["S_mmurinus"].reindex(g.var_name).to_numpy()
    tab = pd.read_csv(RU.ROOT / "data/analysis/bone_marrow/ortholog_table_bone_marrow.csv").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["id_mouse"] = g.hsapiens_gene_id.map(tab.mmusculus_identity).to_numpy()
    g["id_lemur"] = g.hsapiens_gene_id.map(tab.mmurinus_identity).to_numpy()
    g["C_chimera"] = g.S_mouse - g.S_lemur
    g["C_B1"] = g.E_human_mouse - g.E_human_mouse_lemur
    g["C_id"] = g.id_mouse - g.id_lemur
    g["E_mouse"] = g.E_human_mouse
    s = pd.read_csv(RU.ROOT / "data/analysis/bone_marrow/shay_coe.tsv", sep="\t").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["COE"] = g.hsapiens_gene_id.map(s.coe)
    g["var_h"] = g.hsapiens_gene_id.map(s.human_log2_range_lineage_means)
    g["var_m"] = g.hsapiens_gene_id.map(s.mouse_log2_range_lineage_means)
    g["tau_shared"] = g.tau
    covs = XR.C_PAT
    u = (np.abs(g.S_ema.to_numpy()) >= 1e-6)
    scores = ["S_mouse", "E_mouse", "id_mouse", "C_chimera", "C_B1", "C_id"]
    res = [run("marrow_mouse_row", g, "COE", covs, ["S_mouse", "E_mouse", "id_mouse"], "S_mouse", u, "mouse row against Shay COE"),
           run("marrow_mouse_vs_lemur", g, "COE", covs, ["C_chimera", "C_B1", "C_id"], "C_chimera", u, "one-sided: gold has mouse only")]
    hv = u & (g.var_h >= 2).to_numpy() & (g.var_m >= 2).to_numpy()
    res.append(run("marrow_mouse_row_matched", g, "COE", covs, ["S_mouse", "E_mouse", "id_mouse"], "S_mouse", hv, "4-fold in both species"))
    pk, _ = pick(g[u], "COE", ["C_chimera", "C_B1", "C_id", "S_mouse", "E_mouse", "id_mouse"], RU.SEED)
    pk["test"] = "marrow"
    return pd.concat(res), pk, g


def eye():
    res = []
    full = pd.read_parquet(RU.RES / "T11/readout/genes_hmfas.parquet")
    b1f = full.set_index("hsapiens_gene_id")
    mm = pd.read_parquet(RU.RES / "T11/readout/genes_hmmul.parquet")
    b1m = mm.set_index("hsapiens_gene_id")
    tab = pd.read_csv(RU.ROOT / "data/analysis/van_zyl/ortholog_table_van_zyl.csv").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    picks = []
    for name, base, other, sp, oth in (("eye_mulatta_to_fas", mm, b1f, "mmulatta", "mfascicularis"), ("eye_fas_to_mulatta", full, b1m, "mfascicularis", "mmulatta")):
        g = base.copy()
        fit = "vz_hmmul" if sp == "mmulatta" else "vz_hmfas"
        W = wide(fit).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
        g["S_row"] = g.hsapiens_gene_id.map(W[f"S_{sp}"])
        g["E_own"] = g.B1_raw
        g["id_own"] = g.hsapiens_gene_id.map(tab[f"{sp}_identity"])
        g["gold"] = g.hsapiens_gene_id.map(other.B1_raw)
        g["tau_shared"] = g.tau
        g["block"] = g.block
        u = (np.abs(g.S_ema.to_numpy()) >= 1e-6)
        res.append(run(name, g, "gold", XR.C_PAT, ["S_row", "S_ema", "D", "E_own", "id_own"], "S_row", u, "held-out sister species, same study; human cells shared"))
        pk, _ = pick(g[u], "gold", ["S_row", "S_ema", "D", "E_own", "id_own"], RU.SEED)
        pk["test"] = name
        picks.append(pk)
    return pd.concat(res), pd.concat(picks)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rt, pt, wt, st, Tt = testis()
    rl, pl, wl, sl, Tl = liver()
    rm, pm, gm = marrow()
    re_, pe_ = eye()
    pd.concat([rt, rl, rm, re_]).to_csv(OUT / "rho.csv", index=False)
    pd.concat([pt, pl, pm, pe_]).to_csv(OUT / "pick.csv", index=False)
    pd.concat([wt, wl]).to_csv(OUT / "pathway.csv", index=False)
    pd.concat([st, sl]).to_csv(OUT / "seeds.csv", index=False)
    Tt.to_csv(OUT / "pathway_sets_testis.csv", index=False)
    Tl.to_csv(OUT / "pathway_sets_liver.csv", index=False)


if __name__ == "__main__":
    main()
