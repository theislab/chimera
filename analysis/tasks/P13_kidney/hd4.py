"""H-D4 program transfer for the kidney case (pre-specified analysis plan, section 8c), AKI vs IRI. No Chimera readout.

Per DE-able type (``results/analysis/P13/de/de_meta.json``):
  * universe: genes tested in the human primary fit (AKI vs healthy DE half, multiome) and, through
    the kidney human-mouse 1:1 table (both Ensembl 110 Compara dumps), in the mouse primary fit
    (IRI vs Sham, ~ stage + condition); human symbols;
  * preranked GSEA on fixed-scale apeGLM log2FC (``log2fc_fixed``, prior scale 1; the adaptive
    prior collapses thin fits, see the liver MASH case, analysis/tasks/T7_3_mash), weighted KS (p = 1),
    10,000 random gene sets of the same size per set size (gene-set permutation), NES = ES / mean
    |null ES| of the same sign (the implementation of the islet case, analysis/tasks/T7_9_islet,
    copied here unchanged);
  * sets: MSigDB Hallmark 2020 and Reactome 2024 (Enrichr GMT files), 15-500 genes in universe;
  * human p two-sided (same-sign null), BH over sets per type; mouse one-sided in the human
    direction, BH over the human-significant sets of that type;
  * transfers = human q < 0.05 and mouse oriented q < 0.05.

Gene-set permutation ignores correlation between genes and is anti-conservative (see T7_3_mash), so calls
are judged against ``--null``: the same calls on the permuted-label fits of ``de.py --perm``
(permuted human vs observed mouse, and observed human vs permuted mouse, 2,000 gene-set
permutations), per type with ``--null --null-type <t>`` (``hd4_null_<t>.tsv``) and combined with
``--null-combine`` into ``hd4_null.tsv`` and ``hd4_null_summary.tsv``.

Also written (Family 3, checks the DE, not Chimera): per type, Spearman(human log2FC, mouse
log2FC) on genes detected in >= 5% of nuclei of the type in both species' DE units.

    PYTHONPATH=<chimera repo> python analysis/tasks/P13_kidney/hd4.py [--n-perm 10000]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from analysis.common import gene_ids

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis" / "P13"
DE = RES / "de"
OUT = RES / "hd4"
FUNC = ROOT / "data" / "analysis" / "functional"
SETS = {"hallmark": FUNC / "Enrichr.MSigDB_Hallmark_2020.server_response.txt",
        "reactome": FUNC / "Enrichr.Reactome_Pathways_2024.server_response.txt"}
MIN_SIZE, MAX_SIZE = 15, 500
SEED = 20261101
DET = 0.05
N_NULL = 200  # 20 at first; raised for precision only (same design)


def read_gmt(path) -> dict:
    out = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) > 2:
            out[p[0]] = sorted({g for g in p[2:] if g})
    return out


def es_matrix(r_sorted: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """ES for many sets at once. r_sorted: weights sorted descending (N,); pos: (S, k) sorted hit positions."""
    N = r_sorted.shape[0]
    k = pos.shape[1]
    w = np.abs(r_sorted)[pos]                       # (S, k)
    cw = np.cumsum(w, axis=1) / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    miss_before = pos - np.arange(k)[None, :]       # misses strictly before each hit
    pm = 1.0 / (N - k)
    top = cw - miss_before * pm                     # running sum just after each hit
    bottom = (cw - w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)) - miss_before * pm  # just before
    mx, mn = top.max(axis=1), bottom.min(axis=1)
    return np.where(np.abs(mx) >= np.abs(mn), mx, mn)


def gsea(ranks: pd.Series, sets: dict, n_perm: int, seed: int) -> pd.DataFrame:
    r = ranks.sort_values(ascending=False)
    genes = r.index.to_numpy()
    rv = r.to_numpy(dtype=float)
    idx = {g: i for i, g in enumerate(genes)}
    N = len(genes)
    rng = np.random.default_rng(seed)
    # n_perm random permutations of the N ranks; the first k columns of each row are a uniform
    # random k-subset without replacement, so one matrix serves every set size
    perms = np.empty((n_perm, N), dtype=np.int32)
    for i in range(0, n_perm, 500):
        perms[i:i + 500] = np.argsort(rng.random((min(500, n_perm - i), N)), axis=1)
    nulls: dict = {}
    rows = []
    for name, members in sets.items():
        hit = np.sort(np.array([idx[g] for g in members if g in idx], dtype=int))
        k = len(hit)
        if k < MIN_SIZE or k > MAX_SIZE:
            continue
        if k not in nulls:
            nulls[k] = es_matrix(rv, np.sort(perms[:, :k], axis=1))
        es = float(es_matrix(rv, hit[None, :])[0])
        nl = nulls[k]
        same = nl[nl >= 0] if es >= 0 else nl[nl < 0]
        nes = es / np.mean(np.abs(same)) if len(same) else np.nan
        p_two = (1 + np.sum(np.abs(same) >= abs(es))) / (1 + len(same))
        p_up = (1 + np.sum(nl >= es)) / (1 + len(nl))
        p_dn = (1 + np.sum(nl <= es)) / (1 + len(nl))
        rows.append(dict(set=name, size=k, es=es, nes=nes, p=p_two, p_up=p_up, p_down=p_dn))
    return pd.DataFrame(rows)


def one2one() -> pd.DataFrame:
    t = gene_ids.load_table("kidney").reset_index()
    return t[["hsapiens_gene_id", "mmusculus_gene_id"]].rename(
        columns={"hsapiens_gene_id": "human_id", "mmusculus_gene_id": "mouse_id"})


def load_type(t: str, o2o: pd.DataFrame, hc: str = "primary", mc: str = "primary"):
    h = pd.read_csv(DE / f"human_{hc}_{t}.tsv", sep="\t")
    m = pd.read_csv(DE / f"mouse_{mc}_{t}.tsv", sep="\t")
    h = h.dropna(subset=["log2fc_fixed"]).drop_duplicates("gene_id", keep=False)
    m = m.dropna(subset=["log2fc_fixed"]).drop_duplicates("gene_id", keep=False)
    j = o2o.merge(h, left_on="human_id", right_on="gene_id").merge(
        m, left_on="mouse_id", right_on="gene_id", suffixes=("_h", "_m"))
    return j.drop_duplicates("gene_symbol_h", keep=False)


def calls(gh: pd.DataFrame, gm: pd.DataFrame) -> pd.DataFrame:
    g = gh.merge(gm.drop(columns="size"), on=["library", "set"])
    g["h_q"] = multipletests(g["h_p"], method="fdr_bh")[1]
    g["m_p_oriented"] = np.where(g["h_nes"] >= 0, g["m_p_up"], g["m_p_down"])
    g["m_p_against"] = np.where(g["h_nes"] >= 0, g["m_p_down"], g["m_p_up"])
    sig = g["h_q"] < 0.05
    g["m_q_oriented"] = np.nan
    g["m_q_against"] = np.nan
    if sig.any():
        g.loc[sig, "m_q_oriented"] = multipletests(g.loc[sig, "m_p_oriented"], method="fdr_bh")[1]
        g.loc[sig, "m_q_against"] = multipletests(g.loc[sig, "m_p_against"], method="fdr_bh")[1]
    g["transfers"] = sig & (g["m_q_oriented"] < 0.05)
    g["call"] = np.where(~sig, "human_not_significant", np.where(
        g["transfers"], "transfers", np.where(g["m_q_against"] < 0.05, "opposite", "does_not_transfer")))
    return g


def sides(rk: pd.Series, sets: dict, n_perm: int, seed: int, tag: str) -> pd.DataFrame:
    out = []
    for lib, s in sets.items():
        g = gsea(rk, s, n_perm, seed).add_prefix(f"{tag}_").rename(columns={f"{tag}_set": "set", f"{tag}_size": "size"})
        g.insert(0, "library", lib)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def run_type(args):
    t, sets, n_perm, o2o, hc, mc = args
    j = load_type(t, o2o, hc, mc)
    rh = pd.Series(j["log2fc_fixed_h"].to_numpy(), index=j["gene_symbol_h"].to_numpy())
    rm = pd.Series(j["log2fc_fixed_m"].to_numpy(), index=j["gene_symbol_h"].to_numpy())
    g = calls(sides(rh, sets, n_perm, SEED, "h"), sides(rm, sets, n_perm, SEED + 1, "m"))
    g.insert(0, "cell_type", t)
    uni = j[(j["det_frac_h"] >= DET) & (j["det_frac_m"] >= DET)]
    rho, prho = spearmanr(uni["log2fc_mle_h"], uni["log2fc_fixed_m"])
    return t, g, len(j), dict(cell_type=t, n_genes_det5=int(len(uni)), spearman_lfc=float(rho), p_two_sided=float(prho))


def run_null(args):
    t, sets, n_perm, o2o, chunk, n_chunks = args
    j = load_type(t, o2o)
    hsym = j.set_index("human_id")["gene_symbol_h"]
    msym = j.set_index("mouse_id")["gene_symbol_h"]
    rh = pd.Series(j["log2fc_fixed_h"].to_numpy(), index=j["gene_symbol_h"].to_numpy())
    rm = pd.Series(j["log2fc_fixed_m"].to_numpy(), index=j["gene_symbol_h"].to_numpy())
    gh0, gm0 = sides(rh, sets, n_perm, SEED, "h"), sides(rm, sets, n_perm, SEED + 1, "m")
    rows = []

    def rec(c, kind, i):
        rows.append(dict(cell_type=t, kind=kind, perm=i, n_human_sig=int((c["h_q"] < 0.05).sum()),
                         n_transfers=int((c["call"] == "transfers").sum()), n_opposite=int((c["call"] == "opposite").sum()),
                         nes_spearman=float(spearmanr(c["h_nes"], c["m_nes"], nan_policy="omit")[0])))

    rec(calls(gh0, gm0), "observed", -1)
    for i in range(N_NULL):
        if i % n_chunks != chunk:
            continue
        for sp_, sym, tag, kind in (("human", hsym, "h", "human_permuted"), ("mouse", msym, "m", "mouse_permuted")):
            f = DE / "perm" / f"{sp_}_perm{i:02d}_{t}.tsv"
            if not f.exists():
                continue
            d = pd.read_csv(f, sep="\t", usecols=["gene_id", "log2fc_fixed"]).dropna().drop_duplicates("gene_id", keep=False)
            d = d[d["gene_id"].isin(sym.index)]
            rk = pd.Series(d["log2fc_fixed"].to_numpy(), index=sym.loc[d["gene_id"]].to_numpy())
            g = sides(rk, sets, n_perm, SEED + 2 + i, tag)
            rec(calls(g, gm0) if tag == "h" else calls(gh0, g), kind, i)
    return rows


def summarise_null(tab: pd.DataFrame) -> pd.DataFrame:
    summ = []
    for t, g in tab.groupby("cell_type", sort=False):
        o = g[g.kind == "observed"].iloc[0]
        for kind in ("human_permuted", "mouse_permuted"):
            n = g[g.kind == kind]
            rec = dict(cell_type=t, kind=kind, n_null=len(n))
            for c in ("n_human_sig", "n_transfers", "n_opposite", "nes_spearman"):
                rec.update({f"{c}_obs": o[c], f"{c}_null_median": n[c].median(), f"{c}_null_max": n[c].max(),
                            f"{c}_null_q95": n[c].quantile(0.95),
                            f"{c}_p_ge": float((1 + (n[c] >= o[c]).sum()) / (1 + len(n)))})
            summ.append(rec)
    return pd.DataFrame(summ)


def null_main(sets: dict, o2o: pd.DataFrame, types: list, jobs: int, n_perm: int, one_type: str | None,
              combine: bool, n_chunks: int = 1) -> int:
    if combine:
        tab = pd.concat([pd.read_csv(OUT / f"hd4_null_{t}.tsv", sep="\t") for t in types
                         if (OUT / f"hd4_null_{t}.tsv").exists()], ignore_index=True)
        tab.to_csv(OUT / "hd4_null.tsv", sep="\t", index=False)
        summarise_null(tab).to_csv(OUT / "hd4_null_summary.tsv", sep="\t", index=False)
        (OUT / "DONE_null").write_text("ok\n")
        return 0
    todo = [one_type] if one_type else types
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(run_null, [(t, sets, n_perm, o2o, c, n_chunks) for t in todo for c in range(n_chunks)]))
    for t in todo:
        rows = [r for rr in res for r in rr if r["cell_type"] == t]
        tab = pd.DataFrame(rows)
        tab = pd.concat([tab[tab.kind == "observed"].head(1), tab[tab.kind != "observed"]]).sort_values(["kind", "perm"])
        tab.to_csv(OUT / f"hd4_null_{t}.tsv", sep="\t", index=False)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--human-contrast", default="primary", help="sensitivity runs only; output in hd4_<h>_<m>/")
    p.add_argument("--mouse-contrast", default="primary")
    p.add_argument("--null", action="store_true", help="calls on the permuted-label fits (2,000 gene-set permutations)")
    p.add_argument("--null-type", default=None)
    p.add_argument("--null-combine", action="store_true")
    a = p.parse_args()
    hc, mc = a.human_contrast, a.mouse_contrast
    out = OUT if (hc, mc) == ("primary", "primary") else RES / f"hd4_{hc}_{mc}"
    out.mkdir(parents=True, exist_ok=True)
    types = json.loads((DE / "de_meta.json").read_text())["de_types"]
    types = [t for t in types if (DE / f"human_{hc}_{t}.tsv").exists() and (DE / f"mouse_{mc}_{t}.tsv").exists()]
    sets = {k: read_gmt(v) for k, v in SETS.items()}
    o2o = one2one()
    if a.null:
        return null_main(sets, o2o, types, a.jobs, min(a.n_perm, 2000), a.null_type, a.null_combine, a.jobs)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(run_type, [(t, sets, a.n_perm, o2o, hc, mc) for t in types]))
    allg, summ, conc = [], [], []
    for t, g, n, c in res:
        allg.append(g)
        conc.append(c)
        summ.append(dict(cell_type=t, n_universe=n, n_sets=len(g), n_human_sig=int((g["h_q"] < 0.05).sum()),
                         n_transfer=int(g["transfers"].sum()),
                         n_not_transfer=int((g["call"] == "does_not_transfer").sum()),
                         nes_spearman_all_sets=float(spearmanr(g["h_nes"], g["m_nes"], nan_policy="omit")[0])))
    pd.concat(allg).to_csv(out / "hd4.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(out / "hd4_summary.tsv", sep="\t", index=False)
    pd.DataFrame(conc).to_csv(out / "lfc_concordance.tsv", sep="\t", index=False)
    (out / "DONE").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
