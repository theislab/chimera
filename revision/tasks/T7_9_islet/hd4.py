"""T7.9: H-D4 program transfer and the Hrovatin anchor (plan section 8b). No Chimera readout.

H-D4 (Family 2), per DE-able type:
  * universe: genes tested in the human T2D vs ND fit and, through an Ensembl Compara 110
    ``ortholog_one2one`` pair (both dumps), in the mouse primary (joint) fit; human symbols;
  * preranked GSEA on apeglm log2FC, weighted KS (p = 1), 10,000 random gene sets of the same
    size per set size (gene-set permutation), NES = ES / mean |null ES| of the same sign;
  * sets: MSigDB Hallmark 2020 and Reactome 2024 (Enrichr GMT files), 15-500 genes in universe;
  * human p two-sided (same-sign null), BH over sets per type; mouse one-sided in the human
    direction, BH over the human-significant sets of that type;
  * transfers = human q < 0.05 and mouse oriented q < 0.05.

Anchor (Family 3): A1 = beta Spearman(human log2FC, mouse log2FC) on the H-D1 gene universe
(testis-table 1:1 genes detected in >= 5% of cells of the type in both species' DE units);
A2 = one-sided Wilcoxon, mouse log2FC of human T2D-up genes (padj < 0.1, log2FC > 0) vs the rest.
Both are also reported for every DE-able type and for the per-model mouse fits.

    PYTHONPATH=<chimera repo> python revision/tasks/T7_9_islet/hd4.py [--n-perm 10000]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from revision.common import gene_ids

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "revision" / "T7_9"
DE = RES / "de"
OUT = RES / "hd4"
FUNC = ROOT / "data" / "revision" / "functional"
SETS = {"hallmark": FUNC / "Enrichr.MSigDB_Hallmark_2020.server_response.txt",
        "reactome": FUNC / "Enrichr.Reactome_Pathways_2024.server_response.txt"}
COMPARA = [ROOT / "data" / "revision" / "ensembl110" / "Compara.110.protein_default.homologies.tsv.gz",
           ROOT / "data" / "revision" / "bone_marrow" / "ensembl110"
           / "Compara.110.protein_default.homologies.mus_musculus.tsv.gz"]
TABLE = ROOT / "data" / "processed" / "ortholog_table_testis.csv"
MIN_SIZE, MAX_SIZE = 15, 500
SEED = 20261101
DET = 0.05


def read_gmt(path) -> dict:
    out = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) > 2:
            out[p[0]] = sorted({g for g in p[2:] if g})
    return out


def one2one() -> pd.DataFrame:
    """Human-mouse ortholog_one2one pairs from both Compara dumps, each read in both orientations.

    A pair sits in only one dump depending on the dump layout, so the human dump alone loses about
    a third of the pairs (as ``P13_kidney/build_orthologs.py``).
    """
    cols = ["gene_stable_id", "species", "homology_type", "homology_gene_stable_id", "homology_species"]
    out = []
    for path in COMPARA:
        df = pd.read_csv(path, sep="\t", usecols=cols, dtype=str)
        df = df[df["homology_type"] == "ortholog_one2one"]
        fwd = df[(df["species"] == "homo_sapiens") & (df["homology_species"] == "mus_musculus")]
        rev = df[(df["species"] == "mus_musculus") & (df["homology_species"] == "homo_sapiens")]
        out.append(pd.DataFrame({"human_id": fwd["gene_stable_id"], "mouse_id": fwd["homology_gene_stable_id"]}))
        out.append(pd.DataFrame({"human_id": rev["homology_gene_stable_id"], "mouse_id": rev["gene_stable_id"]}))
    pairs = pd.concat(out, ignore_index=True).drop_duplicates()
    if pairs["human_id"].duplicated().any() or pairs["mouse_id"].duplicated().any():
        raise ValueError("one2one pairs are not one to one")
    return pairs


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


def load_type(t: str, contrast: str, o2o: pd.DataFrame):
    h = pd.read_csv(DE / f"human_t2d_vs_nd_{t}.tsv", sep="\t")
    m = pd.read_csv(DE / f"mouse_{contrast}_{t}.tsv", sep="\t")
    h = h.dropna(subset=["log2fc_shrunk"]).drop_duplicates("gene_id", keep=False)
    m = m.dropna(subset=["log2fc_shrunk"]).drop_duplicates("gene_id", keep=False)
    j = o2o.merge(h, left_on="human_id", right_on="gene_id").merge(
        m, left_on="mouse_id", right_on="gene_id", suffixes=("_h", "_m"))
    j = j.drop_duplicates("gene_symbol_h", keep=False)
    return h, m, j


def run_type(args):
    t, sets, n_perm, o2o = args
    _, _, j = load_type(t, "joint", o2o)
    rh = pd.Series(j["log2fc_shrunk_h"].to_numpy(), index=j["gene_symbol_h"].to_numpy())
    rm = pd.Series(j["log2fc_shrunk_m"].to_numpy(), index=j["gene_symbol_h"].to_numpy())
    out = []
    for lib, s in sets.items():
        gh = gsea(rh, s, n_perm, SEED).add_prefix("h_").rename(columns={"h_set": "set", "h_size": "size"})
        gm = gsea(rm, s, n_perm, SEED + 1).add_prefix("m_").rename(columns={"m_set": "set", "m_size": "size"})
        g = gh.merge(gm.drop(columns="size"), on="set")
        g.insert(0, "library", lib)
        out.append(g)
    g = pd.concat(out, ignore_index=True)
    g.insert(0, "cell_type", t)
    g["h_q"] = multipletests(g["h_p"], method="fdr_bh")[1]
    g["m_p_oriented"] = np.where(g["h_nes"] >= 0, g["m_p_up"], g["m_p_down"])
    sig = g["h_q"] < 0.05
    g["m_q_oriented"] = np.nan
    if sig.any():
        g.loc[sig, "m_q_oriented"] = multipletests(g.loc[sig, "m_p_oriented"], method="fdr_bh")[1]
    g["transfers"] = sig & (g["m_q_oriented"] < 0.05)
    g["call"] = np.where(~sig, "human_not_significant", np.where(g["transfers"], "transfers", "does_not_transfer"))
    return t, g, len(j)


def anchor(t: str, o2o: pd.DataFrame, testis: pd.DataFrame) -> list:
    rows = []
    for contrast in ("joint", "dbdb", "mstz"):
        if not (DE / f"mouse_{contrast}_{t}.tsv").exists():
            continue
        h, m, j = load_type(t, contrast, o2o)
        uni = j[j["human_id"].isin(testis["hsapiens_gene_id"]) & j["mouse_id"].isin(testis["mmusculus_gene_id"])
                & (j["det_frac_h"] >= DET) & (j["det_frac_m"] >= DET)]
        rho, prho = spearmanr(uni["log2fc_shrunk_h"], uni["log2fc_shrunk_m"])
        up = (uni["padj_h"] < 0.1) & (uni["log2fc_shrunk_h"] > 0)
        if up.sum() >= 5:
            w = mannwhitneyu(uni.loc[up, "log2fc_shrunk_m"], uni.loc[~up, "log2fc_shrunk_m"], alternative="greater")
            a2 = dict(a2_n_up=int(up.sum()), a2_p=float(w.pvalue),
                      a2_median_m_up=float(uni.loc[up, "log2fc_shrunk_m"].median()),
                      a2_median_m_rest=float(uni.loc[~up, "log2fc_shrunk_m"].median()))
        else:
            a2 = dict(a2_n_up=int(up.sum()), a2_p=np.nan)
        rows.append(dict(cell_type=t, mouse_contrast=contrast, n_genes=int(len(uni)), a1_spearman=float(rho),
                         a1_p_two_sided=float(prho), **a2))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DE / "de_meta.json").read_text())
    types = meta["de_types"]
    sets = {k: read_gmt(v) for k, v in SETS.items()}
    o2o = one2one()
    testis = pd.read_csv(TABLE)
    an = [r for t in types for r in anchor(t, o2o, testis)]
    pd.DataFrame(an).to_csv(OUT / "anchor.tsv", sep="\t", index=False)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(run_type, [(t, sets, a.n_perm, o2o) for t in types]))
    allg, summ = [], []
    for t, g, n in res:
        allg.append(g)
        summ.append(dict(cell_type=t, n_universe=n, n_sets=len(g), n_human_sig=int((g["h_q"] < 0.05).sum()),
                         n_transfer=int(g["transfers"].sum()),
                         n_not_transfer=int((g["call"] == "does_not_transfer").sum()),
                         nes_spearman_all_sets=float(spearmanr(g["h_nes"], g["m_nes"], nan_policy="omit")[0])))
    pd.concat(allg).to_csv(OUT / "hd4.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(OUT / "hd4_summary.tsv", sep="\t", index=False)
    (OUT / "DONE").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
