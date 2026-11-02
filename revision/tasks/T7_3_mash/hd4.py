"""T7.3: H-D4, hepatocyte MASH programs, Family 3 descriptive (amendment of 2026-11-01). No Chimera
readout enters.

GSEA machinery is T7.9's (``T7_9_islet/hd4.py``), so all disease cases share one implementation:
preranked on log2FC, weighted KS, gene-set permutations, MSigDB Hallmark 2020 and Reactome
2024, 15-500 genes in the universe. Universe: genes tested in the human and mouse primary fits, as
Ensembl Compara 110 one2one orthologs, on human symbols. This is wider than the liver H+Mouse
vocabulary, so genes such as TREM2 count here.

Ranking: fixed-scale apeGLM log2FC (prior scale 1, ``log2fc_fixed``) in both species. T7.9 ranks on
the adaptive prior, but with 3 healthy donors that prior collapses the human fit (de.py), and a
weighted KS on values near zero has no meaning. The adaptive prior and the Wald statistic are shown
beside it (h4_rank_sensitivity.tsv).

Calls (section 8 gives none; the rule is 8b's, plus "opposite"):
  transfers           human q < 0.05 (BH over sets) and mouse one-sided p in the human direction has
                      q < 0.05 (BH over the human-significant sets)
  opposite            human q < 0.05 and the mouse one-sided p against the human direction has q < 0.05
  does_not_transfer   human q < 0.05, neither of the above
  human_not_significant

Robustness: the same calls on leave-one-healthy-out, P98 included, NAFLD, bulk GSE135251, training
mice and the 36 leave-one-mouse-per-arm-out fits. Calibration: calls on the fits with permuted
condition labels, since gene-set permutation ignores correlation between genes.

    PYTHONPATH=<chimera repo> python revision/tasks/T7_3_mash/hd4.py [--n-perm 10000] [--jobs 8]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from revision.common import gene_ids
from revision.tasks.T7_9_islet import hd4 as G

ROOT = gene_ids.REPO_ROOT
OUT = ROOT / "results" / "revision" / "T7_3"
DE = OUT / "de"
SEED = 20261102
_S: dict = {}


def universe() -> pd.DataFrame:
    o2o = G.one2one()
    h = pd.read_csv(DE / "human_primary.tsv", sep="\t").dropna(subset=["log2fc_shrunk", "gene_id"]).drop_duplicates("gene_id", keep=False)
    m = pd.read_csv(DE / "mouse_primary.tsv", sep="\t").dropna(subset=["log2fc_shrunk", "gene_id"]).drop_duplicates("gene_id", keep=False)
    j = o2o.merge(h[["gene_id", "gene_symbol"]], left_on="human_id", right_on="gene_id").merge(
        m[["gene_id"]], left_on="mouse_id", right_on="gene_id", suffixes=("", "_m"))
    return j.drop_duplicates("gene_symbol", keep=False)[["human_id", "mouse_id", "gene_symbol"]]


def ranks(name: str, uni: pd.DataFrame, col: str = "log2fc_fixed") -> pd.Series:
    key = "mouse_id" if name.startswith(("mouse", "perm_mouse")) else "human_id"
    r = pd.read_csv(DE / f"{name}.tsv", sep="\t").dropna(subset=[col, "gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
    v = pd.Series(r[col].reindex(uni[key]).to_numpy(), index=uni["gene_symbol"].to_numpy())
    return v.dropna()


def _gsea(args):
    name, n_perm, seed = args
    name, col = name if isinstance(name, tuple) else (name, "log2fc_fixed")
    rk = ranks(name, _S["uni"], col)
    out = []
    for lib, s in _S["sets"].items():
        g = G.gsea(rk, s, n_perm, seed)
        g.insert(0, "library", lib)
        out.append(g)
    return name, pd.concat(out, ignore_index=True)


def combine(h: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    g = h.add_prefix("h_").rename(columns={"h_set": "set", "h_size": "size", "h_library": "library"}).merge(
        m.drop(columns="size").add_prefix("m_").rename(columns={"m_set": "set", "m_library": "library"}), on=["library", "set"])
    g["h_q"] = multipletests(g["h_p"], method="fdr_bh")[1]
    g["m_q_any"] = multipletests(g["m_p"], method="fdr_bh")[1]
    g["m_p_oriented"] = np.where(g["h_nes"] >= 0, g["m_p_up"], g["m_p_down"])
    g["m_p_against"] = np.where(g["h_nes"] >= 0, g["m_p_down"], g["m_p_up"])
    sig = g["h_q"] < 0.05
    g["m_q_oriented"] = np.nan
    g["m_q_against"] = np.nan
    if sig.any():
        g.loc[sig, "m_q_oriented"] = multipletests(g.loc[sig, "m_p_oriented"], method="fdr_bh")[1]
        g.loc[sig, "m_q_against"] = multipletests(g.loc[sig, "m_p_against"], method="fdr_bh")[1]
    g["call"] = np.where(~sig, "human_not_significant", np.where(g["m_q_oriented"] < 0.05, "transfers",
                         np.where(g["m_q_against"] < 0.05, "opposite", "does_not_transfer")))
    g.insert(0, "direction_h", np.where(g["h_nes"] >= 0, "up", "down"))
    return g


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-perm-sens", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    uni = universe()
    sets = {k: G.read_gmt(v) for k, v in G.SETS.items()}
    _S.update(uni=uni, sets=sets)
    summ_s = pd.concat([pd.read_csv(DE / f"summary_{x}.tsv", sep="\t") for x in ("sens", "source")])["name"].tolist()
    summ_p = pd.read_csv(DE / "summary_perm.tsv", sep="\t")["name"].tolist()
    hum_s = [n for n in summ_s if n.startswith(("human", "source"))] + ["bulk_nash_f0f3", "bulk_nash_all", "bulk_nafl"]
    mus_s = [n for n in summ_s if n.startswith("mouse")]
    perm_h = [n for n in summ_p if n.startswith("perm_human")][:20]
    perm_m = [n for n in summ_p if n.startswith("perm_mouse")][:20]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        prim = dict(ex.map(_gsea, [("human_primary", a.n_perm, SEED), ("mouse_primary", a.n_perm, SEED + 1)]))
        g = combine(prim["human_primary"], prim["mouse_primary"])
        g.insert(0, "family", 3)
        g.insert(1, "type", "hepatocyte")
        g.to_csv(OUT / "h4.tsv", sep="\t", index=False)
        alt = []
        for col in ("log2fc_shrunk", "stat"):
            rh = _gsea((("human_primary", col), a.n_perm_sens, SEED))[1]
            rm = _gsea((("mouse_primary", col), a.n_perm_sens, SEED + 1))[1]
            c = combine(rh, rm)
            c.insert(0, "rank_column", col)
            alt.append(c)
        pd.concat(alt).to_csv(OUT / "h4_rank_sensitivity.tsv", sep="\t", index=False)
        sig = g[g["h_q"] < 0.05]
        print(g["call"].value_counts().to_dict(), "universe", len(uni), flush=True)
        jobs = [(n, a.n_perm_sens, SEED + 2 + i) for i, n in enumerate(hum_s + mus_s + perm_h + perm_m)]
        res = dict(ex.map(_gsea, jobs, chunksize=1))
    # per-program NES of every human-side contrast, with BH over sets: the tissue-source check of the ledger
    long = []
    for n in hum_s + ["human_primary"]:
        x = (prim[n] if n == "human_primary" else res[n])[["library", "set", "nes", "p"]].copy()
        x["q"] = multipletests(x["p"], method="fdr_bh")[1]
        x.insert(0, "contrast", n)
        long.append(x)
    pd.concat(long).to_csv(OUT / "h4_human_contrasts.tsv", sep="\t", index=False)
    calls = {}
    for n in hum_s:
        c = combine(res[n], prim["mouse_primary"])
        calls[n] = c.set_index(["library", "set"])["call"]
    for n in mus_s:
        c = combine(prim["human_primary"], res[n])
        calls[n] = c.set_index(["library", "set"])["call"]
    cdf = pd.DataFrame(calls)
    main_call = g.set_index(["library", "set"])["call"].rename("primary")
    rob = pd.concat([main_call, cdf], axis=1)
    groups = {"lohd": [c for c in cdf if c.startswith("human_lohd")], "lomo": [c for c in cdf if c.startswith("mouse_lomo")]}
    for k, cols in groups.items():
        rob[f"{k}_same_call_frac"] = (cdf[cols].eq(main_call, axis=0)).mean(1)
    rob.reset_index().to_csv(OUT / "h4_sensitivity.tsv", sep="\t", index=False)
    rows = [dict(contrast="primary", n_sets=len(g), n_human_sig=int((g["h_q"] < 0.05).sum()),
                 n_transfers=int((g["call"] == "transfers").sum()), n_opposite=int((g["call"] == "opposite").sum()),
                 n_does_not=int((g["call"] == "does_not_transfer").sum()),
                 nes_spearman=float(spearmanr(g["h_nes"], g["m_nes"])[0]),
                 nes_spearman_hallmark=float(spearmanr(*g.loc[g["library"] == "hallmark", ["h_nes", "m_nes"]].to_numpy().T)[0]),
                 nes_sign_agree_human_sig=float((np.sign(sig["h_nes"]) == np.sign(sig["m_nes"])).mean()))]
    for n in hum_s + mus_s:
        c = combine(res[n], prim["mouse_primary"]) if n in hum_s else combine(prim["human_primary"], res[n])
        rows.append(dict(contrast=n, n_sets=len(c), n_human_sig=int((c["h_q"] < 0.05).sum()),
                         n_transfers=int((c["call"] == "transfers").sum()), n_opposite=int((c["call"] == "opposite").sum()),
                         n_does_not=int((c["call"] == "does_not_transfer").sum()),
                         nes_spearman=float(spearmanr(c["h_nes"], c["m_nes"], nan_policy="omit")[0])))
    pd.DataFrame(rows).to_csv(OUT / "h4_summary.tsv", sep="\t", index=False)
    null = []
    for n in perm_h:
        c = combine(res[n], prim["mouse_primary"])
        null.append(dict(kind="human_permuted", name=n, n_human_sig=int((c["h_q"] < 0.05).sum()),
                         n_transfers=int((c["call"] == "transfers").sum()), n_opposite=int((c["call"] == "opposite").sum()),
                         nes_spearman=float(spearmanr(c["h_nes"], c["m_nes"], nan_policy="omit")[0])))
    for n in perm_m:
        c = combine(prim["human_primary"], res[n])
        null.append(dict(kind="mouse_permuted", name=n, n_human_sig=int((c["h_q"] < 0.05).sum()),
                         n_transfers=int((c["call"] == "transfers").sum()), n_opposite=int((c["call"] == "opposite").sum()),
                         nes_spearman=float(spearmanr(c["h_nes"], c["m_nes"], nan_policy="omit")[0])))
    pd.DataFrame(null).to_csv(OUT / "h4_null.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).head(12).to_string())
    print(pd.DataFrame(null).groupby("kind")[["n_human_sig", "n_transfers", "n_opposite", "nes_spearman"]].describe().T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
