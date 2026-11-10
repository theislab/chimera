"""O1 follow-up: text-cited set statistics recomputed without the genes the ortholog audit dropped.

Genes: OTC (human source zero), ASS1 (not in the liver table), COL17A1, CD40LG, SELE, SELP (zebrafish
one-to-one partner another protein or low-confidence; SELE also no mouse counts). They are removed
from the gene universe of each liver cohort, so they enter neither a set nor its matched null.

Cited numbers and their sources (same code, inputs, seeds and set/score positions as the originals):
* X2 ``drivers.csv``: Fig 4c urea-cycle / gluconeogenesis set, score D, matchings C and C_hep,
  universes all and complete, every liver cohort (main text and Extended Results: 4-sp d -1.36,
  Human+Macaque -2.14, Human+Mouse +0.66, p 0.40).
* X2 ``set_species_share.csv``: species shares of the same set's summed margin (human 69%).
* T2.6 ``category_matched.csv``: broad Cytoskeleton & ECM on liver 4 species, expression- and
  length-matched d (+0.24), the only text-cited category that holds one of the genes (COL17A1).
  The other cited categories hold none of them (``classify_extended`` / ``classify_broad``).

A control arm with the original gene universe must reproduce the original rows exactly before the
excluded arm is read. Outputs beside the originals, never over them:
``results/analysis/X2/o1_excluded/`` and ``results/analysis/T2_6/o1_excluded/``.

Run from ``chimera/``::

    PYTHONPATH=$PWD mamba run -n chimera_env python -m analysis.tasks.O1_orthologs.set_exclusion
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.common import cds as CDS
from analysis.common import cohorts as C
from analysis.common import gene_ids

sys.path.insert(0, str(gene_ids.REPO_ROOT / "analysis" / "tasks" / "X2_liver_biology"))
sys.path.insert(0, str(gene_ids.REPO_ROOT / "analysis" / "tasks" / "T2_6_matched"))
import analyses as AN  # noqa: E402
import stats as SX  # noqa: E402
from analysis.tasks.T2_6_matched import run as T26  # noqa: E402

DROP = {"OTC", "ASS1", "COL17A1", "CD40LG", "SELE", "SELP"}
X2_SETS = ["fig4c:Urea cycle / GNG"]
X2_SCORES = ["D"]
T26_COHORT = "liv_4sp"
T26_SETS = ["broad:Cytoskeleton & ECM"]
OUT_X2 = SX.OUT / "o1_excluded"
OUT_T26 = gene_ids.REPO_ROOT / "results" / "analysis" / "T2_6" / "o1_excluded"


def _x2_rows(g: pd.DataFrame, cohort: str, sets: dict, scores: list, keep_sets: list, keep_scores: list,
             B: int, seed: int, match_cols: list) -> list[dict]:
    """``stats._set_excess`` restricted to some sets and scores, with the seeds of their full positions."""
    g = g.reset_index(drop=True)
    rows = []
    okc = g[match_cols].notna().all(1).to_numpy()
    Z = np.full((len(g), len(match_cols)), np.nan)
    Z[okc] = np.column_stack([SX.rank01(g.loc[okc, c]) for c in match_cols])
    present = [s for s in scores if s in g.columns]
    for j, (name, mask) in enumerate(sets.items()):
        if name not in keep_sets:
            continue
        mask = np.asarray(mask, bool)
        for k, sc in enumerate(scores):
            if sc not in keep_scores or sc not in present:
                continue
            v = g[sc].to_numpy(dtype=float)
            ok = okc & np.isfinite(v)
            idx, pool = np.where(mask & ok)[0], np.where(~mask & ok)[0]
            r = dict(cohort=cohort, set=name, score=sc, n_set=int(mask.sum()), n=len(idx),
                     symbols=";".join(sorted(map(str, g.loc[idx, "symbol"].dropna())))[:600])
            if len(idx) >= 3:
                rng = np.random.default_rng(seed + 1000 * j + k)
                null = SX.knn_null(v, Z, idx, pool, B, rng)
                obs, sd, mu = v[idx].mean(), v[ok].std(ddof=1), null.mean()
                r.update(set_mean=obs, null_mean=mu, excess=obs - mu, excess_d=(obs - mu) / sd,
                         p=(1 + (np.abs(null - mu) >= abs(obs - mu)).sum()) / (B + 1))
            rows.append(r)
    return rows


def x2(drop: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    drv, share = [], []
    for c in AN.LIVER:
        g0 = SX.load_genes(c)
        sets0 = AN.figure_sets(g0, c)
        scores = AN.driver_scores(g0)
        g = g0[~g0.symbol.isin(drop)].reset_index(drop=True)
        sets = AN.figure_sets(g, c)
        assert list(sets) == list(sets0)
        for uni in ("all", "complete"):
            keep = np.ones(len(g), bool) if uni == "all" else g["complete"].to_numpy()
            gg = g[keep].reset_index(drop=True)
            ss = {k: v[keep] for k, v in sets.items()}
            for mname, cols in SX.MATCHINGS.items():
                for r in _x2_rows(gg, c, ss, scores, X2_SETS, X2_SCORES, AN.B, 7, cols):
                    drv.append(dict(r, matching=mname, universe=uni))
        cs = [x for x in g.columns if x.startswith("Cs__")]
        for name in X2_SETS:
            sub = g[sets[name]]
            tot = sub["S_full_meanz"].sum()
            share.append(dict(cohort=c, set=name, n=len(sub), sum_S=tot,
                              **{f"share_{x}": sub[x].sum() / tot for x in cs}))
    return pd.DataFrame(drv), pd.DataFrame(share)


def t26(drop: set) -> pd.DataFrame:
    i = C.COHORT_ORDER.index(T26_COHORT)
    cds = CDS.human_cds_table(str(CDS.ENSEMBL_DIR))
    co = C.load_cohort(T26_COHORT, keep_adata=False)
    g0 = co.genes.join(cds[["log_cds_length"]], on="hsapiens_gene_id")
    g0["da_expr_resid"] = C.expression_residual(g0["delta_alpha"], g0["log_mean"])
    order = list(T26.define_sets(g0, T26_COHORT))
    g = g0[~g0["symbol"].isin(drop)].copy()
    if drop:
        g["da_expr_resid"] = C.expression_residual(g["delta_alpha"], g["log_mean"])
    sets = T26.define_sets(g, T26_COHORT)
    assert list(sets) == order, "set order changed; seeds would not match"
    # Same seed per set position as run.py; only the cited sets are computed.
    rows = []
    for j, name in enumerate(order):
        if name in T26_SETS:
            one = T26.set_rows(g, T26_COHORT, {name: sets[name]}, 2000, 1000 * i + j)
            rows += one
    return pd.DataFrame(rows)


def main() -> None:
    OUT_X2.mkdir(parents=True, exist_ok=True)
    OUT_T26.mkdir(parents=True, exist_ok=True)
    # Control: original universe reproduces the original rows exactly.
    d0, s0 = x2(set())
    ref = pd.read_csv(SX.OUT / "drivers.csv")
    m = d0.merge(ref, on=["cohort", "set", "score", "matching", "universe"], suffixes=("", "_ref"))
    assert len(m) == len(d0) and np.allclose(m.excess_d, m.excess_d_ref, equal_nan=True) \
        and np.allclose(m.p, m.p_ref, equal_nan=True), "X2 control does not reproduce drivers.csv"
    sref = pd.read_csv(SX.OUT / "set_species_share.csv")
    ms = s0.merge(sref, on=["cohort", "set"], suffixes=("", "_ref"))
    assert np.allclose(ms.share_Cs__Human, ms.share_Cs__Human_ref), "X2 control does not reproduce shares"
    t0 = t26(set())
    tref = pd.read_csv(gene_ids.REPO_ROOT / "results" / "analysis" / "T2_6" / "category_matched.csv")
    mt = t0.merge(tref, on=["cohort", "set"], suffixes=("", "_ref"))
    assert len(mt) == len(t0) and np.allclose(mt.clamped_exprlenmatched_d, mt.clamped_exprlenmatched_d_ref), \
        "T2.6 control does not reproduce category_matched.csv"
    print("controls reproduce the originals", flush=True)
    d1, s1 = x2(DROP)
    t1 = t26(DROP)
    d0.to_csv(OUT_X2 / "drivers_control.csv", index=False)
    d1.to_csv(OUT_X2 / "drivers.csv", index=False)
    s1.to_csv(OUT_X2 / "set_species_share.csv", index=False)
    t1.to_csv(OUT_T26 / "category_matched.csv", index=False)
    cmp = d0.merge(d1, on=["cohort", "set", "score", "matching", "universe"], suffixes=("_orig", "_excl"))
    pd.set_option("display.width", 250)
    print(cmp[["cohort", "matching", "universe", "n_orig", "n_excl", "excess_d_orig", "excess_d_excl", "p_orig",
               "p_excl"]].round(4).to_string())
    print(s0[["cohort", "share_Cs__Human"]].merge(s1[["cohort", "n", "share_Cs__Human"]], on="cohort",
                                                   suffixes=("_orig", "_excl")).round(4).to_string())
    cols = ["set", "n", "clamped_exprlenmatched_d", "clamped_exprlenmatched_p", "raw_exprlenmatched_d"]
    print(pd.concat([t0[cols].assign(arm="orig"), t1[cols].assign(arm="excl")]).round(4).to_string())


if __name__ == "__main__":
    main()
