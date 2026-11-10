"""T12.3: tests E1-E5 of plan section 12 (bone marrow vs Shay et al. 2013 COE), Family 3 throughout.

Inputs: ``results/analysis/P12/readout/bm_genes.parquet`` (``readout.py``: D, S_ema, D_raw, C, ESM
blocks, B1), ``results/analysis/P12/drivers/drivers_genes.parquet`` (``drivers.py``: per-type and
per-species S_g contributions, B1 leave-one-type-out), ``data/analysis/bone_marrow/shay_coe.tsv``
(T12.1 recomputation, the pre-declared filter reading, 4,974 genes; 161 divergent).

Mechanics (amendment-log row "Section 12 T12.3 mechanics", committed before D was computed):

* E1: rho_p(score, COE | C) (section 3: ranks residualised on splines of rank pi_raw, expression,
  detection, CDS length; Pearson of residuals). Universe: genes with D, COE, C and B1 all defined (one
  universe for every score). One-sided p from 10,000 permutations of the score residuals within
  pi_raw x expression strata; 95% CI from 10,000 ESM-block bootstraps (refit in each draw); gene
  bootstrap CI beside. Scores: D, B1_pi (B1 residualised on pi), B1_raw, and S_ema, D_raw beside
  (Family 3), and D_all (D over every finite seed, G-V failures included). Delta = rho_p(D) - rho_p(B) on the same block-bootstrap draws. Repeated within the top
  tertile of rank pi_raw (design rebuilt there). Raw Spearman and rho_pi beside.
* E2: set = Shay divergent genes; comparison = genes with COE above the median COE of the E1
  universe; both inside the E1 universe. Stratified AUROC within pi_raw decile x expression decile
  cells taken over the vocabulary, cells kept with >= 1 divergent and >= 5 conserved genes per
  divergent gene (section 4 rule). One-sided p (lower) from 10,000 within-stratum label permutations;
  95% CI from 2,000 stratified bootstraps, the same draws for every score; Delta oriented so that
  positive favours D: AUROC(B) - AUROC(D). Raw AUROC and the pi x expression tertile version (the
  P13 E-K3 mechanics) beside.
* E3: the ten genes whose divergent lineage Shay names (Dataset S3 not obtained, so no further
  rows). Chimera: the top divergent type (argmax_k max(-C_gk, 0), five-seed mean, all species rows);
  a gene with no negative C_gk counts as a miss. Chance per gene = the vocabulary base rate of
  top divergent types falling in that gene's named set (genes with a defined top type); exact
  one-sided Poisson-binomial p. Uniform chance (named types / 9) beside; mouse-rows-only top type
  beside. Baseline: top type of the B1 leave-one-type-out increase, its own base rate.
* E4: Shay divergent genes in the vocabulary: mean over genes of (mouse-row share - lemur-row share)
  of the divergent S_g mass; gene-bootstrap 95% CI (10,000), one-sided Wilcoxon signed-rank p
  (mouse > lemur); vocabulary and COE-conserved genes beside as the expectation. Baseline: mean of
  E(human, lemur) - E(human, mouse) (pseudobulk correlation), same CI and test.
* E5: needs the Shay Dataset S3 list, not obtained (T12.1): not run.
* Success (section 12, as stated): D non-inferior to B1_pi on E1 (all-gene universe) and on E2,
  i.e. Delta >= -0.03 or its 95% CI includes 0, and E3 above chance (one-sided p < 0.05).

    PYTHONPATH=<chimera repo> python analysis/tasks/T12_3_tests/tests.py [--n-perm 10000] [--n-boot 10000]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_3_scoreboard import stats as S

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis" / "P12"
OUT = RES / "t12_3"
COE = ROOT / "data" / "analysis" / "bone_marrow" / "shay_coe.tsv"
C_COLS = K.COVARIATE_COLUMNS
SEED = 20261104
SCORES = ["D", "B1_pi", "B1_raw", "S_ema", "D_raw", "D_all"]
MARGIN = 0.03
TYPES = ["B", "HSPC", "NK", "T", "erythroid", "macrophage", "monocyte", "neutrophil", "plasma"]
NAMED = {  # section 12 E3; Shay granulocyte = our neutrophil
    "DPP4": ["B"], "NT5E": ["B"], "CD27": ["NK"], "TFEC": ["B"], "LMO2": ["B"],
    "IL15": ["neutrophil", "monocyte", "B"], "MYBL1": ["NK", "T"], "LDB2": ["HSPC", "NK"],
    "NFIX": ["HSPC", "monocyte"], "PRDM1": ["monocyte", "NK"],
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    tab = pd.read_parquet(RES / "readout" / "bm_genes.parquet")
    coe = pd.read_csv(COE, sep="\t").drop_duplicates("hsapiens_gene_id", keep=False).set_index("hsapiens_gene_id")
    tab["COE"] = coe["coe"].reindex(tab["hsapiens_gene_id"]).to_numpy()
    tab["divergent"] = coe["divergent"].reindex(tab["hsapiens_gene_id"]).fillna(False).astype(bool).to_numpy()
    drv = pd.read_parquet(RES / "drivers" / "drivers_genes.parquet").set_index("var_name").loc[tab["var_name"]]
    return tab, drv.reset_index()


# --------------------------------------------------------------------------- E1
def e1(u: pd.DataFrame, n_perm: int, n_boot: int, rng, label: str) -> tuple[list, list]:
    u = u.reset_index(drop=True)
    Z = S.design(u, C_COLS)
    Zpi = S.design(u, ["prior_raw"])
    R = np.column_stack([S.midrank(u["COE"])] + [S.midrank(u[s]) for s in SCORES])
    est = S.rho_p_matrix(R, Z)[1:]
    pis = S.rho_p_matrix(R, Zpi)[1:]
    E = S.resid(R, Z)
    lab = S.strata(u["prior_raw"].to_numpy(), u["log_cp10k_spmean"].to_numpy())
    bb = S.bootstrap(R, Z, n_boot, rng, u["block"].to_numpy())[:, 1:]
    bg = S.bootstrap(R, Z, min(n_boot, 2000), rng, None)[:, 1:]
    rows = []
    for j, s in enumerate(SCORES):
        null = S.perm_null(E[:, j + 1], E[:, 0], lab, n_perm, rng)
        lo, hi = S.pct_ci(bb[:, j])
        glo, ghi = S.pct_ci(bg[:, j])
        rows.append(dict(test="E1", universe=label, score=s, n=len(u), rho_raw=float(spearmanr(u[s], u["COE"])[0]),
                         rho_pi=float(pis[j]), rho_p=float(est[j]), ci_lo=lo, ci_hi=hi, ci_gene_lo=glo, ci_gene_hi=ghi,
                         p_one_sided=S.p_one_sided(float(est[j]), null), null_sd=float(null.std()),
                         sign_stat=S.sign_concordance(u[s].to_numpy(), u["COE"].to_numpy(), lab)))
    deltas = []
    iD = SCORES.index("D")
    for j, s in enumerate(SCORES):
        if s == "D":
            continue
        d = bb[:, iD] - bb[:, j]
        lo, hi = S.pct_ci(d)
        deltas.append(dict(test="E1", universe=label, comparison=f"D - {s}", delta=float(est[iD] - est[j]),
                           ci_lo=lo, ci_hi=hi, noninferior=bool(est[iD] - est[j] >= -MARGIN or (lo <= 0 <= hi))))
    for c in ("prior_raw", "log_cp10k_spmean"):
        rows.append(dict(test="E1", universe=label, score=c, n=len(u), rho_raw=float(spearmanr(u[c], u["COE"])[0])))
    return rows, deltas


# --------------------------------------------------------------------------- E2
def joint_auroc_boot(V: np.ndarray, inset: np.ndarray, lab: np.ndarray, n_boot: int, rng) -> np.ndarray:
    idx = [(np.flatnonzero((lab == s) & inset), np.flatnonzero((lab == s) & ~inset)) for s in np.unique(lab)]
    out = np.empty((n_boot, V.shape[1]))
    for b in range(n_boot):
        sel = []
        for i1, i0 in idx:
            if len(i1):
                sel.append(i1[rng.integers(0, len(i1), len(i1))])
            if len(i0):
                sel.append(i0[rng.integers(0, len(i0), len(i0))])
        sel = np.concatenate(sel)
        for j in range(V.shape[1]):
            out[b, j] = S.strat_auroc(V[sel, j], inset[sel], lab[sel])
    return out


def tertiles(x: np.ndarray) -> np.ndarray:
    r = S.midrank(x)
    return np.minimum((3 * (r - 0.5) / len(r)).astype(int), 2)


def e2(tab: pd.DataFrame, u: pd.DataFrame, n_perm: int, n_boot: int, rng) -> tuple[list, list, dict]:
    voc = tab.dropna(subset=["D"] + C_COLS).reset_index(drop=True)
    fine = pd.Series(S.cells(voc["prior_raw"].to_numpy(), voc["log_cp10k_spmean"].to_numpy()), index=voc["var_name"])
    coarse = pd.Series(tertiles(voc["prior_raw"].to_numpy()) * 3 + tertiles(voc["log_cp10k_spmean"].to_numpy()),
                       index=voc["var_name"])
    med = float(u["COE"].median())
    w = u[u["divergent"] | (u["COE"] > med)].reset_index(drop=True)
    inset = w["divergent"].to_numpy()
    V = w[SCORES].to_numpy(dtype=float)
    rows, deltas = [], []
    info = dict(coe_median=med, n_divergent_in_universe=int(inset.sum()), n_conserved_in_universe=int((~inset).sum()))
    for strat_name, labs in (("pi_x_expr_deciles_5to1", fine), ("pi_x_expr_tertiles", coarse)):
        lab_all = labs.loc[w["var_name"]].to_numpy()
        if strat_name.endswith("5to1"):
            keep = S.kept_strata(inset, lab_all)
        else:
            keep = np.zeros(len(w), bool)
            for s in np.unique(lab_all):
                m = lab_all == s
                if inset[m].any() and (~inset[m]).any():
                    keep |= m
        Vk, ik, lk = V[keep], inset[keep], lab_all[keep]
        draws = joint_auroc_boot(Vk, ik, lk, n_boot if strat_name.endswith("5to1") else min(n_boot, 1000), rng)
        est = []
        for j, s in enumerate(SCORES):
            auc = S.strat_auroc(Vk[:, j], ik, lk)
            est.append(auc)
            null = S.auroc_perm(Vk[:, j], ik, lk, n_perm, rng)
            lo, hi = S.pct_ci(draws[:, j])
            rows.append(dict(test="E2", strata=strat_name, score=s, n_divergent=int(ik.sum()), n_conserved=int((~ik).sum()),
                             n_strata=int(len(np.unique(lk))), auroc=auc, ci_lo=lo, ci_hi=hi,
                             p_one_sided_lower=S.p_one_sided(auc, null, lower=True), null_mean=float(null.mean()),
                             auroc_raw=S.strat_auroc(Vk[:, j], ik, np.zeros(len(ik), int))))
        iD = SCORES.index("D")
        for j, s in enumerate(SCORES):
            if s == "D":
                continue
            d = draws[:, j] - draws[:, iD]
            lo, hi = S.pct_ci(d)
            dl = float(est[j] - est[iD])
            deltas.append(dict(test="E2", universe=strat_name, comparison=f"AUROC({s}) - AUROC(D)", delta=dl,
                               ci_lo=lo, ci_hi=hi, noninferior=bool(dl >= -MARGIN or (lo <= 0 <= hi))))
        info[f"{strat_name}_divergent_kept"] = int(ik.sum())
    return rows, deltas, info


# --------------------------------------------------------------------------- E3
def poisson_binomial_sf(p: np.ndarray, k: int) -> float:
    """P(X >= k), X = sum of independent Bernoulli(p_i)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= (1 - q)
    return float(dist[k:].sum())


def e3(drv: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    sym = drv["symbol"].astype(str)
    out, recs = [], []
    for readout, col in (("Chimera C_gk, all species rows", "top_div_type"),
                         ("Chimera mouse rows only", "top_div_type_mouse_rows"),
                         ("B1 leave-one-type-out", "b1_top_type")):
        top_all = drv[col].astype(str)
        defined = top_all != ""
        hits, p_emp, p_uni = [], [], []
        for g, named in NAMED.items():
            i = np.flatnonzero(sym.to_numpy() == g)
            if len(i) != 1:
                raise AssertionError(f"{g}: {len(i)} rows in the vocabulary")
            top = top_all.iloc[i[0]]
            hit = top in named
            hits.append(hit)
            p_emp.append(float(top_all[defined].isin(named).mean()))
            p_uni.append(len(named) / len(TYPES))
            recs.append(dict(readout=readout, gene=g, named=",".join(named), top_type=top or "(none)", hit=hit,
                             chance_empirical=p_emp[-1], chance_uniform=p_uni[-1]))
        x = int(np.sum(hits))
        out.append(dict(test="E3", readout=readout, n_genes=len(NAMED), n_hit=x, frac_hit=x / len(NAMED),
                        expected_empirical=float(np.sum(p_emp)), p_empirical=poisson_binomial_sf(np.array(p_emp), x),
                        expected_uniform=float(np.sum(p_uni)), p_uniform=poisson_binomial_sf(np.array(p_uni), x),
                        n_vocab_defined=int(defined.sum())))
    return out, pd.DataFrame(recs)


# --------------------------------------------------------------------------- E4
def mean_ci(x: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = S.pct_ci(b)
    return float(x.mean()), lo, hi


def e4(tab: pd.DataFrame, drv: pd.DataFrame, u: pd.DataFrame, n_boot: int, rng) -> list:
    med = float(u["COE"].median())
    sets = {"Shay divergent": tab["divergent"].to_numpy(),
            "COE conserved (above median)": (tab["COE"] > med).to_numpy(),
            "all vocabulary": np.ones(len(tab), bool)}
    chim = (drv["div_share_mouse"] - drv["div_share_mouse_lemur"]).to_numpy()
    base = (tab["E_human_mouse_lemur"] - tab["E_human_mouse"]).to_numpy()
    rows = []
    for name, m in sets.items():
        for readout, v in (("Chimera: mouse-row minus lemur-row share of divergent S_g", chim),
                           ("B1 pair: E(human, lemur) - E(human, mouse)", base)):
            x = v[m & np.isfinite(v)]
            est, lo, hi = mean_ci(x, n_boot, rng)
            p = float(wilcoxon(x, alternative="greater").pvalue) if len(x) >= 10 else np.nan
            extra = {}
            if readout.startswith("Chimera"):
                for sp in ("human", "mouse_lemur", "mouse"):
                    s_ = drv[f"div_share_{sp}"].to_numpy()[m]
                    extra[f"mean_share_{sp}"] = float(np.nanmean(s_))
                extra["frac_mouse_gt_lemur"] = float((x > 0).mean())
            rows.append(dict(test="E4", gene_set=name, readout=readout, n=len(x), mean=est, ci_lo=lo, ci_hi=hi,
                             p_wilcoxon_greater=p, median=float(np.median(x)), **extra))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tab, drv = load()
    meta = json.loads((RES / "readout" / "readout_meta.json").read_text())
    u = tab.dropna(subset=["D", "COE", "B1_raw", "B1_pi"] + C_COLS).reset_index(drop=True)
    top = u[S.midrank(u["prior_raw"]) > 2 * len(u) / 3]
    r1, d1 = e1(u, a.n_perm, a.n_boot, rng, "all")
    r1t, d1t = e1(top, a.n_perm, a.n_boot, rng, "top_pi_tertile")
    r2, d2, info2 = e2(tab, u, a.n_perm, min(a.n_boot, 2000), rng)
    r3, g3 = e3(drv)
    r4 = e4(tab, drv, u, a.n_boot, rng)
    E1 = pd.DataFrame(r1 + r1t)
    E2 = pd.DataFrame(r2)
    deltas = pd.DataFrame(d1 + d1t + d2)
    E3 = pd.DataFrame(r3)
    E4 = pd.DataFrame(r4)
    E1.to_csv(OUT / "e1.tsv", sep="\t", index=False)
    E2.to_csv(OUT / "e2.tsv", sep="\t", index=False)
    deltas.to_csv(OUT / "deltas.tsv", sep="\t", index=False)
    E3.to_csv(OUT / "e3.tsv", sep="\t", index=False)
    g3.to_csv(OUT / "e3_genes.tsv", sep="\t", index=False)
    E4.to_csv(OUT / "e4.tsv", sep="\t", index=False)

    dE1 = deltas[(deltas.test == "E1") & (deltas.universe == "all") & (deltas.comparison == "D - B1_pi")].iloc[0]
    dE2 = deltas[(deltas.test == "E2") & (deltas.universe == "pi_x_expr_deciles_5to1")
                 & (deltas.comparison == "AUROC(B1_pi) - AUROC(D)")].iloc[0]
    e3c = E3[E3.readout == "Chimera C_gk, all species rows"].iloc[0]
    rule = dict(E1_noninferior=bool(dE1.noninferior), E1_delta=float(dE1.delta), E1_ci=[float(dE1.ci_lo), float(dE1.ci_hi)],
                E2_noninferior=bool(dE2.noninferior), E2_delta=float(dE2.delta), E2_ci=[float(dE2.ci_lo), float(dE2.ci_hi)],
                E3_above_chance=bool(e3c.p_empirical < 0.05), E3_hits=int(e3c.n_hit), E3_p=float(e3c.p_empirical))
    rule["success"] = bool(rule["E1_noninferior"] and rule["E2_noninferior"] and rule["E3_above_chance"])
    (OUT / "success_rule.json").write_text(json.dumps(dict(rule=rule, e2_info=info2, seeds=meta["seeds"],
                                                           n_universe_E1=len(u), n_top_tertile=len(top)), indent=2))

    # tidy summary
    def e1row(univ, s):
        return E1[(E1.universe == univ) & (E1.score == s)].iloc[0]

    def e2row(s):
        return E2[(E2.strata == "pi_x_expr_deciles_5to1") & (E2.score == s)].iloc[0]

    fmt = lambda lo, hi: f"[{lo:.4f}, {hi:.4f}]"  # noqa: E731
    summ = []
    for univ in ("all", "top_pi_tertile"):
        b = e1row(univ, "B1_pi")
        for s in ("D", "S_ema", "D_raw", "D_all", "B1_raw"):
            r = e1row(univ, s)
            summ.append(dict(test="E1", readout=f"rho_p({s}, COE | C), {univ}", estimate=r.rho_p, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             p=r.p_one_sided, baseline_estimate=b.rho_p, baseline_ci=fmt(b.ci_lo, b.ci_hi)))
        d = deltas[(deltas.test == "E1") & (deltas.universe == univ) & (deltas.comparison == "D - B1_pi")].iloc[0]
        summ.append(dict(test="E1", readout=f"Delta rho_p(D) - rho_p(B1_pi), {univ}", estimate=d.delta, ci_lo=d.ci_lo,
                         ci_hi=d.ci_hi, p=np.nan, baseline_estimate=np.nan, baseline_ci=""))
    b = e2row("B1_pi")
    for s in ("D", "S_ema", "D_raw", "D_all", "B1_raw"):
        r = e2row(s)
        summ.append(dict(test="E2", readout=f"stratified AUROC({s}), divergent vs COE-conserved", estimate=r.auroc,
                         ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p_one_sided_lower, baseline_estimate=b.auroc,
                         baseline_ci=fmt(b.ci_lo, b.ci_hi)))
    summ.append(dict(test="E2", readout="Delta AUROC(B1_pi) - AUROC(D) (positive favours D)", estimate=dE2.delta,
                     ci_lo=dE2.ci_lo, ci_hi=dE2.ci_hi, p=np.nan, baseline_estimate=np.nan, baseline_ci=""))
    b3 = E3[E3.readout == "B1 leave-one-type-out"].iloc[0]
    for _, r in E3[E3.readout != "B1 leave-one-type-out"].iterrows():
        summ.append(dict(test="E3", readout=f"named-lineage hits, {r.readout} (n_hit of 10; chance {r.expected_empirical:.2f})",
                         estimate=r.n_hit, ci_lo=np.nan, ci_hi=np.nan, p=r.p_empirical, baseline_estimate=b3.n_hit,
                         baseline_ci=f"chance {b3.expected_empirical:.2f}, p {b3.p_empirical:.3f}"))
    for _, r in E4[E4.readout.str.startswith("Chimera")].iterrows():
        bb = E4[(E4.gene_set == r.gene_set) & E4.readout.str.startswith("B1")].iloc[0]
        summ.append(dict(test="E4", readout=f"mouse minus lemur share of divergent S_g, {r.gene_set} (n {r.n})",
                         estimate=r["mean"], ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p_wilcoxon_greater,
                         baseline_estimate=bb["mean"], baseline_ci=fmt(bb.ci_lo, bb.ci_hi)))
    summ.append(dict(test="E5", readout="not run: Shay Dataset S3 not obtained", estimate=np.nan, ci_lo=np.nan,
                     ci_hi=np.nan, p=np.nan, baseline_estimate=np.nan, baseline_ci=""))
    summ.append(dict(test="success_rule", readout=f"E1 non-inferior {rule['E1_noninferior']}; E2 non-inferior "
                     f"{rule['E2_noninferior']}; E3 above chance {rule['E3_above_chance']}",
                     estimate=float(rule["success"]), ci_lo=np.nan, ci_hi=np.nan, p=np.nan, baseline_estimate=np.nan,
                     baseline_ci=""))
    pd.DataFrame(summ).to_csv(RES / "t12_3_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80):
        print(E1.round(4).to_string(index=False))
        print(E2.round(4).to_string(index=False))
        print(deltas.round(4).to_string(index=False))
        print(E3.round(4).to_string(index=False))
        print(g3.to_string(index=False))
        print(E4.round(4).to_string(index=False))
        print(json.dumps(rule, indent=2), json.dumps(info2, indent=2))
    (OUT / "DONE").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
