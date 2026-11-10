"""Liver MASH: H-D1 and H-D1b on hepatocytes, Family 3 descriptive (pre-specified analysis plan, sections 8
and 3; amendment log of the plan, 2026-11-01). G-D1 failed, so the frozen pass rule (>= 3 of 5 types)
does not apply and no status is computed.

Outcome: Q = sign(l^h) * m, m the apeGLM-shrunken mouse log2FC (adaptive prior, as the plan says),
l^h the unshrunk human log2FC (the plan shrinks m only; the adaptive prior collapses the human fit,
de.py), on liver H+Mouse vocabulary genes detected
in >= 5% of hepatocyte nuclei in both species. Join on hsapiens_gene_id (human), mmusculus_gene_id
of the same ortholog row (mouse). Readout: D, 10 seeds, the seed consensus of
analysis/tasks/T3_3_consensus; S_ema beside it.

rho_p: ranks of Q and of the score, each residualized on C (plan, section 3) plus splines of rank |l^h|
and rank SE(m), Pearson of the residuals. Null: D residuals permuted within pi_raw x expression
deciles, 10,000 draws, one-sided. CI: block bootstrap on the ESM modules, 10,000 draws, refit per
draw; gene-level CI and variance ratio beside it.

Tertile test: among human-DE genes (FDR < 0.1) of the universe, sign agreement of l^h and m in the
top D tertile minus the bottom tertile; same stratified permutation of D, one-sided.

H-D1b: OLS on rank Q. Step 0 pi + expression; step 1 + B1b, B2b; step 2 + D (splines df 4 of each
rank). On genes with all scores. Delta R2 with block-bootstrap CI and stratified permutation p.

All outputs land in ``results/analysis/T7_3``: h1.tsv, h1b.tsv, h1_head_to_head.tsv, h1_anchor.tsv,
h1_sensitivity.tsv, h1_controls.tsv, h1_genes.tsv, named_genes.tsv.

    PYTHONPATH=<chimera repo> python analysis/tasks/T7_3_mash/hd1.py [--n-perm 10000] [--n-boot 10000]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import mannwhitneyu, rankdata, spearmanr

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.T4_3_scoreboard import stats as S

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis"
OUT = RES / "T7_3"
DE = OUT / "de"
COHORT = "liv_hmouse"
C_COLS = K.COVARIATE_COLUMNS
NUIS = ["abs_l", "se_m"]
DET = 0.05
FDR = 0.1
SEED = 20261102
INTEGRATED = (42, 43, 44, 47, 48, 51)
TYPES5 = ["hepatocyte", "hsc_fibroblast", "endothelial", "macrophage", "cholangiocyte"]
NAMED = ["PNPLA3", "TM6SF2", "HSD17B13", "MBOAT7", "GCKR", "APOB", "PPARA", "PPARG", "CD36", "FASN", "SCD", "ACACA",
         "SREBF1", "FGF21", "CIDEB", "CIDEC", "PLIN2", "TREM2", "COL1A1", "SPP1", "SOX9", "GDF15", "ATF3", "DDIT3"]


def vocabulary() -> pd.DataFrame:
    tab = pd.read_parquet(RES / "T4_3" / "cache" / f"{COHORT}__inputs.parquet")
    cons = pd.read_csv(RES / "T3_3" / f"{COHORT}__consensus.csv")
    if not (cons["var_name"].to_numpy() == tab["var_name"].to_numpy()).all():
        raise AssertionError("consensus gene order differs from the T4.3 inputs")
    if not (cons["hsapiens_gene_id"].to_numpy() == tab["hsapiens_gene_id"].to_numpy()).all():
        raise AssertionError("consensus human IDs differ from the T4.3 inputs")
    if int(cons["n_seeds"].iloc[0]) != 10:
        raise AssertionError("D must be the 10-seed consensus")
    tab["D"] = cons["D"].to_numpy()
    tab["S_ema"] = cons["S_ema"].to_numpy()
    tab["symbol"] = cons["symbol"].to_numpy()
    o = gene_ids.load_table(COHORT)
    ids = tab["var_name"].str.replace("ortho_", "", regex=False).astype(int)
    if not (o.loc[ids, "hsapiens_gene_id"].to_numpy() == tab["hsapiens_gene_id"].to_numpy()).all():
        raise AssertionError("ortholog table human IDs differ")
    tab["mouse_id"] = o.loc[ids, "mmusculus_gene_id"].to_numpy()
    b = pd.read_csv(RES / "T4_2" / f"baselines__{COHORT}.csv").set_index("var_name")
    tab["B1b"] = b["B1b_raw"].reindex(tab["var_name"]).to_numpy()
    tab["B2b"] = b["B2b"].reindex(tab["var_name"]).to_numpy()
    mg = pd.read_parquet(RES / "T2_5" / f"{COHORT}__genes.parquet").set_index("var_name")
    tab["margin_support"] = mg["margin_support"].reindex(tab["var_name"]).to_numpy()
    hep = pd.read_parquet(RES / "X2" / "genes" / f"{COHORT}.parquet").set_index("var_name")["hep_enrich"]
    tab["hep_enrich"] = hep.reindex(tab["var_name"]).to_numpy()
    runs = I.seed_runs_used(COHORT)
    M = pd.DataFrame({int(s): K.read_run(r)["D"] for s, r in sorted(runs.items())})
    if len(M.columns) != 10:
        raise AssertionError("expected 10 seeds")
    for s in M.columns:
        tab[f"Dloso_{s}"] = M.drop(columns=s).mean(1).to_numpy()
    tab["D_integrated"] = M[list(INTEGRATED)].mean(1).to_numpy()
    if abs(spearmanr(M.mean(1), tab["D"])[0] - 1) > 1e-6:
        raise AssertionError("seed mean does not reproduce D")
    return tab


def hep_share() -> pd.Series:
    """Hepatocyte share of expression over the five types (human primary units): the hepatocyte-dominant rule."""
    a = ad.read_h5ad(ROOT / "data" / "analysis" / "disease" / "disease_pseudobulk_human.h5ad")
    a = a[a.obs["primary"].astype(bool).to_numpy()].copy()
    X = np.asarray(a.X.toarray(), dtype=float)
    out = {}
    for t in TYPES5:
        m = (a.obs["cell_ontology_class"].astype(str) == t).to_numpy()
        # healthy and MASH weighted equally, so 27 MASH donors do not set the share alone
        arms = []
        for c in ("control", "disease"):
            k = m & (a.obs["condition"].astype(str) == c).to_numpy()
            arms.append(X[k].sum(0) / max(X[k].sum(), 1) * 1e4)
        out[t] = np.mean(arms, axis=0)
    d = pd.DataFrame(out, index=a.var["gene_id_ens110"].astype(object).to_numpy())
    d = d[d.index.notna()]
    d = d[~d.index.duplicated(keep=False)]
    return (d["hepatocyte"] / d.sum(1).replace(0, np.nan)).rename("hep_share")


def read_de(name: str, id_col: str = "gene_id") -> pd.DataFrame:
    r = pd.read_csv(DE / f"{name}.tsv", sep="\t")
    r = r.dropna(subset=["log2fc_mle", "log2fc_shrunk", id_col]).drop_duplicates(id_col, keep=False)
    return r.set_index(id_col)


def universe(tab: pd.DataFrame, hum: pd.DataFrame, mus: pd.DataFrame, det_from: pd.DataFrame | None = None,
             l_col: str = "log2fc_mle") -> pd.DataFrame:
    """Vocabulary genes with l^h, m, SE(m) and C, detected >= 5% in both species. For bulk or batch
    contrasts ``det_from`` supplies the human detection, since those fits have none."""
    d = tab.copy()
    d["l"] = hum[l_col].reindex(d["hsapiens_gene_id"]).to_numpy()
    d["padj_h"] = hum["padj"].reindex(d["hsapiens_gene_id"]).to_numpy()
    src = hum if det_from is None else det_from
    d["det_h"] = src["det_frac"].reindex(d["hsapiens_gene_id"]).to_numpy()
    d["m"] = mus["log2fc_shrunk"].reindex(d["mouse_id"]).to_numpy()
    d["se_m"] = mus["lfcse_shrunk"].reindex(d["mouse_id"]).to_numpy()
    d["padj_m"] = mus["padj"].reindex(d["mouse_id"]).to_numpy()
    d["det_m"] = mus["det_frac"].reindex(d["mouse_id"]).to_numpy()
    n_join = int((d["l"].notna() & d["m"].notna()).sum())
    d = d[(d["det_h"] >= DET) & (d["det_m"] >= DET)]
    d = d.dropna(subset=["l", "m", "se_m", "D"] + C_COLS).copy()
    d["abs_l"] = d["l"].abs()
    d["Q"] = np.sign(d["l"]) * d["m"]
    d["agree"] = (np.sign(d["l"]) == np.sign(d["m"])).astype(float)
    d.attrs["n_join"] = n_join
    return d.reset_index(drop=True)


def perm_indices(lab: np.ndarray, n: int, rng, chunk: int = 250):
    groups = [np.flatnonzero(lab == s) for s in np.unique(lab)]
    done = 0
    while done < n:
        c = min(chunk, n - done)
        P = np.empty((c, len(lab)), dtype=np.int64)
        for idx in groups:
            P[:, idx] = idx[np.argsort(rng.random((c, len(idx))), axis=1)]
        yield P
        done += c


def tertile_stat(d_vals: np.ndarray, agree: np.ndarray) -> np.ndarray:
    """Top minus bottom D-tertile sign agreement. d_vals: (draws, genes) or (genes,)."""
    x = np.atleast_2d(d_vals)
    r = np.argsort(np.argsort(x, axis=1), axis=1)
    n = x.shape[1]
    k = n // 3
    top, bot = r >= n - k, r < k
    return ((top * agree).sum(1) / k - (bot * agree).sum(1) / k)


def tertile(u: pd.DataFrame, score: str, lab: np.ndarray, n_perm: int, rng, resid_on: np.ndarray | None = None) -> dict:
    de = np.flatnonzero((u["padj_h"] < FDR).to_numpy())
    out = dict(n_human_de=int(len(de)), tertile_defined=bool(len(de) >= 150))
    if len(de) < 30:
        return {**out, "tert_top": np.nan, "tert_bottom": np.nan, "tert_diff": np.nan, "tert_p": np.nan}
    v = u[score].to_numpy(dtype=float)
    if resid_on is not None:
        v = S.resid(S.midrank(v), resid_on)
    ag = u["agree"].to_numpy()[de]
    x = v[de]
    r = np.argsort(np.argsort(x))
    k = len(x) // 3
    T = float(tertile_stat(x, ag)[0])
    cnt, tot = 0, 0
    for P in perm_indices(lab, n_perm, rng):
        cnt += int((tertile_stat(v[P][:, de], ag) >= T).sum())
        tot += P.shape[0]
    return {**out, "tert_top": float(ag[r >= len(x) - k].mean()), "tert_bottom": float(ag[r < k].mean()),
            "tert_diff": T, "tert_p": (1 + cnt) / (tot + 1)}


def rho_row(u: pd.DataFrame, score: str, n_perm: int, n_boot: int, rng, extra_cols=(), with_tertile=True) -> dict:
    u = u.dropna(subset=[score] + list(extra_cols)).reset_index(drop=True)
    cols = C_COLS + NUIS + list(extra_cols)
    Z = S.design(u, cols)
    Rm = np.column_stack([S.midrank(u["Q"]), S.midrank(u[score])])
    E = S.resid(Rm, Z)
    rho = float(np.corrcoef(E[:, 0], E[:, 1])[0, 1])
    lab = S.strata(u["prior_raw"].to_numpy(), u["log_cp10k_spmean"].to_numpy())
    row = dict(score=score, n=len(u), sd_Q=float(u["Q"].std()), rho_raw=float(spearmanr(u[score], u["Q"])[0]))
    Zpi = S.design(u, ["prior_raw"])
    row["rho_pi"] = float(S.rho_p_matrix(Rm, Zpi)[1])
    row["rho_p"] = rho
    if n_perm:
        null = S.perm_null(E[:, 1], E[:, 0], lab, n_perm, rng)
        row.update(p_one_sided=S.p_one_sided(rho, null), null_sd=float(null.std()), null_q975=float(np.quantile(null, 0.975)))
    if n_boot:
        bb = S.bootstrap(Rm, Z, n_boot, rng, u["block"].to_numpy())[:, 1]
        bg = S.bootstrap(Rm, Z, min(n_boot, 2000), rng, None)[:, 1]
        lo, hi = S.pct_ci(bb)
        glo, ghi = S.pct_ci(bg)
        row.update(ci_lo=lo, ci_hi=hi, ci_gene_lo=glo, ci_gene_hi=ghi, var_ratio_block_gene=float(bb.var() / bg.var()))
    row["sign_stat"] = S.sign_concordance(u[score].to_numpy(), u["Q"].to_numpy(), lab)
    if with_tertile:
        row.update(tertile(u, score, lab, min(n_perm, 10000) or 1000, rng))
    return row


def pi_row(u: pd.DataFrame) -> dict:
    """pi and log mean have no rho_p on C (plan, section 3): raw Spearman, plus a partial on the other C terms."""
    out = []
    for s in ("prior_raw", "log_cp10k_spmean"):
        others = [c for c in C_COLS if c != s] + NUIS
        Z = S.design(u, others)
        Rm = np.column_stack([S.midrank(u["Q"]), S.midrank(u[s])])
        out.append(dict(score=s, n=len(u), rho_raw=float(spearmanr(u[s], u["Q"])[0]),
                        rho_p_given_other_C=float(S.rho_p_matrix(Rm, Z)[1])))
    return out


def head_to_head(u: pd.DataFrame, scores: list, n_boot: int, rng) -> pd.DataFrame:
    w = u.dropna(subset=scores).reset_index(drop=True)
    Z = S.design(w, C_COLS + NUIS)
    R = np.column_stack([S.midrank(w["Q"])] + [S.midrank(w[s]) for s in scores])
    est = S.rho_p_matrix(R, Z)[1:]
    draws = S.bootstrap(R, Z, n_boot, rng, w["block"].to_numpy())[:, 1:]
    rows = []
    for j, s in enumerate(scores):
        lo, hi = S.pct_ci(draws[:, j])
        rows.append(dict(score=s, n_pair=len(w), rho_p=float(est[j]), ci_lo=lo, ci_hi=hi))
        if s != "D":
            d = draws[:, 0] - draws[:, j]
            lo, hi = S.pct_ci(d)
            rows[-1].update(delta_D_minus_score=float(est[0] - est[j]), delta_lo=lo, delta_hi=hi)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- H-D1b
def spl(u: pd.DataFrame, cols: list) -> np.ndarray:
    return S.design(u, cols)


def r2(y: np.ndarray, Z: np.ndarray, w: np.ndarray | None = None) -> float:
    if w is None:
        e = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    m = w > 0
    Zm, ym, wm = Z[m], y[m], w[m]
    ZtW = Zm.T * wm
    b = np.linalg.lstsq(ZtW @ Zm, ZtW @ ym, rcond=None)[0]
    e = ym - Zm @ b
    ybar = (wm * ym).sum() / wm.sum()
    return 1 - (wm * e * e).sum() / (wm * (ym - ybar) ** 2).sum()


def nested(u: pd.DataFrame, n_perm: int, n_boot: int, rng, variant: str, base_extra=(), last="D") -> pd.DataFrame:
    w = u.dropna(subset=["B1b", "B2b", last]).reset_index(drop=True)
    y = S.midrank(w["Q"])
    base = ["prior_raw", "log_cp10k_spmean"] + list(base_extra)
    steps = [("pi + expression", base), ("+ B1b, B2b", base + ["B1b", "B2b"]), (f"+ {last}", base + ["B1b", "B2b", last])]
    Zs = [spl(w, c) for _, c in steps]
    r = [r2(y, Z) for Z in Zs]
    lab = S.strata(w["prior_raw"].to_numpy(), w["log_cp10k_spmean"].to_numpy())
    # stratified permutation of the last score: delta R2 of the last step
    last_col = S.midrank(w[last])
    null = []
    for P in perm_indices(lab, n_perm, rng, chunk=100):
        for row in P:
            f = w.copy()
            f[last] = last_col[row]
            null.append(r2(y, spl(f, base + ["B1b", "B2b", last])) - r[1])
    null = np.array(null)
    boots = np.empty((n_boot, 3))
    blk = w["block"].to_numpy()
    for b in range(n_boot):
        wt = S.block_weights(blk, rng)
        boots[b] = [r2(y, Z, wt) for Z in Zs]
    rows = []
    for i, (name, cols) in enumerate(steps):
        k = Zs[i].shape[1]
        row = dict(variant=variant, step=i, terms=name, n=len(w), r2=r[i], n_coef=k)
        if i:
            d = r[i] - r[i - 1]
            k0 = Zs[i - 1].shape[1]
            F = (d / (k - k0)) / ((1 - r[i]) / (len(w) - k))
            lo, hi = S.pct_ci(boots[:, i] - boots[:, i - 1])
            row.update(delta_r2=d, delta_ci_lo=lo, delta_ci_hi=hi, F=F, p_F=float(f_dist.sf(F, k - k0, len(w) - k)))
            if i == 2:
                row.update(p_perm=float((1 + (null >= d).sum()) / (len(null) + 1)), null_delta_mean=float(null.mean()))
        rows.append(row)
    # reverse order, so the D-alone increment is shown too
    Zd = spl(w, base + [last])
    rows.append(dict(variant=variant, step=9, terms=f"pi + expression + {last} (reverse order)", n=len(w), r2=r2(y, Zd),
                     n_coef=Zd.shape[1], delta_r2=r2(y, Zd) - r[0]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- anchor
def anchor(name: str, u: pd.DataFrame) -> dict:
    up = (u["padj_h"] < FDR) & (u["l"] > 0)
    dn = (u["padj_h"] < FDR) & (u["l"] < 0)
    rho, p = spearmanr(u["l"], u["m"])
    de = (u["padj_h"] < FDR)
    r = dict(contrast=name, n=len(u), spearman_l_m=float(rho), p_two_sided=float(p), sign_agree_all=float(u["agree"].mean()),
             n_de_up=int(up.sum()), n_de_down=int(dn.sum()), sign_agree_de=float(u.loc[de, "agree"].mean()) if de.any() else np.nan,
             n_mouse_de=int((u["padj_m"] < FDR).sum()))
    both = de & (u["padj_m"] < FDR)
    r.update(n_de_both=int(both.sum()), sign_agree_de_both=float(u.loc[both, "agree"].mean()) if both.any() else np.nan)
    if up.sum() >= 5:
        r["a2_up_p"] = float(mannwhitneyu(u.loc[up, "m"], u.loc[~up, "m"], alternative="greater").pvalue)
    if dn.sum() >= 5:
        r["a2_down_p"] = float(mannwhitneyu(u.loc[dn, "m"], u.loc[~dn, "m"], alternative="less").pvalue)
    return r


# --------------------------------------------------------------------------- jobs
_TAB = None


def _quick(args) -> dict:
    """rho_p, Spearman and tertile difference for one (human, mouse) fit pair, without CI."""
    hname, mname, kind, seed = args
    u = universe(_TAB, read_de(hname), read_de(mname))
    Z = S.design(u, C_COLS + NUIS)
    Rm = np.column_stack([S.midrank(u["Q"]), S.midrank(u["D"]), S.midrank(u["S_ema"])])
    rho = S.rho_p_matrix(Rm, Z)
    de = (u["padj_h"] < FDR).to_numpy()
    t = float(tertile_stat(u["D"].to_numpy()[de], u["agree"].to_numpy()[de])[0]) if de.sum() >= 30 else np.nan
    return dict(kind=kind, human=hname, mouse=mname, n=len(u), rho_p_D=float(rho[1]), rho_p_S=float(rho[2]),
                spearman_l_m=float(spearmanr(u["l"], u["m"])[0]), sign_agree=float(u["agree"].mean()),
                n_human_de=int(de.sum()), tert_diff=t)


def _init(tab):
    global _TAB
    _TAB = tab


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--primary-only", action="store_true")
    a = p.parse_args()
    rng = np.random.default_rng(SEED)
    tab = vocabulary()
    share = hep_share()
    tab["hep_share"] = share.reindex(tab["hsapiens_gene_id"]).to_numpy()
    hp, mp = read_de("human_primary"), read_de("mouse_primary")
    u = universe(tab, hp, mp)
    print(f"universe: vocabulary {len(tab)}; with l and m {u.attrs['n_join']}; detected >=5% both {len(u)}", flush=True)
    fam = dict(family=3, type="hepatocyte", cohort=COHORT, seeds=10)
    rows = []
    for s in ("D", "S_ema", "B1b", "B2b"):
        rows.append({**fam, "contrast": "primary", **rho_row(u, s, a.n_perm, a.n_boot, rng)})
    u2 = u.copy()
    u2["random"] = rng.random(len(u2))
    rows.append({**fam, "contrast": "primary", **rho_row(u2, "random", a.n_perm, min(a.n_boot, 2000), rng)})
    for r in pi_row(u):
        rows.append({**fam, "contrast": "primary", **r})
    h1 = pd.DataFrame(rows)
    h1.to_csv(OUT / "h1.tsv", sep="\t", index=False)
    print(h1[["score", "n", "rho_raw", "rho_pi", "rho_p", "ci_lo", "ci_hi", "p_one_sided", "n_human_de", "tert_diff", "tert_p"]].to_string(), flush=True)
    h2h = head_to_head(u, ["D", "S_ema", "B1b", "B2b"], a.n_boot, rng)
    h2h.to_csv(OUT / "h1_head_to_head.tsv", sep="\t", index=False)
    print(h2h.to_string(), flush=True)
    cols = ["symbol", "hsapiens_gene_id", "mouse_id", "l", "padj_h", "m", "se_m", "padj_m", "Q", "agree", "D", "S_ema", "B1b", "B2b",
            "prior_raw", "log_cp10k_spmean", "detection_spmean", "hep_enrich", "hep_share", "det_h", "det_m"]
    u[cols].to_csv(OUT / "h1_genes.tsv", sep="\t", index=False)
    # named genes (H-D3): descriptive, from the DE tables and D where the vocabulary has the gene
    hs = pd.read_csv(DE / "human_primary.tsv", sep="\t")
    named = pd.DataFrame({"gene_symbol": NAMED}).merge(
        hs[["gene_symbol", "gene_id", "log2fc_mle", "padj", "det_frac"]].rename(
            columns={"log2fc_mle": "l", "padj": "padj_h", "det_frac": "det_h"}), on="gene_symbol", how="left")
    ms = pd.read_csv(DE / "mouse_primary.tsv", sep="\t")
    ms["gene_symbol"] = ms["gene_symbol"].str.upper()
    named = named.merge(ms[["gene_symbol", "log2fc_shrunk", "padj", "det_frac"]].rename(
        columns={"log2fc_shrunk": "m", "padj": "padj_m", "det_frac": "det_m"}), on="gene_symbol", how="left")
    named = named.merge(tab[["symbol", "D", "S_ema", "prior_raw"]], left_on="gene_symbol", right_on="symbol", how="left").drop(columns="symbol")
    named.to_csv(OUT / "named_genes.tsv", sep="\t", index=False)
    if a.primary_only:
        return 0

    # H-D1b
    nb = [nested(u, 2000, 2000, rng, "as specified"),
          nested(u, 2000, 2000, rng, "with |l| and SE(m) in every step", base_extra=NUIS),
          nested(u, 2000, 2000, rng, "S_ema last", last="S_ema")]
    pd.concat(nb).to_csv(OUT / "h1b.tsv", sep="\t", index=False)
    print(pd.concat(nb).to_string(), flush=True)

    # sensitivities: full statistic on each
    B = min(a.n_boot, 2000)
    sens = []
    # the human logFC column: sign and |l| from the fixed-scale shrinkage instead of the MLE
    sens.append(dict(group="human_logfc", contrast="fixed-scale shrunken l", **rho_row(
        universe(tab, hp, mp, l_col="log2fc_fixed"), "D", a.n_perm, B, rng)))
    summ_s = pd.concat([pd.read_csv(DE / f"summary_{g}.tsv", sep="\t") for g in ("sens", "source")])
    anchors = [anchor("primary", u)]

    def add(label: str, uu: pd.DataFrame, group: str, score: str = "D", extra=(), n_perm=None):
        sens.append(dict(group=group, contrast=label, **rho_row(uu, score, a.n_perm if n_perm is None else n_perm, B, rng, extra_cols=extra)))

    for nm in summ_s["name"]:
        if nm.startswith(("human", "source")):
            uu = universe(tab, read_de(nm), mp)
        else:
            uu = universe(tab, hp, read_de(nm))
        gr = {"human_lohd": "leave_one_healthy_out", "mouse_lomo": "leave_one_mouse_per_arm_out"}.get(nm[:10], nm)
        add(nm, uu, gr, n_perm=2000 if gr.startswith("leave_one_mouse") else None)
        anchors.append(anchor(nm, uu))
        pd.DataFrame(sens).to_csv(OUT / "h1_sensitivity.tsv", sep="\t", index=False)

    # bulk human direction
    det = pd.read_csv(DE / "human_primary.tsv", sep="\t").dropna(subset=["gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
    for nm in ("bulk_nash_f0f3", "bulk_nash_all", "bulk_nafl"):
        hb = read_de(nm)
        ub = universe(tab, hb, mp, det_from=det)
        add(f"{nm}__all_universe", ub, "bulk")
        anchors.append(anchor(f"{nm}__all_universe", ub))
        # hepatocyte-dominant: top quartile of hep_enrich of X2_liver_biology, the dominance rule of T4_3_scoreboard. A first
        # rule, hepatocyte share of expression >= 50%, left 34 genes and stays as a disclosed row
        ud = ub[ub["hep_enrich"] >= tab["hep_enrich"].quantile(0.75)].reset_index(drop=True)
        add(f"{nm}__hep_dominant", ud, "bulk")
        anchors.append(anchor(f"{nm}__hep_dominant", ud))
        add(f"{nm}__hep_dominant__S_ema", ud, "bulk", score="S_ema")
        us = ub[ub["hep_share"] >= 0.5].reset_index(drop=True)
        add(f"{nm}__hep_share50", us, "bulk", n_perm=2000)
    # alternative explanations and readout variants on the primary contrast
    for s in [f"Dloso_{x}" for x in (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)] + ["D_integrated"]:
        add(s, u, "seed", score=s, n_perm=2000)
    add("C + tau_shared", u, "extra_covariates", extra=("tau_shared",))
    add("C + hep_enrich", u, "extra_covariates", extra=("hep_enrich",))
    add("C + hep_enrich + tau_shared", u, "extra_covariates", extra=("hep_enrich", "tau_shared"))
    add("margin supported", u[u["margin_support"] == True].reset_index(drop=True), "margin")  # noqa: E712
    add("lacks margin support", u[u["margin_support"] == False].reset_index(drop=True), "margin")  # noqa: E712
    F = u[[f"esm{j}" for j in range(64)]].to_numpy()
    Z = S.design(u, C_COLS + NUIS, extra=F)
    rho = float(S.rho_p_matrix(np.column_stack([S.midrank(u["Q"]), S.midrank(u["D"])]), Z)[1])
    sens.append(dict(group="features", contrast="C + 64 ESM PCs", score="D", n=len(u), rho_p=rho))
    pd.DataFrame(sens).to_csv(OUT / "h1_sensitivity.tsv", sep="\t", index=False)
    pd.DataFrame(anchors).to_csv(OUT / "h1_anchor.tsv", sep="\t", index=False)

    # controls
    ctrl = []
    uh = pd.read_csv(DE / "summary_perm.tsv", sep="\t")["name"].tolist()
    ph = [n for n in uh if n.startswith("perm_human")]
    pm = [n for n in uh if n.startswith("perm_mouse")]
    jobs = [(n, "mouse_primary", "human_permuted", 0) for n in ph] + [("human_primary", n, "mouse_permuted", 0) for n in pm] \
        + [(x, y, "both_permuted", 0) for x, y in zip(ph, pm)] + [("human_primary", "mouse_primary", "observed", 0)]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init, initargs=(tab,)) as ex:
        ctrl = list(ex.map(_quick, jobs))
    pd.DataFrame(ctrl).to_csv(OUT / "h1_controls.tsv", sep="\t", index=False)
    batch = []
    for nm in ("ctrl_gribben_vs_guilliams_cs171", "ctrl_gribben_vs_guilliams_cs170_cs171"):
        ub = universe(tab, read_de(nm), mp, det_from=det)
        batch.append(dict(contrast=nm, **rho_row(ub, "D", a.n_perm, B, rng),
                          spearman_batch_l_vs_disease_l=float(spearmanr(ub["l"], hp["log2fc_mle"].reindex(ub["hsapiens_gene_id"]), nan_policy="omit")[0]),
                          spearman_batch_l_vs_m=float(spearmanr(ub["l"], ub["m"])[0])))
    pd.DataFrame(batch).to_csv(OUT / "h1_batch_control.tsv", sep="\t", index=False)
    (OUT / "hd1_meta.json").write_text(json.dumps(dict(n_perm=a.n_perm, n_boot=a.n_boot, seed=SEED, det=DET, fdr=FDR,
                                                      hep_dominant="top quartile of X2 hep_enrich over the vocabulary", integrated_seeds=INTEGRATED), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
