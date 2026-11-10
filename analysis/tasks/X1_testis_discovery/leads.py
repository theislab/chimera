"""EXPLORATORY: testis discovery leads on the per-gene tables from ``build.py``.

Every lead is reported whatever its outcome. Chimera readouts: D_raw (alpha - raw prior,
median-centered; primary here), D_clamp (alpha - clamped prior, median-centered), S_g_ema and
its stage decomposition (T2_5_celltype_margin identity). Baselines beside each: B1b / B1b_raw (cross-species
cell-type profile correlation, T4_2_gene_baselines), B1c (rank-matched level, T4_2_gene_baselines) and per-stage
pseudobulk differences (build.py). Uncentered delta-alpha is never tested against zero.

Stage groups: each gene's peak stage in HUMAN pseudobulk only (no cross-species
information), among SG, SC, rSD, eSD, Sertoli, Other_somatic; a gene is assigned when
peak - mean over stages >= 1 log2 unit and human detection in the peak stage >= 10 %;
otherwise "broad/low". Sensitivity: >= 2 log2 units and >= 20 %.

Matched excess: for each set gene, the 40 nearest non-set genes in rank space of the
covariates (C = prior_raw, species-balanced log CP10K, species-balanced detection;
C_pat = C + tau; C_pat_pool = C_pat + cell-pooled log mean and detection, which weight
deep cells and abundant stages), one drawn per set gene, B draws; excess = (set mean - null mean) / SD of
all genes; two-sided empirical p.

Reserved sets of the frozen plan (H5/H6: housekeeping, HGNC ribosomal / proteasome /
spliceosome, Hallmark interferon / inflammatory, GO extracellular region) are never formed
or tested here. No gold standard, held-out species, constraint or dN/dS data is read.

Writes results/analysis/X1/*.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.neighbors import NearestNeighbors

from analysis.common import gene_ids

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "X1"
GDIR = OUT / "genes"
STAGES = ["SG", "SC", "rSD", "eSD", "Sertoli", "Other_somatic"]
GROUPS = STAGES + ["broad/low"]
TESTIS = ["test_hmac", "test_hmouse", "test_3sp"]
OTHER = {"test_hmac": ["Macaque"], "test_hmouse": ["Mouse"], "test_3sp": ["Macaque", "Mouse"]}
TISSUE_PAIRS = [("liv_hmonkey", "test_hmac"), ("liv_hmouse", "test_hmouse"), ("liv_4sp", "test_3sp")]
COV_C = ["prior_raw", "log_cp10k_spmean", "detection_spmean"]
COV_PAT = COV_C + ["tau"]
COV_POOL = COV_PAT + ["log_mean", "detection"]  # + cell-pooled mean and detection (depth-weighted)
MATCHINGS = {"C": COV_C, "C_pat": COV_PAT, "C_pat_pool": COV_POOL}
SCORES = ["D_raw", "D_clamp", "S_g_ema", "B1b_raw", "B1b", "B1c"]
K_NN = 40


def load(label: str) -> pd.DataFrame:
    return pd.read_parquet(GDIR / f"{label}.parquet")


def by_ensg(g: pd.DataFrame) -> pd.DataFrame:
    return g[g.hsapiens_gene_id.notna()].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")


def peak_stage(g: pd.DataFrame, spec_min: float = 1.0, det_min: float = 0.10) -> np.ndarray:
    H = g[[f"x_Human_{k}" for k in STAGES]].to_numpy()
    pk = H.argmax(1)
    spec = H.max(1) - H.mean(1)
    fpk = g[[f"f_Human_{k}" for k in STAGES]].to_numpy()[np.arange(len(g)), pk]
    return np.where((spec >= spec_min) & (fpk >= det_min), np.array(STAGES)[pk], "broad/low")


def knn_excess(g: pd.DataFrame, flag: np.ndarray, col: str, covs: list[str], B: int, rng) -> dict:
    v = g[col].to_numpy(dtype=float)
    Zr = g[covs].to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(Zr).all(1)
    f = flag[ok]
    if f.sum() < 5:
        return dict(n=int(f.sum()), excess_d=np.nan, p=np.nan, mean_set=np.nan, mean_null=np.nan)
    Z = np.column_stack([rankdata(Zr[ok, j]) / ok.sum() for j in range(Zr.shape[1])])
    vv = v[ok]
    pool = np.where(~f)[0]
    _, idx = NearestNeighbors(n_neighbors=K_NN).fit(Z[pool]).kneighbors(Z[f])
    nb = vv[pool[idx]]
    draws = nb[np.arange(len(nb))[None, :], rng.integers(0, K_NN, (B, len(nb)))].mean(1)
    obs = vv[f].mean()
    c = draws.mean()
    p = (1 + np.sum(np.abs(draws - c) >= abs(obs - c))) / (B + 1)
    return dict(n=int(f.sum()), mean_set=obs, mean_null=c, excess_d=(obs - c) / vv.std(), p=p)


# ---------------------------------------------------------------------------
# Lead 1 / 4: stage groups, matched excess, attribution
# ---------------------------------------------------------------------------

def stage_tables(labels: list[str], B: int, seed: int):
    grp_rows, ex_rows, att_rows = [], [], []
    weights = pd.read_csv(GDIR / "stage_weights.csv")
    wf = GDIR / "stage_weights_seeds.csv"
    if wf.exists():
        weights = pd.concat([weights, pd.read_csv(wf)]).drop_duplicates(["run_label", "species", "stage"])
    for lab in labels:
        g = load(lab)
        cohort = g["cohort"].iloc[0]
        rng = np.random.default_rng(seed)
        w = weights[weights.run_label == lab].groupby("stage").weight.sum()
        for variant, (sm, dm) in {"main": (1.0, 0.10), "strict": (2.0, 0.20)}.items():
            g["peak"] = peak_stage(g, sm, dm)
            for grp in GROUPS:
                f = (g["peak"] == grp).to_numpy()
                if variant == "main":
                    grp_rows.append(dict(run_label=lab, cohort=cohort, peak=grp, n=int(f.sum()),
                                         median_prior_raw=g.prior_raw[f].median(),
                                         median_log_cp10k=g.log_cp10k_spmean[f].median(),
                                         median_tau=g.tau[f].median(),
                                         frac_margin_support=g.margin_support[f].mean(),
                                         frac_tail=g["tail"][f].mean(),
                                         frac_X=(g.chrom[f] == "X").mean(),
                                         median_B1b_raw=g.B1b_raw[f].median()))
                if grp in STAGES:
                    # per-stage pseudobulk baseline at the group's own stage: minus mean |species-centered
                    # log2 difference to human| (higher = more conserved), and minus mean |log2 FC|
                    g["PB_dc_own"] = -np.mean([g[f"dc_{s}_{grp}"].abs() for s in OTHER[cohort]], axis=0)
                    g["PB_lfc_own"] = -np.mean([g[f"lfc_{s}_{grp}"].abs() for s in OTHER[cohort]], axis=0)
                for col in SCORES + (["PB_dc_own", "PB_lfc_own"] if grp in STAGES else []):
                    for mname, covs in MATCHINGS.items():
                        if variant == "strict" and mname == "C":
                            continue
                        r = knn_excess(g, f, col, covs, B, rng)
                        ex_rows.append(dict(run_label=lab, cohort=cohort, peak_def=variant, peak=grp, score=col,
                                            matching=mname, **r))
            if variant == "main":
                for grp in GROUPS:
                    f = (g["peak"] == grp).to_numpy()
                    Cm = g.loc[f, [f"C_{k}" for k in STAGES]]
                    pos = Cm.clip(lower=0).sum()
                    neg = -Cm.clip(upper=0).sum()
                    for k in STAGES:
                        att_rows.append(dict(run_label=lab, cohort=cohort, peak=grp, stage=k, stage_weight=w[k],
                                             share_pos=pos[f"C_{k}"] / pos.sum(), share_neg=neg[f"C_{k}"] / neg.sum(),
                                             enrich_pos=pos[f"C_{k}"] / pos.sum() / w[k],
                                             enrich_neg=neg[f"C_{k}"] / neg.sum() / w[k]))
        print(f"stage tables: {lab}", flush=True)
    return pd.DataFrame(grp_rows), pd.DataFrame(ex_rows), pd.DataFrame(att_rows)


def sc_gene_table() -> pd.DataFrame:
    """Genes with a human spermatocyte peak in any testis cohort, readouts side by side."""
    cols = ["symbol", "chrom", "prior_raw", "alpha", "D_raw", "D_clamp", "S_g_ema", "margin_support", "B1b_raw",
            "B1c", "C_SC", "C_rSD", "C_eSD"]
    parts = []
    for c in TESTIS:
        g = load(c)
        g["peak"] = peak_stage(g)
        g = by_ensg(g)
        g["share_SC"] = g["C_SC"] / g[[f"C_{k}" for k in STAGES]].abs().sum(1)
        parts.append(g[cols + ["share_SC", "peak"]].add_suffix(f"__{c}"))
    t = pd.concat(parts, axis=1)
    sc = (t[[f"peak__{c}" for c in TESTIS]] == "SC").any(axis=1)
    t = t[sc].copy()
    t["symbol"] = t[[f"symbol__{c}" for c in TESTIS]].bfill(axis=1).iloc[:, 0]
    t["n_cohorts_SC_peak"] = (t[[f"peak__{c}" for c in TESTIS]] == "SC").sum(1)
    return t.sort_values("D_raw__test_hmouse", ascending=False).reset_index()


def lead4_table(labels: list[str]) -> pd.DataFrame:
    """High prior (top tertile), lowest 5 % D within it, with margin support: counts by stage group."""
    rows = []
    for lab in labels:
        g = load(lab)
        g["peak"] = peak_stage(g)
        hi = g.prior_raw >= g.prior_raw.quantile(2 / 3)
        for col in ("D_raw", "D_clamp"):
            sel = hi & (g[col] <= g.loc[hi, col].quantile(0.05)) & g.margin_support
            for grp in GROUPS:
                rows.append(dict(run_label=lab, score=col, peak=grp, n_selected=int((sel & (g.peak == grp)).sum()),
                                 n_high_prior_in_group=int((hi & (g.peak == grp)).sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lead 2: X chromosome
# ---------------------------------------------------------------------------

def x_tables(B: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for lab in TESTIS + [lv for lv, _ in TISSUE_PAIRS]:
        g = load(lab)
        g = g[g.chrom.isin([str(i) for i in range(1, 23)] + ["X"])].reset_index(drop=True)
        testis = lab.startswith("test")
        if testis:
            g["peak"] = peak_stage(g)
        scores = ["D_raw", "D_clamp", "S_g_ema"] + (["B1b_raw", "B1b", "B1c"] if testis else [])
        subsets = {"all": np.ones(len(g), bool)}
        if testis:
            subsets.update({f"peak={k}": (g.peak == k).to_numpy() for k in ("SG", "Sertoli", "Other_somatic",
                                                                                "broad/low")})
        for sname, m in subsets.items():
            gg = g[m].reset_index(drop=True)
            fX = (gg.chrom == "X").to_numpy()
            for col in scores:
                for mname, covs in (("C", COV_C), ("C_pat", COV_PAT)):
                    if mname == "C_pat" and "tau" not in gg:
                        continue
                    r = knn_excess(gg, fX, col, covs, B, rng)
                    rows.append(dict(run_label=lab, tissue="testis" if testis else "liver", subset=sname,
                                     score=col, matching=mname, **r))
    return pd.DataFrame(rows)


def x_on_shared_genes(B: int, seed: int) -> pd.DataFrame:
    """Same X-linked genes in liver and testis (same species pair), matched excess in each tissue."""
    rows = []
    rng = np.random.default_rng(seed)
    for lv, ts in TISSUE_PAIRS:
        l, t = by_ensg(load(lv)), by_ensg(load(ts))
        common = l.index.intersection(t.index)
        for tissue, g in (("liver", l.loc[common]), ("testis", t.loc[common])):
            g = g[g.chrom.isin([str(i) for i in range(1, 23)] + ["X"])].reset_index()
            fX = (g.chrom == "X").to_numpy()
            for col in ("D_raw", "D_clamp", "S_g_ema"):
                r = knn_excess(g, fX, col, COV_C, B, rng)
                rows.append(dict(pair=f"{lv}|{ts}", tissue=tissue, score=col, matching="C", **r))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lead 3: divergence depth (H+Macaque vs H+Mouse)
# ---------------------------------------------------------------------------

def depth_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a, b = by_ensg(load("test_hmac")), by_ensg(load("test_hmouse"))
    j = a.join(b, lsuffix="_mac", rsuffix="_mus", how="inner")
    rows = []
    for col in ["prior_raw", "alpha", "D_raw", "D_clamp", "S_g_ema", "B1b_raw", "B1b", "B1c", "log_cp10k_spmean"]:
        x, y = j[f"{col}_mac"], j[f"{col}_mus"]
        ok = x.notna() & y.notna()
        rows.append(dict(quantity=col, n=int(ok.sum()), spearman_mac_vs_mus=spearmanr(x[ok], y[ok])[0],
                         median_mac=x[ok].median(), median_mus=y[ok].median(),
                         frac_mus_below_mac=float((y[ok] < x[ok]).mean())))
    rows.append(dict(quantity="frac S_g_ema < 0", n=len(j), median_mac=float((j.S_g_ema_mac < 0).mean()),
                     median_mus=float((j.S_g_ema_mus < 0).mean())))
    pair = pd.DataFrame(rows)
    # Per-stage baseline divergence and one-fit species contrast on test_3sp
    t = load("test_3sp")
    st_rows = []
    for k in STAGES:
        ex = (t[f"f_Human_{k}"] >= 0.05) | (t[f"f_Macaque_{k}"] >= 0.05) | (t[f"f_Mouse_{k}"] >= 0.05)
        x = t[ex]
        st_rows.append(dict(stage=k, n_expr=int(ex.sum()),
                            median_absdc_Macaque=x[f"dc_Macaque_{k}"].abs().median(),
                            median_absdc_Mouse=x[f"dc_Mouse_{k}"].abs().median(),
                            frac_absdc_gt1_Macaque=(x[f"dc_Macaque_{k}"].abs() > 1).mean(),
                            frac_absdc_gt1_Mouse=(x[f"dc_Mouse_{k}"].abs() > 1).mean(),
                            frac_Sbar_neg_Macaque=(x[f"Sbar_Macaque_{k}"] < 0).mean(),
                            frac_Sbar_neg_Mouse=(x[f"Sbar_Mouse_{k}"] < 0).mean(),
                            frac_Sbarwt_Mouse_below_Macaque=(x[f"Sbarwt_Mouse_{k}"] < x[f"Sbarwt_Macaque_{k}"]).mean(),
                            rho_Sbarwt_absdc_Macaque=spearmanr(x[f"Sbarwt_Macaque_{k}"], x[f"dc_Macaque_{k}"].abs())[0],
                            rho_Sbarwt_absdc_Mouse=spearmanr(x[f"Sbarwt_Mouse_{k}"], x[f"dc_Mouse_{k}"].abs())[0]))
    stage = pd.DataFrame(st_rows)
    # Genes deployed-conserved with macaque but not mouse: Chimera (S_g sign + support) vs baseline (B1b_raw)
    sel = []
    for name, m in {
        "Chimera: S_mac>0 supported, S_mus<0 supported": (j.S_g_ema_mac > 0) & j.margin_support_mac
        & (j.S_g_ema_mus < 0) & j.margin_support_mus,
        "Chimera: D_raw_mac - D_raw_mus top 1%": (j.D_raw_mac - j.D_raw_mus) >= (j.D_raw_mac - j.D_raw_mus).quantile(0.99),
        "Baseline: B1b_raw_mac>0.8, B1b_raw_mus<0": (j.B1b_raw_mac > 0.8) & (j.B1b_raw_mus < 0),
    }.items():
        sel.append(dict(selection=name, n=int(m.sum())))
    return pair, stage, pd.DataFrame(sel)


# ---------------------------------------------------------------------------
# Lead 5: mtDNA stage and species drivers
# ---------------------------------------------------------------------------

def mtdna_table() -> pd.DataFrame:
    rows = []
    for c in TESTIS:
        g = load(c)
        mt = g[g.symbol.astype(str).str.startswith("MT-")]
        spp = ["Human"] + OTHER[c]
        for _, r in mt.iterrows():
            d = dict(cohort=c, symbol=r.symbol, D_raw=r.D_raw, D_clamp=r.D_clamp, S_g_ema=r.S_g_ema,
                     margin_support=r.margin_support, B1b_raw=r.B1b_raw)
            for k in STAGES:
                d[f"C_{k}"] = r[f"C_{k}"]
            for s in spp:
                d[f"C_species_{s}"] = sum(r[f"C_{s}_{k}"] for k in STAGES)
            for s in OTHER[c]:
                for k in STAGES:
                    d[f"lfc_{s}_{k}"] = r[f"lfc_{s}_{k}"]
                    d[f"det_{s}_{k}"] = r[f"f_{s}_{k}"]
            for k in STAGES:
                d[f"det_Human_{k}"] = r[f"f_Human_{k}"]
            rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lead 6: tissue discordance by testis stage group
# ---------------------------------------------------------------------------

def tissue_table() -> pd.DataFrame:
    rows = []
    for lv, ts in TISSUE_PAIRS:
        l, t = by_ensg(load(lv)), by_ensg(load(ts))
        t["peak"] = peak_stage(t.reset_index()).astype(str)
        j = l.join(t, lsuffix="_l", rsuffix="_t", how="inner")
        for grp in GROUPS + ["ALL"] + [f"drop {k}" for k in STAGES]:
            if grp == "ALL":
                m = np.ones(len(j), bool)
            elif grp.startswith("drop "):
                m = (j.peak != grp[5:]).to_numpy()
            else:
                m = (j.peak == grp).to_numpy()
            d = dict(pair=f"{lv}|{ts}", genes=grp, n=int(m.sum()))
            for col in ("D_raw", "D_clamp", "S_g_ema", "alpha", "prior_raw", "log_mean"):
                d[f"rho_{col}"] = spearmanr(j[f"{col}_l"][m], j[f"{col}_t"][m])[0]
            d["rho_Draw_liver_vs_prior"] = spearmanr(j.D_raw_l[m], j.prior_raw_l[m])[0]
            d["rho_Draw_testis_vs_prior"] = spearmanr(j.D_raw_t[m], j.prior_raw_t[m])[0]
            rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Robustness: per seed and multi-seed consensus
# ---------------------------------------------------------------------------

def seed_robustness(seed_labels: list[str], B: int, seed: int) -> pd.DataFrame:
    """Matched excess (C_pat, main peak definition) per seed and on the consensus over seeds.

    Consensus per cohort: mean over the canonical run and every finished seed of
    alpha - prior_raw (and alpha - prior, and S_g_ema), minus the cohort median.
    """
    rows = []
    by_cohort: dict[str, list[str]] = {}
    for lab in seed_labels:
        by_cohort.setdefault(lab.rsplit("_s", 1)[0], []).append(lab)
    for cohort, labs in by_cohort.items():
        base = load(cohort)
        base["peak"] = peak_stage(base)
        runs = [cohort] + labs
        tabs = {r: load(r).set_index("var_name") for r in runs}
        cons = base.set_index("var_name").copy()
        for col, src in (("D_raw", "delta_alpha_raw"), ("D_clamp", "delta_alpha"), ("S_g_ema", "S_g_ema")):
            m = pd.concat([tabs[r][src] for r in runs], axis=1).mean(1)
            cons[col] = m - (m.median() if col != "S_g_ema" else 0.0)
        cons = cons.reset_index()
        for lab, g in [(r, tabs[r].reset_index()) for r in runs] + [(f"{cohort}_consensus{len(runs)}", cons)]:
            rng = np.random.default_rng(seed)
            g["peak"] = peak_stage(g)
            for grp in STAGES:
                f = (g["peak"] == grp).to_numpy()
                for col in ("D_raw", "D_clamp", "S_g_ema"):
                    r = knn_excess(g, f, col, COV_PAT, B, rng)
                    rows.append(dict(cohort=cohort, run_label=lab, n_runs=len(runs) if "consensus" in lab else 1,
                                     peak=grp, score=col, matching="C_pat", **r))
        # cross-seed agreement of D_raw residual ranks
        for i, r1 in enumerate(runs):
            for r2 in runs[i + 1:]:
                rows.append(dict(cohort=cohort, run_label=f"{r1}|{r2}", peak="(all genes)", score="spearman D_raw",
                                 excess_d=spearmanr(tabs[r1].delta_alpha_raw, tabs[r2].delta_alpha_raw)[0]))
                rows.append(dict(cohort=cohort, run_label=f"{r1}|{r2}", peak="(all genes)", score="spearman D_clamp",
                                 excess_d=spearmanr(tabs[r1].delta_alpha, tabs[r2].delta_alpha)[0]))
        print(f"seed robustness: {cohort} {runs}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-runs", nargs="*", default=[], help="extra seed labels with gene tables in genes/")
    args = ap.parse_args(argv)
    labels = TESTIS
    grp, ex, att = stage_tables(labels, args.n_null, args.seed)
    grp.to_csv(OUT / "stage_groups.csv", index=False)
    ex.to_csv(OUT / "stage_excess.csv", index=False)
    att.to_csv(OUT / "stage_attribution.csv", index=False)
    sc_gene_table().to_csv(OUT / "sc_peak_genes.csv", index=False)
    lead4_table(TESTIS).to_csv(OUT / "lead4_high_prior_low_D_counts.csv", index=False)
    x_tables(args.n_null, args.seed).to_csv(OUT / "x_chromosome.csv", index=False)
    x_on_shared_genes(args.n_null, args.seed).to_csv(OUT / "x_chromosome_shared_genes.csv", index=False)
    pair, stage, sel = depth_tables()
    pair.to_csv(OUT / "depth_pairs.csv", index=False)
    stage.to_csv(OUT / "depth_stage_3sp.csv", index=False)
    sel.to_csv(OUT / "depth_selection_counts.csv", index=False)
    mtdna_table().to_csv(OUT / "mtdna_stage.csv", index=False)
    tissue_table().to_csv(OUT / "tissue_discordance_by_stage.csv", index=False)
    seeds = [s for s in args.seed_runs if (GDIR / f"{s}.parquet").exists()]
    if seeds:
        seed_robustness(seeds, args.n_null, args.seed).to_csv(OUT / "seed_robustness.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
