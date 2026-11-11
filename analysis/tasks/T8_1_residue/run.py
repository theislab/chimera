"""Specificity of the residue-level GIN attention (the residue layer of the protein figure), on cached inputs only.

Per-residue attention is recomputed from cached ESM-2 35M human embeddings, the 64-PC IG
attribution of each gene and the 480 -> 64 bridge:

    D_g = W_pc @ (a_g / sqrt(eig)),    attn_r = | (e_r - mean_480) . D_g |

Step 1 checks that this reproduces the cached ``abs_attn`` (per-gene Pearson > 0.999).

Two nulls replace a_g and recompute every residue's attention, ``--draws`` times each, fixed seed:
- gene-swap: a derangement of the 70 genes; each gene takes another gene's attribution;
- random direction: a Gaussian direction in the 64-PC attribution space, rescaled to ||a_g||.

Statistic per gene: difference of category medians of attn. ``scaled`` divides by the gene's
median attn (swapped attributions change amplitude); ``unscaled`` is the
raw difference (the per-gene test of the residue figures). Across genes: median, fraction > 0, Wilcoxon p.
Empirical p: (1 + #null with |stat| >= |obs|) / (1 + draws), two-sided; one-sided in the
observed direction is also written.

Rows: five contrasts on all genes and hydrophobic-vs-polar on the curated M class and on
Delta-alpha < -0.05 (the two class rows of A8), plus data-defined classes (QC rule, sign bins).
Also: R^2_marg of the published MixedLM split into residue-level and gene-level parts.

Inputs (``notebooks/cache/gin_attribution/``): ``L_analysis/{residue_table.pkl,attr_l.npz,
delta_alpha_attention.pkl,mixedlm_meta.json}``, ``embeddings/<gene>__hsapiens.npy``,
``pca_basis_v3__4sp.npz``, ``bridge/W_pc_v2_panc_hmonkey.npz`` (identical W to the ``panc_4sp``
file used for CPS1; checked).
Outputs: ``results/analysis/T8_1/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.common.outputs import TaskOutput, add_common_args  # noqa: E402
from analysis.common import gene_ids  # noqa: E402

CACHE = gene_ids.REPO_ROOT / "notebooks" / "cache" / "gin_attribution"
SEED = 20261101

CONTRASTS = [  # key, column, a, b, label
    ("hyd_pos", "chem", "hydrophobic", "positive", "Hydrophobic vs positive"),
    ("hyd_neg", "chem", "hydrophobic", "negative", "Hydrophobic vs negative"),
    ("int_bur", "sasa_bin", "intermediate", "buried", "Intermediate vs buried"),
    ("exp_bur", "sasa_bin", "exposed", "buried", "Exposed vs buried"),
    ("hyd_pol", "chem", "hydrophobic", "polar", "Hydrophobic vs polar"),
]
KEYS = [c[0] for c in CONTRASTS]


def sasa_bin(rs):
    return np.where(rs < 0.20, "buried", np.where(rs < 0.50, "intermediate", "exposed"))


def plddt_bin(p):  # builder section L.2
    return np.where(p > 90, "very high (>90)", np.where(p > 70, "high (70–90)",
                    np.where(p > 50, "low (50–70)", "very low (<50)")))


def load():
    d = CACHE / "L_analysis"
    rt = pd.read_pickle(d / "residue_table.pkl").copy()
    rt["sasa_bin"] = sasa_bin(rt.rsasa.values)
    rt["plddt_bin"] = plddt_bin(rt.plddt.values)
    A = np.load(d / "attr_l.npz")
    eig = np.load(CACHE / "pca_basis_v3__4sp.npz")["eig"]
    b = np.load(CACHE / "bridge" / "W_pc_v2_panc_hmonkey.npz")
    b4 = np.load(CACHE / "bridge" / "W_pc_v2_panc_4sp.npz")
    assert np.array_equal(b["W_pc"], b4["W_pc"]) and np.array_equal(b["mean_480"], b4["mean_480"])
    da = pd.read_pickle(d / "delta_alpha_attention.pkl").set_index("gene")
    meta = json.loads((d / "mixedlm_meta.json").read_text())
    return rt, A, eig, b, da, meta


class Attn:
    """Recompute |attention| per residue for any attribution vector."""

    def __init__(self, rt, eig, bridge):
        self.genes = sorted(rt.gene.unique())
        self.W = bridge["W_pc"].astype(np.float64)
        self.mu = bridge["mean_480"].astype(np.float64)
        self.isq = 1.0 / np.sqrt(np.maximum(eig.astype(np.float64), 1e-12))
        self.E, self.sub = {}, {}
        for g, s in rt.groupby("gene"):
            e = np.load(CACHE / "embeddings" / f"{g}__hsapiens.npy").astype(np.float64)
            self.E[g] = e[s.residue.values - 1] - self.mu
            self.sub[g] = s
        # category masks per gene
        self.masks = {g: {(col, lev): (self.sub[g][col].values == lev)
                          for col in ("chem", "sasa_bin")
                          for lev in self.sub[g][col].unique()} for g in self.genes}

    def attn(self, g, a64):
        return np.abs(self.E[g] @ (self.W @ (a64 * self.isq)))

    def per_gene(self, attrs):
        """(n_genes, n_contrasts) scaled and unscaled per-gene differences of medians."""
        sc = np.full((len(self.genes), len(CONTRASTS)), np.nan)
        un = sc.copy()
        for i, g in enumerate(self.genes):
            v = self.attn(g, attrs[g])
            m = self.masks[g]
            med = np.median(v) + 1e-9
            for j, (_, col, a, b, _) in enumerate(CONTRASTS):
                ma, mb = m.get((col, a)), m.get((col, b))
                if ma is None or mb is None or not ma.any() or not mb.any():
                    continue
                dif = np.median(v[ma]) - np.median(v[mb])
                un[i, j] = dif
                sc[i, j] = dif / med
        return sc, un


def summarise(x):
    x = x[~np.isnan(x)]
    p = wilcoxon(x).pvalue if len(x) >= 5 else np.nan
    return float(np.median(x)), float((x > 0).mean()), float(p), len(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws", type=int, default=2000, help="draws per null (>= 1000)")
    ap.add_argument("--seed", type=int, default=SEED)
    add_common_args(ap)
    args = ap.parse_args()
    assert args.draws >= 1000
    out = TaskOutput("T8_1", args.outdir, prerun=args.prerun, force=args.force)
    t0 = time.time()

    rt, A, eig, bridge, da, meta = load()
    at = Attn(rt, eig, bridge)
    genes = at.genes
    G = len(genes)
    obs_attr = {g: A[g][:64].astype(np.float64) for g in genes}

    # ---- step 1: reproduction of abs_attn
    rep = []
    for g in genes:
        x = at.attn(g, obs_attr[g])
        y = at.sub[g].abs_attn.values
        rep.append(dict(gene=g, n_res=len(y), pearson=np.corrcoef(x, y)[0, 1],
                        max_abs_diff=float(np.max(np.abs(x - y))),
                        max_rel_diff=float(np.max(np.abs(x - y)) / (np.max(np.abs(y)) + 1e-12))))
    rep = pd.DataFrame(rep)
    out.csv(rep, "abs_attn_reproduction.csv")
    rmin = rep.pearson.min()
    print(f"abs_attn reproduction: per-gene Pearson median {rep.pearson.median():.6f}, "
          f"min {rmin:.6f}, max |diff| {rep.max_abs_diff.max():.2e}")
    if not rmin > 0.999:
        raise SystemExit("abs_attn does not reproduce (min Pearson <= 0.999); stopping")

    # ---- gene classes
    g = da.loc[genes].copy()
    g["qc_rule"] = np.where((g.prior >= 0.80) & (g.delta_alpha <= -0.10), "M_rule",
                            np.where((g.prior <= 0.65) & (g.delta_alpha >= 0.10), "F_rule", "none"))
    g["sign_bin"] = np.where(g.delta_alpha < -0.05, "neg", np.where(g.delta_alpha > 0.05, "pos", "near"))
    gi = {x: i for i, x in enumerate(genes)}
    row_defs = [(k, lab, "all", np.arange(G)) for k, _, _, _, lab in CONTRASTS]
    classes = [
        ("curated_M", "Hydrophobic vs polar, curated M", g.direction == "M", True),
        ("da_lt_-0.05", "Hydrophobic vs polar, Δα<-0.05", g.delta_alpha < -0.05, True),
        ("curated_F", "Hydrophobic vs polar, curated F", g.direction == "F", False),
        ("curated_ref", "Hydrophobic vs polar, curated ref", g.direction == "ref", False),
        ("da_gt_0.05", "Hydrophobic vs polar, Δα>0.05", g.delta_alpha > 0.05, False),
        ("da_near", "Hydrophobic vs polar, |Δα|≤0.05", g.sign_bin == "near", False),
        ("qc_rule_M", "Hydrophobic vs polar, QC-rule M", g.qc_rule == "M_rule", False),
    ]
    primary_groups = {"all"} | {c[0] for c in classes if c[3]}
    for cname, lab, mask, _ in classes:
        idx = np.array([gi[x] for x in mask.index[mask.values]])
        if len(idx) >= 5:
            row_defs.append(("hyd_pol", lab, cname, idx))
    hp_j = KEYS.index("hyd_pol")

    def row_stats(sc, un):
        res = []
        for key, lab, grp, idx in row_defs:
            j = KEYS.index(key)
            res.append((summarise(sc[idx, j]), summarise(un[idx, j])))
        return res

    sc0, un0 = at.per_gene(obs_attr)
    obs = row_stats(sc0, un0)

    # ---- nulls
    rng = np.random.default_rng(args.seed)
    nulls = {}
    norms = {x: np.linalg.norm(obs_attr[x]) for x in genes}
    for null in ("gene_swap", "random_direction"):
        rec = []
        t = time.time()
        for _ in range(args.draws):
            if null == "gene_swap":
                p = rng.permutation(G)
                while (p == np.arange(G)).any():
                    p = rng.permutation(G)
                attrs = {x: obs_attr[genes[p[i]]] for i, x in enumerate(genes)}
            else:
                attrs = {}
                for x in genes:
                    v = rng.standard_normal(64)
                    attrs[x] = v / np.linalg.norm(v) * norms[x]
            rec.append(row_stats(*at.per_gene(attrs)))
        nulls[null] = rec
        print(f"{null}: {args.draws} draws in {time.time() - t:.0f}s")

    rows = []
    for r, (key, lab, grp, idx) in enumerate(row_defs):
        for s, sname in enumerate(("scaled", "unscaled")):
            o_med, o_fp, o_p, o_n = obs[r][s]
            for null, rec in nulls.items():
                a = np.array([d[r][s][:3] for d in rec])
                sig_same = ((a[:, 2] < 0.05) & (np.sign(a[:, 0]) == np.sign(o_med))).mean()
                rows.append(dict(
                    row=lab, contrast=key, genes=grp, primary=grp in primary_groups, n_genes=o_n, statistic=sname, null=null, draws=len(a),
                    observed=o_med, obs_frac_pos=o_fp, obs_wilcoxon_p=o_p,
                    null_median=np.median(a[:, 0]), null_q025=np.quantile(a[:, 0], 0.025),
                    null_q975=np.quantile(a[:, 0], 0.975),
                    frac_null_p05_same_sign=sig_same,
                    emp_p_two_sided=(1 + (np.abs(a[:, 0]) >= abs(o_med)).sum()) / (1 + len(a)),
                    emp_p_one_sided=(1 + (np.sign(o_med) * a[:, 0] >= abs(o_med)).sum()) / (1 + len(a)),
                ))
    res = pd.DataFrame(rows)
    out.csv(res, "residue_specificity.csv")

    # ---- data-defined classes
    ct = pd.concat({"qc_rule": pd.crosstab(g.direction, g.qc_rule),
                    "sign_bin": pd.crosstab(g.direction, g.sign_bin)}, axis=1)
    out.csv(ct, "class_crosstab.csv", index=True)
    hp_s, hp_u = sc0[:, hp_j], un0[:, hp_j]
    rho_s = spearmanr(hp_s, g.delta_alpha.values)
    rho_u = spearmanr(hp_u, g.delta_alpha.values)
    pg = g[["direction", "qc_rule", "sign_bin", "prior", "delta_alpha"]].copy()
    for j, k in enumerate(KEYS):
        pg[f"{k}_scaled"] = sc0[:, j]
        pg[f"{k}_unscaled"] = un0[:, j]
    out.csv(pg.reset_index().rename(columns={"index": "gene"}), "per_gene_observed.csv")

    # ---- mechanism: attention follows the embedding norm
    rho_norm, nrm = [], []
    for x in genes:
        n = np.linalg.norm(at.E[x], axis=1)
        rho_norm.append(spearmanr(n, at.attn(x, obs_attr[x]))[0])
        nrm.append(pd.DataFrame({"chem": at.sub[x].chem.values, "sasa_bin": at.sub[x].sasa_bin.values, "norm": n}))
    nrm = pd.concat(nrm)
    swap_corr = []
    for i, x in enumerate(genes):
        h = genes[(i + 7) % G]
        swap_corr.append(spearmanr(at.attn(x, obs_attr[x]), at.attn(x, obs_attr[h]))[0])

    # ---- R^2_marg split (published MixedLM, same formula and coding as builder section L.4)
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings("ignore")
    df = rt.copy()
    df["log_abs_attn"] = np.log(df.abs_attn + 1e-6)
    for col in ("plddt", "hydropathy", "rel_pos", "entropy", "esm2_pc_frac"):
        df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
    df["ss"] = pd.Categorical(df.ss, categories=["C", "H", "E"])
    df["sasa_bin"] = pd.Categorical(df.sasa_bin, categories=["intermediate", "buried", "exposed"])
    df["chem"] = pd.Categorical(df.chem, categories=["polar", "hydrophobic", "positive", "negative"])
    df["plddt_bin"] = pd.Categorical(df.plddt_bin, categories=["high (70–90)", "very low (<50)",
                                                               "low (50–70)", "very high (>90)"])
    res_terms = "C(ss) + C(sasa_bin) + C(plddt_bin) + hydropathy_z + C(chem) + entropy_z + rel_pos_z"
    vy = float(np.var(df.log_abs_attn))
    r2rows = []
    fits = {}
    for name, rhs in [("full", res_terms + " + esm2_pc_frac_z"), ("residue_terms_only", res_terms),
                      ("gene_covariate_only", "esm2_pc_frac_z")]:
        f = smf.mixedlm("log_abs_attn ~ " + rhs, df, groups=df["gene"]).fit(method="bfgs", maxiter=1000, disp=False)
        # statsmodels 0.15 lbfgs stops at the zero-variance boundary on this fit; bfgs and powell agree
        fits[name] = f
        vf = float(np.var(f.predict(df)))
        re, sc_ = float(f.cov_re.iloc[0, 0]), float(f.scale)
        r2rows.append(dict(model=name, r2_marginal=vf / vy, r2_conditional=(vf + re) / (vf + re + sc_),
                           icc=re / (re + sc_)))
    full = fits["full"]
    X = full.model.exog
    names = full.model.exog_names
    beta = full.fe_params.values
    jpc = names.index("esm2_pc_frac_z")
    jres = [i for i, n in enumerate(names) if n not in ("Intercept", "esm2_pc_frac_z")]
    lp_res = X[:, jres] @ beta[jres]
    lp_pc = X[:, jpc] * beta[jpc]
    gid = pd.factorize(df.gene)[0]
    cnt = np.bincount(gid)
    lp_res_between = (np.bincount(gid, lp_res) / cnt)[gid]
    r2rows += [
        dict(model="full: residue-term part var(Xb_res)/var(y)", r2_marginal=np.var(lp_res) / vy),
        dict(model="full: within-gene residue part", r2_marginal=np.var(lp_res - lp_res_between) / vy),
        dict(model="full: gene-level part var(b_pc*pc)/var(y)", r2_marginal=np.var(lp_pc) / vy),
        dict(model="full: 2cov(residue, gene)", r2_marginal=2 * np.cov(lp_res, lp_pc, bias=True)[0, 1] / vy),
    ]
    r2 = pd.DataFrame(r2rows)
    out.csv(r2, "r2_marginal_split.csv")
    r2_full = r2.loc[r2.model == "full", "r2_marginal"].item()
    if abs(r2_full - meta["r2_marginal"]) > 0.002:
        raise SystemExit(f"MixedLM refit R2_marg {r2_full:.4f} does not match published {meta['r2_marginal']:.4f}")
    print(r2.to_string())

    # ---- summary
    def pick(contrast, grp, null, stat="scaled"):
        q = res[(res.contrast == contrast) & (res.genes == grp) & (res.null == null) & (res.statistic == stat)]
        return q.iloc[0]

    lines = ["# T8.1 residue specificity check", "",
             f"Draws per null: {args.draws}; seed {args.seed}; {G} genes, {len(rt)} human residues.",
             f"abs_attn reproduction: per-gene Pearson median {rep.pearson.median():.6f}, min {rmin:.6f}; "
             f"max |diff| {rep.max_abs_diff.max():.2e}.",
             f"Published MixedLM R²_marg {meta['r2_marginal']:.4f}; refit here {r2_full:.4f}.", "",
             "## Primary rows (scaled statistic; A8 section 2.3 layout)", "",
             "| Row | n | Observed | Swap null median [95%] | Swaps p<0.05 same sign | Emp. p swap / random (two-sided) | one-sided swap / random |",
             "|---|---|---|---|---|---|---|"]
    prim_rows = [(k, "all") for k in KEYS] + [("hyd_pol", "curated_M"), ("hyd_pol", "da_lt_-0.05")]
    for k, grp in prim_rows:
        s, rnd = pick(k, grp, "gene_swap"), pick(k, grp, "random_direction")
        lines.append(f"| {s.row} | {s.n_genes} | {s.observed:.3f} | {s.null_median:.3f} [{s.null_q025:.3f}, {s.null_q975:.3f}] | "
                     f"{100 * s.frac_null_p05_same_sign:.0f}% | {s.emp_p_two_sided:.4f} / {rnd.emp_p_two_sided:.4f} | "
                     f"{s.emp_p_one_sided:.4f} / {rnd.emp_p_one_sided:.4f} |")
    lines += ["", "## Same rows, unscaled per-gene difference (published statistic)", "",
              "| Row | n | Observed | Wilcoxon p | Emp. p swap / random (two-sided) |", "|---|---|---|---|---|"]
    for k, grp in prim_rows:
        s, rnd = pick(k, grp, "gene_swap", "unscaled"), pick(k, grp, "random_direction", "unscaled")
        lines.append(f"| {s.row} | {s.n_genes} | {s.observed:.3f} | {s.obs_wilcoxon_p:.4f} | "
                     f"{s.emp_p_two_sided:.4f} / {rnd.emp_p_two_sided:.4f} |")
    lines += ["", "## Data-defined classes, hydrophobic vs polar (scaled)", "",
              "| Grouping | n | Observed | Wilcoxon p | Emp. p swap / random (two-sided) |", "|---|---|---|---|---|"]
    for cname, lab, _, _ in classes:
        q = res[(res.genes == cname) & (res.statistic == "scaled")]
        if len(q):
            s, rnd = q[q.null == "gene_swap"].iloc[0], q[q.null == "random_direction"].iloc[0]
            lines.append(f"| {lab} | {s.n_genes} | {s.observed:.3f} | {s.obs_wilcoxon_p:.4f} | "
                         f"{s.emp_p_two_sided:.4f} / {rnd.emp_p_two_sided:.4f} |")
    lines += ["", f"Continuous: Spearman(per-gene hydrophobic-polar, Δα) scaled ρ={rho_s[0]:.3f} p={rho_s[1]:.3f}; "
              f"unscaled ρ={rho_u[0]:.3f} p={rho_u[1]:.3f} (n={G}).", "",
              "Curated class vs data (`class_crosstab.csv`):", "", "```", ct.to_string(), "```", "",
              "## R²_marg split", "", "```", r2.to_string(index=False), "```", "",
              "## Mechanism", "",
              f"Spearman(|attn|, ||e-μ||) per gene: median {np.median(rho_norm):.3f}. "
              f"True vs another gene's attribution, per-residue Spearman median {np.median(swap_corr):.3f}.",
              f"Median ||e-μ|| by chemistry: {nrm.groupby('chem')['norm'].median().round(2).to_dict()}; "
              f"by SASA bin: {nrm.groupby('sasa_bin')['norm'].median().round(2).to_dict()}.", ""]

    hp_all, hp_m, hp_n = pick("hyd_pol", "all", "gene_swap"), pick("hyd_pol", "curated_M", "gene_swap"), \
        pick("hyd_pol", "da_lt_-0.05", "gene_swap")
    hp_m_r, hp_n_r = pick("hyd_pol", "curated_M", "random_direction"), pick("hyd_pol", "da_lt_-0.05", "random_direction")
    hpos = pick("hyd_pos", "all", "gene_swap")
    gen = [pick(k, "all", "gene_swap") for k in ("hyd_pos", "hyd_neg", "int_bur", "exp_bur")]
    r_res = r2.loc[r2.model == "residue_terms_only", "r2_marginal"].item()
    r_gene = r2.loc[r2.model == "gene_covariate_only", "r2_marginal"].item()
    sentence = (
        f"Replacing each protein's attribution with another protein's ({args.draws:,} gene swaps) or with a random "
        f"direction of equal norm ({args.draws:,} draws) reproduces the hydrophobic-versus-charged and "
        f"solvent-exposure contrasts (empirical p {min(x.emp_p_two_sided for x in gen):.2f}-"
        f"{max(x.emp_p_two_sided for x in gen):.2f}; {100 * hpos.frac_null_p05_same_sign:.0f}% of swaps recover "
        f"hydrophobic-versus-positive at p<0.05), so these follow the ESM-2 embedding rather than the GIN; only "
        f"hydrophobic-versus-polar in negative-Δα proteins exceeds both nulls (curated M, n={hp_m.n_genes}: "
        f"p={hp_m.emp_p_one_sided:.3f} swap, {hp_m_r.emp_p_one_sided:.3f} random; Δα<-0.05, n={hp_n.n_genes}: "
        f"p={hp_n.emp_p_one_sided:.3f}, {hp_n_r.emp_p_one_sided:.3f}), and of R²_marg={r2_full:.3f} the residue-level "
        f"terms alone give {r_res:.3f}, the rest coming from the gene-level ESM-2 PC fraction ({r_gene:.3f} alone).")
    hp_p, hp_p_r = pick("hyd_pol", "da_gt_0.05", "gene_swap"), pick("hyd_pol", "da_gt_0.05", "random_direction")
    caveat = (f"Caveats: the shift is not sign-specific (Δα>0.05, n={hp_p.n_genes}: median {hp_p.observed:.3f}, "
              f"empirical p {hp_p.emp_p_two_sided:.3f} swap / {hp_p_r.emp_p_two_sided:.3f} random, but per-gene Wilcoxon "
              f"p={hp_p.obs_wilcoxon_p:.2f}); QC-rule M (n=9) and the continuous test (ρ={rho_s[0]:.2f}) are null; "
              f"on the unscaled statistic the random-direction null gives "
              f"{pick('hyd_pol', 'curated_M', 'random_direction', 'unscaled').emp_p_two_sided:.3f} (curated M) and "
              f"{pick('hyd_pol', 'da_lt_-0.05', 'random_direction', 'unscaled').emp_p_two_sided:.3f} (Δα<-0.05).")
    lines += ["## Sentence", "", sentence, "", caveat, ""]

    # ---- supplementary LaTeX table (scaled statistic; unscaled Wilcoxon p for reference)
    def fp(p):
        if p >= 0.001:
            return f"${p:.3f}$"
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    tex_rows = prim_rows + [("hyd_pol", "da_gt_0.05"), ("hyd_pol", "da_near"), ("hyd_pol", "qc_rule_M")]
    names = {"curated_M": "curated M", "da_lt_-0.05": r"$\Delta\alpha<-0.05$", "da_gt_0.05": r"$\Delta\alpha>0.05$",
             "da_near": r"$|\Delta\alpha|\le0.05$", "qc_rule_M": "QC-rule M"}
    labels = {k: lab for k, _, _, _, lab in CONTRASTS}
    body = []
    for i, (k, grp) in enumerate(tex_rows):
        s, rnd = pick(k, grp, "gene_swap"), pick(k, grp, "random_direction")
        u = pick(k, grp, "gene_swap", "unscaled")
        lab = labels[k] if grp == "all" else f"{labels[k]}, {names[grp]}"
        if i == len(prim_rows) - 1:  # the rule stands above the first class defined from the data
            body.append(r"  \midrule")
        body.append(f"  {lab} & {s.n_genes} & ${s.observed:+.3f}$ & ${s.null_median:+.3f}$ [${s.null_q025:+.3f}$, ${s.null_q975:+.3f}$] & "
                    f"{100 * s.frac_null_p05_same_sign:.0f}\\% & {fp(s.emp_p_two_sided)} & {fp(rnd.emp_p_two_sided)} & "
                    f"{fp(u.obs_wilcoxon_p)} \\\\")
    draws_tex = f"{args.draws:,}".replace(",", "{,}")
    tex = "\n".join([
        "% Residue specificity control. Generated by chimera/analysis/tasks/T8_1_residue/run.py",
        f"% ({args.draws} draws per null, seed {args.seed}); do not edit numbers by hand.",
        r"\begin{table}[!h]",
        r"  \centering",
        r"  \scriptsize",
        r"  \caption[Supplementary table]{Residue-level contrasts against attribution-replacement nulls. Per-residue "
        r"$|\text{attribution}|$ was recomputed with each protein's attribution replaced by another protein's (gene swap, a "
        f"derangement of the $70$ proteins) or by a random direction of the same norm in the $64$-PC attribution space, "
        f"${draws_tex}$ draws each, fixed seed. Statistic: per-protein difference of category medians divided by the "
        r"protein's median attribution, summarized as the median across proteins. Empirical $p$ is two-sided, "
        r"$(1+\#\{|\text{null}|\ge|\text{observed}|\})/(1+\text{draws})$. ``Swaps $p<0.05$'' is the share of swap draws "
        r"whose per-gene Wilcoxon test reaches $p<0.05$ with the observed sign. Wilcoxon $p$ is the per-gene "
        r"test on unscaled differences. The four rows below the rule within the body use classes defined from the data "
        r"(bins of $\Delta\alpha$ at $\pm0.05$, and the M flag of the ortholog quality rule, \cref{ssec:ortholog-qc}); "
        r"curated M is the hand-assigned class of candidates for a readout below the prior.}",
        r"  \label{tab:residue-specificity}",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{@{}lrrlrrrr@{}}",
        r"  \toprule",
        r"  Contrast & $n$ & Observed & Swap null median [95\%] & Swaps $p<0.05$ & $p$ swap & $p$ random & Wilcoxon $p$ \\",
        r"  \midrule",
        *body,
        r"  \bottomrule",
        r"  \end{tabular}",
        r"\end{table}", ""])
    final, tmp = out._tmp("residue-specificity.tex")
    tmp.write_text(tex)
    out._commit(tmp, final)
    final, tmp = out._tmp("summary.md")
    tmp.write_text("\n".join(lines))
    out._commit(tmp, final)
    print("\n".join(lines))
    out.finish(dict(draws=args.draws, seed=args.seed, runtime_s=round(time.time() - t0),
                    repro_min_pearson=float(rmin), sentence=sentence))


if __name__ == "__main__":
    main()
