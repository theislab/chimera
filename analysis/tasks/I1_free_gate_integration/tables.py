"""Tables: both gates side by side.

(a) Fig 3 rows: the ten frozen methods unchanged (s42, as Fig 3), Chimera canonical gate as in Fig 3 (s42 run), canonical gate over seeds,
    per-gene gate over seeds; mixing score, conservation score, composite on the frozen panel bounds.
(b) Harmonized cohorts: kNN mixing entropy, label transfer, kNN cell-type purity (H1_harmonise key) and the Z2_embedding within-label AUROC,
    both gates; scVI on the same object beside (H1_harmonise table for the cortex, lung, muscle and tumor cohorts; the scVI latents of
    Z2_embedding elsewhere).
    python -m analysis.tasks.I1_free_gate_integration.tables
Writes results/analysis/I1/table_fig3.csv, table_harmonised.csv and, in the same folder, the two tables as LaTeX
(integration_per_gene_gate.tex, integration_per_gene_gate_harmonised.tex). Nothing is written into the manuscript repository, whose
tables/ folder takes the two tables from these files by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.tasks.I1_free_gate_integration import rules as RU
from analysis.tasks.I1_free_gate_integration import score as SC

SE = SC.SE
TEX_A = RU.OUT / "integration_per_gene_gate.tex"
TEX_B = RU.OUT / "integration_per_gene_gate_harmonised.tex"
ORDER = ["PCA", "Harmony", "scVI-sp", "scVI-donor", "sysVI", "CAMEX", "SATURN-Leiden", "scANVI", "SATURN"]
SHORT = {"liv_hmonkey": "Liver H+Monkey", "liv_hmouse": "Liver H+Mouse", "liv_hzeb": "Liver H+Zeb", "liv_4sp": "Liver 4-sp",
         "test_hmac": "Testis H+Mac", "test_hmouse": "Testis H+Mouse", "test_3sp": "Testis 3-sp"}
SCORES = [("mixing", "Species-mixing score"), ("biology", "Cell-type conservation score"), ("total_score", "Composite")]


def fmt(v, d=2):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def cell(g: pd.Series, d=2) -> str:
    g = g.dropna()
    if not len(g):
        return "--"
    if len(g) == 1:
        return fmt(g.iloc[0], d)
    return f"{fmt(g.mean(), d)} ({fmt(g.min(), d)}--{fmt(g.max(), d)})"


def mark(call: str) -> str:
    return {"better": "$^{\\uparrow}$", "worse": "$^{\\downarrow}$"}.get(call, "")


def load():
    P = pd.read_csv(RU.OUT / "per_seed.csv")
    C = pd.read_csv(RU.OUT / "compare.csv")
    global KEPT
    KEPT = C[(C.analysis == "free_kept_json") & (C.line == "main")]
    return P[P.included & (P.line == "main")], C[(C.analysis == "primary") & (C.line == "main")]


KEPT = pd.DataFrame()
NAMES = {"h1|knn_mixing_entropy": "mixing", "h1|label_transfer_acc": "label transfer", "h1|knn_celltype_purity": "cell-type purity",
         "z2|within_auroc": "within-label AUROC", "fig3|mixing": "mixing score", "fig3|biology": "conservation score", "fig3|total_score": "composite"}


def kept_note(metrics) -> str:
    """Footnote: the variant with per-gene seeds restricted to the integration-filtered list of the per-gene training (mixing criterion)."""
    q = KEPT[KEPT.metric.isin(metrics)]
    if not len(q):
        return ""
    from analysis.tasks.H1_harmonise import rules as H1R
    H = H1R.cohorts()
    parts = []
    for c, g in q.groupby("cohort", sort=False):
        lab = SHORT.get(c, H[c]["label"]) if c in SHORT else H[c]["label"]
        vals = "; ".join(f"{NAMES[r.metric]} {fmt(r.mean_free)} against {fmt(r.mean_can)}" + (" (" + {"better": "above", "worse": "below"}[r.call] + " every canonical seed)" if r.call in ("better", "worse") else "")
                         for r in g.itertuples())
        parts.append(f"{lab}, per-gene seeds {g.seeds_free.iloc[0].replace(',', ', ')}: {vals}")
    return (" Selecting per-gene seeds also on species mixing (half the canonical median mixing entropy) would drop only "
            "poorly mixed seeds and gives: " + "; ".join(parts) + ".")


def table_a(P, C):
    T = pd.read_csv(SE.OUT / "master_table_ext.csv")
    cohorts = [c for c in RU.FIG3 if ((P.cohort == c) & (P.gate == "free") & P.metric.eq("fig3|mixing")).any()]
    rows, csv = [], []
    for key, title in SCORES:
        block = []
        for m in ORDER + ["Chimera"]:
            cells = []
            for c in cohorts:
                v = T[(T.dataset == RU.FIG3[c]) & (T.model == m)][key]
                cells.append(fmt(float(v.iloc[0])) if len(v) else "--")
                csv.append(dict(score=key, row=m if m != "Chimera" else "Chimera, canonical gate, Fig. 3 run (s42)", cohort=c, value=float(v.iloc[0]) if len(v) else np.nan))
            name = m if m != "Chimera" else "\\textit{Chimera}, canonical, Fig.~3 run"
            block.append(f"{name} & " + " & ".join(cells) + " \\\\")
        for gate, name in (("canonical", "\\textit{Chimera}, canonical, seeds"), ("free", "\\textit{Chimera}, per-gene, seeds")):
            cells = []
            for c in cohorts:
                g = P[(P.cohort == c) & (P.gate == gate) & (P.metric == f"fig3|{key}")].value
                k = C[(C.cohort == c) & (C.metric == f"fig3|{key}")]
                cells.append(cell(g) + (mark(k.call.iloc[0]) if gate == "free" and len(k) else ""))
                csv.append(dict(score=key, row=f"Chimera, {gate} gate, seeds", cohort=c, value=g.mean(), min=g.min(), max=g.max(), n=len(g),
                                delta=float(k.delta.iloc[0]) if len(k) and gate == "free" else np.nan,
                                ci_lo=float(k.ci_lo.iloc[0]) if len(k) and gate == "free" else np.nan,
                                ci_hi=float(k.ci_hi.iloc[0]) if len(k) and gate == "free" else np.nan,
                                call=k.call.iloc[0] if len(k) and gate == "free" else ""))
            block.append(f"{name} & " + " & ".join(cells) + " \\\\")
        rows.append((title, block))
    seeds = {c: (P[(P.cohort == c) & (P.gate == "canonical") & (P.metric == "fig3|mixing")].seed.nunique(),
                 P[(P.cohort == c) & (P.gate == "free") & (P.metric == "fig3|mixing")].seed.nunique()) for c in cohorts}
    pd.DataFrame(csv).to_csv(RU.OUT / "table_fig3.csv", index=False)
    return cohorts, rows, seeds


def write_a(cohorts, rows, seeds, caption_extra: str):
    n = len(cohorts)
    head = " & ".join(f"\\textbf{{{SHORT[c]}}}" for c in cohorts)
    sub = " & ".join(f"{seeds[c][0]} / {seeds[c][1]} seeds" for c in cohorts)
    L = ["% @@REV1:NEW@@ New table: integration scores of Chimera with the canonical gate and with the per-gene gate, beside the ten methods of Fig. 3. Why: owner asked whether the per-gene gate integrates better. [I1, Z3, Z4]",
         "% Generated by chimera/analysis/tasks/I1_free_gate_integration/tables.py from results/analysis/I1/; do not edit by hand.",
         "% @@REV1:NEW@@ New: table of both gates on the Fig. 3 cohorts. Why: both readouts side by side. [I1]",
         "\\begin{table*}[!t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{1.08}",
         "\\caption[Supplementary table]{" + caption_extra + "}", "\\label{tab:integration-per-gene-gate}",
         "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{3.3cm}} *{{{n}}}{{c}}@{{}}}}", "\\toprule",
         f"\\textbf{{Method}} & {head} \\\\", f" & {sub} \\\\"]
    for i, (title, block) in enumerate(rows):
        L.append("\\midrule")
        L.append(f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\textbf{{{title}}}}} \\\\")
        L += block
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    TEX_A.write_text("\n".join(L))


def scvi_refs() -> pd.DataFrame:
    from analysis.tasks.Z2_embedding import rules as Z2R
    from analysis.tasks.H1_harmonise import rules as H1R
    ct = pd.read_csv(RU.RES / "H1" / "cohort_table.csv").set_index("cohort")
    out = {}
    for c in ct.index:
        d = dict(scvi_mixing=ct.loc[c, "scvi_mixing"], scvi_label_transfer=ct.loc[c, "scvi_label_transfer"], scvi_within_auroc=np.nan, scvi_n=np.nan)
        fs = sorted((Z2R.OUT / "measure").glob(f"{c}__scvi_s*.parquet"))
        if fs:
            M = pd.concat([pd.read_parquet(f) for f in fs])
            M = M[~M.task.astype(str).str.endswith(("__odd", "__even"))]
            au = M[M.metric == "auroc_lr"].groupby(["latent", "task"]).value.mean().groupby("latent").mean()
            d["scvi_within_auroc"] = au.mean() if len(au) else np.nan
            if not np.isfinite(d["scvi_mixing"]):
                gm = M[(M.task == "global") & (M.metric == "knn_mixing_entropy")].value
                d["scvi_mixing"] = gm.mean() if len(gm) else np.nan
                if Z2R.LABEL_KEY.get(c) == H1R.cohorts()[c]["ct"]:
                    gl = M[(M.task == "global") & (M.metric == "label_transfer_acc")].value
                    d["scvi_label_transfer"] = gl.mean() if len(gl) else np.nan
            d["scvi_n"] = len(fs)
        out[c] = d
    return pd.DataFrame(out).T


def table_b(P, C):
    from analysis.tasks.H1_harmonise import rules as H1R
    H = H1R.cohorts()
    S = scvi_refs()
    cols = [("h1|knn_mixing_entropy", "mix"), ("h1|label_transfer_acc", "lt"), ("h1|knn_celltype_purity", "ctp"), ("z2|within_auroc", "wl")]
    both = [c for c in H if ((P.cohort == c) & (P.gate == "free")).any() and ((P.cohort == c) & (P.gate == "canonical")).any()]
    rows, tex = [], []
    tissue = None
    for c in both:
        r = dict(cohort=c, label=H[c]["label"], tissue=H[c]["tissue"],
                 n_can=P[(P.cohort == c) & (P.gate == "canonical")].seed.nunique(), n_free=P[(P.cohort == c) & (P.gate == "free")].seed.nunique())
        cells = []
        for m, k in cols:
            for gate in ("canonical", "free"):
                g = P[(P.cohort == c) & (P.gate == gate) & (P.metric == m)].value
                r[f"{k}_{gate}_mean"], r[f"{k}_{gate}_min"], r[f"{k}_{gate}_max"] = g.mean(), g.min(), g.max()
                kk = C[(C.cohort == c) & (C.metric == m)]
                call = kk.call.iloc[0] if len(kk) else ""
                if gate == "free":
                    r[f"{k}_delta"] = float(kk.delta.iloc[0]) if len(kk) else np.nan
                    r[f"{k}_ci_lo"], r[f"{k}_ci_hi"], r[f"{k}_call"] = (float(kk.ci_lo.iloc[0]), float(kk.ci_hi.iloc[0]), call) if len(kk) else (np.nan, np.nan, "")
                cells.append(cell(g) + (mark(call) if gate == "free" else ""))
        for k in ("scvi_mixing", "scvi_label_transfer", "scvi_within_auroc"):
            r[k] = float(S.loc[c, k]) if c in S.index else np.nan
        rows.append(r)
        sc = f"{fmt(r['scvi_mixing'])}; {fmt(r['scvi_label_transfer'])}; {fmt(r['scvi_within_auroc'])}"
        if H[c]["tissue"] != tissue:
            tissue = H[c]["tissue"]
            tex.append(("\\addlinespace\n" if tex else "") + f"\\multicolumn{{11}}{{@{{}}l}}{{\\textbf{{{tissue[0].upper() + tissue[1:]}}}}}\\\\")
        tex.append(f"{H[c]['label']} & {r['n_can']} / {r['n_free']} & " + " & ".join(cells) + f" & {sc} \\\\")
    pd.DataFrame(rows).to_csv(RU.OUT / "table_harmonised.csv", index=False)
    return rows, tex


def write_b(tex, caption: str):
    hdr = ("\\textbf{Cohort} & \\textbf{Seeds} & \\multicolumn{2}{c}{\\textbf{Mixing entropy}} & \\multicolumn{2}{c}{\\textbf{Label transfer}} & "
           "\\multicolumn{2}{c}{\\textbf{Cell-type purity}} & \\multicolumn{2}{c}{\\textbf{Within-label AUROC}} & \\textbf{scVI, same object} \\\\\n"
           " & can. / per-gene & canonical & per-gene & canonical & per-gene & canonical & per-gene & canonical & per-gene & mix.; l.t.; AUROC \\\\")
    L = ["% @@REV1:NEW@@ New table: cell embedding of every harmonised cohort under the canonical and the per-gene gate. Why: owner asked whether the per-gene gate integrates better. [I1, Z4]",
         "% Generated by chimera/analysis/tasks/I1_free_gate_integration/tables.py from results/analysis/I1/; do not edit by hand.",
         "% @@REV1:NEW@@ New: harmonised-cohort table, both gates. Why: both readouts side by side. [I1]",
         "\\begingroup", "\\setlength{\\tabcolsep}{2pt}", "\\setlength{\\LTcapwidth}{\\linewidth}", "\\renewcommand{\\arraystretch}{1.08}", "\\begin{landscape}", "\\scriptsize",
         "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{2.0cm} c *{8}{c} c@{}}",
         "\\caption[Supplementary table]{" + caption + "}", "\\label{tab:integration-per-gene-gate-harmonised}\\\\",
         "\\toprule", hdr, "\\midrule", "\\endfirsthead",
         "\\multicolumn{11}{@{}l}{\\textit{Supplementary \\cref{tab:integration-per-gene-gate-harmonised}, continued}}\\\\",
         "\\toprule", hdr, "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    L += tex
    L += ["\\end{longtable}", "\\end{landscape}", "\\endgroup", ""]
    TEX_B.write_text("\n".join(L))


def counts(C, metric: str, cohorts) -> tuple[int, int, int]:
    q = C[(C.metric == metric) & C.cohort.isin(cohorts)]
    return int((q.call == "better").sum()), int((q.call == "worse").sum()), len(q)


def caption_a(C, cohorts) -> str:
    b = [counts(C, f"fig3|{k}", cohorts) for k, _ in SCORES]
    n = len(cohorts)
    if all(x[0] == 0 and x[1] == 0 for x in b):
        head = (f"\\textbf{{The per-gene gate integrates like the canonical gate.}} On all {n} cohorts with both gates, the species-mixing, "
                "cell-type conservation and composite scores of \\textit{Chimera} with one free gate offset per gene stay within the range of its "
                "canonical seeds. ")
    else:
        parts = [f"{t.lower()} {x[0]} and {x[1]}" for (k, t), x in zip(SCORES, b)]
        head = ("\\textbf{The per-gene gate integrates about as the canonical gate does.} Of the cohort and score pairs, "
                f"{sum(x[2] - x[0] - x[1] for x in b)} of {sum(x[2] for x in b)} lie within the range of the canonical seeds; the per-gene gate "
                f"is better and worse than every canonical seed on, respectively: " + "; ".join(parts) + f" of {n} cohorts. ")
    return (head + "Scores as in \\cref{fig:integration}: scIB-style metrics, min--max scaled within each cohort, composite $0.4$ mixing $+$ $0.6$ "
            "conservation. Rows of the other methods and of the \\textit{Chimera} run shown in \\cref{fig:integration} are unchanged (seed 42); "
            "the seed rows give mean (range) over the healthy fits of each gate, with the number of seeds under each cohort (canonical / "
            "per-gene). Liver H+Mouse and testis H+Mouse use the same seed numbers and platform under both gates; elsewhere the per-gene fits "
            "are new seeds of the same recipe with only the gate switched (liver H+Zebrafish and testis H+Macaque have one kept canonical "
            "run; on liver H+Zebrafish one of two per-gene seeds passed health). Seed rows are scaled with the bounds of the ten-method panel, so no "
            "other value moves; a seed outside the panel's range scores below $0$ or above $1$ (poorly mixed liver H+Mouse seeds of both gates "
            "fall below the panel's narrow label-transfer range). $^{\\uparrow}$, $^{\\downarrow}$: every per-gene seed above, below every "
            "canonical seed. Per-gene seeds are selected by the health of the gate only, not by mixing." + kept_note([f"fig3|{k}" for k, _ in SCORES]))


def caption_b(C, rb) -> str:
    cs = [r["cohort"] for r in rb]
    bm, wm, nm = counts(C, "h1|knn_mixing_entropy", cs)
    bl, wl, nl = counts(C, "h1|label_transfer_acc", cs)
    bw, ww, nw = counts(C, "z2|within_auroc", cs)
    lt = C[(C.metric == "h1|label_transfer_acc") & C.cohort.isin(cs)]
    lower = f"Label transfer is lower under the per-gene gate in {int((lt.delta < 0).sum())} of {len(lt)} cohorts (median difference {fmt(lt.delta.median(), 3)}). "
    return ("\\textbf{Cell embedding of every harmonised cohort under both gates.} "
            f"Beyond the canonical seed range, the per-gene gate mixes species better in {bm} and worse in {wm} of {nm} cohorts, transfers "
            f"labels better in {bl} and worse in {wl}, and resolves states inside a label better in {bw} and worse in {ww} of {nw}. " + lower +
            "Mixing entropy, cross-species label transfer and kNN cell-type purity use the function, $k$ and cell-type key of "
            "Supplementary~\\cref{tab:harmonised-cohorts}. Within-label AUROC: mean cross-validated AUROC of known substates inside a cell "
            "type (hepatocyte zonation; kidney proximal tubule segment and state, thick ascending limb and endothelial state; beta-cell stress; "
            "marrow maturation and progression; testis stages), from the latent of the model. Values are mean (range) over seeds; canonical "
            "fits are the kept runs of Supplementary~\\cref{tab:harmonised-cohorts}, per-gene fits pass the same health rule for the gate "
            "(species mixing is not used to select them). scVI on the same object, unchanged: mixing; label transfer; within-label AUROC. "
            "$^{\\uparrow}$, $^{\\downarrow}$: every per-gene seed above, below every canonical seed."
            + kept_note(["h1|knn_mixing_entropy", "h1|label_transfer_acc", "h1|knn_celltype_purity", "z2|within_auroc"]))


def main():
    P, C = load()
    cohorts, rows, seeds = table_a(P, C)
    write_a(cohorts, rows, seeds, caption_a(C, cohorts))
    rb, tb = table_b(P, C)
    write_b(tb, caption_b(C, rb))
    print("fig3 cohorts:", cohorts, "harmonised cohorts:", len(rb), TEX_A, TEX_B)
    return cohorts, rows, seeds, rb, tb


if __name__ == "__main__":
    main()
