"""H1: write chimera-manuscript/supplementary/harmonised_cohorts.tex from results/revision/H1/. Every number comes from a result table, so a re-run with new cohorts rewrites the text.

    PYTHONPATH=<chimera repo> python revision/tasks/H1_harmonise/tex.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from revision.tasks.H1_harmonise import rules as RU

OUT = RU.OUT
MS = Path(os.environ.get("CHIMERA_MANUSCRIPT", RU.REPO.parent / "chimera-manuscript"))
TAG = "% @@REV1:NEW@@ {what}. Why: harmonise all cohorts. [H1]"
T_SRC = {
    "liver": "Liver: \\textit{Guilliams et al.} 2022 (human, macaque, mouse) and \\textit{Wang et al.} 2023 (zebrafish), as processed by CAMEX",
    "testis": "Testis: \\textit{Murat et al.} 2023, as processed by CAMEX",
    "bone marrow": "Bone marrow: Tabula Sapiens, Tabula Microcebus and Tabula Muris Senis, bone marrow and spleen",
    "kidney": "Kidney: KPMP v2.0 human and mouse, multiome nuclei of healthy tissue",
    "islet": "Pancreatic islet: human 48-donor atlas and \\textit{Hrovatin et al.} 2023 mouse atlas, healthy donors and animals",
    "eye": "Eye: \\textit{van Zyl et al.} 2020, outflow pathway",
}


def num(x, nd=2):
    return f"{x:.{nd}f}"


def pct(x):
    return f"{100 * x:.0f}\\%"


def intc(x):
    return f"{int(x):,}".replace(",", "{,}")


def cohort_rows(t: pd.DataFrame, audit: pd.DataFrame) -> str:
    out = []
    a = audit.set_index("cohort")
    for tissue in RU.TISSUE_ORDER:
        g = t[t.tissue == tissue]
        if not len(g):
            continue
        out.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\textbf{{{T_SRC[tissue]}}}}}\\\\")
        order = list(g.cohort)
        if tissue == "eye":
            order = [c for c in ("vz_hmouse", "vz_hmmul", "vz_hmfas", "vz_hpig", "vz_primates", "vz_full") if c in order]
        for c in order:
            r = g.set_index("cohort").loc[c]
            sp = r.species.replace("M. ", "\\textit{M.}~")
            out.append(" & ".join([
                r.label.replace("Liver ", "").replace("Testis ", "").replace("Eye ", "").replace("M. ", "\\textit{M.}~").replace("H+", "H+") + ("$^{\\dagger}$" if c == RU.PRIMARY[tissue] else ""),
                sp, f"{intc(r.cells)} $\\times$ {intc(r.genes)}", f"{r.cell_types} ({r.cell_types_all_species})", r.recipe_source, f"{r.runs_kept} of {r.runs_trained}", pct(r.count_support_share),
                num(r.sd_delta_alpha_anchored, 3), f"{pct(r.tail_up_D)} / {pct(r.tail_dn_D)}", f"{pct(r.tail_up_Draw)} / {pct(r.tail_dn_Draw)}", f"{num(r.rho_DS_z1)} ({num(r.rho_DS_partial_z1)})",
                f"{num(r.mixing_entropy)} ({num(r.mixing_entropy_random)}); {num(r.label_transfer)}"]) + " \\\\")
        out.append("\\addlinespace")
    return "\n".join(out[:-1])


def main() -> int:
    t = pd.read_csv(OUT / "cohort_table.csv")
    audit = pd.read_csv(OUT / "audit_cohorts.csv")
    ax = pd.read_csv(OUT / "axis.csv")
    R = pd.read_csv(OUT / "recurrence.csv")
    sa = pd.read_csv(OUT / "set_axis.csv").set_index(["a", "b"]).spearman_over_sets
    ts = pd.read_csv(OUT / "tissue_specific.csv")
    L = pd.read_csv(OUT / "lineage_z.csv")
    idx = R.pivot(index="key", columns="readout", values="recurrent")
    D = R[R.readout == "D"].set_index("key")
    prim = [RU.PRIMARY[x] for x in RU.TISSUE_ORDER]
    a6 = ax[ax.cohort.isin(prim) & (ax.universe == "all")]
    a6s = ax[ax.cohort.isin(prim) & (ax.universe == "count_support")]
    n_sets = len(D)
    n_rec = int(D.recurrent.sum())
    strong = idx.D & idx.D_tau & idx.D_block & idx.S & idx.S_block
    n_strong = int(strong.sum())
    n_ab, n_be = int((D.direction[strong[strong].index] == "above").sum()), int((D.direction[strong[strong].index] == "below").sum())
    Ltab = L[L.cohort.isin(prim)]

    def lz(tissue, s, rd):
        c = RU.PRIMARY[tissue]
        return float(Ltab[(Ltab.cohort == c) & (Ltab.readout == rd) & (Ltab.key == f"lineage|{tissue}|{s}")].z.iloc[0])

    def setz(key, tissue, rd="D"):
        return float(R[(R.readout == rd) & (R.key == key)][f"z_{tissue}"].iloc[0])

    def strict(c, syms):
        g = pd.read_parquet(OUT / "genes" / f"{c}.parquet").set_index("symbol")
        bad = [s for s in syms if not bool(g.strict_support.get(s, False))]
        if bad:
            raise AssertionError(f"{c}: named without strict support: {bad}")
        return ", ".join(f"\\textit{{{s}}}" for s in syms)

    tspec = ts[(ts.readout == "D") & ts.tissue_specific]
    lines = []
    lines.append(TAG.format(what="New file: harmonised table, data rules, main-figure candidate and one supplementary figure per new tissue"))
    # ---- main-figure candidate
    lines.append(TAG.format(what="Main-figure candidate, one reading across six tissues"))
    lines.append(r"""\begin{figure*}[!t]
  \begin{center}
    \resizebox*{!}{0.73\textheight}{%
    \begin{tabular}{@{}c@{}}
    \includegraphics[width=17cm]{figures/revision/fig_h1_axis.pdf}\\[-0.1em]
    \textbf{(a)}\\[0.2em]
    \begin{minipage}[c]{12.6cm}\centering
      \includegraphics[width=12.6cm]{figures/revision/fig_h1_recurrence.pdf}\\ \textbf{(b)}
    \end{minipage}\hspace{1mm}
    \begin{minipage}[c]{4.2cm}\centering
      \revfig{fig_h1_y1_slot}{4.2cm}\\ \textbf{(c)}
    \end{minipage}\\[0.2em]
    \includegraphics[width=17cm]{figures/revision/fig_h1_genes.pdf}\\[-0.1em]
    \textbf{(d)}
    \end{tabular}}""")
    lines.append(
        "    \\caption[Main figure]{\\textbf{One reading across six tissues: deployment runs on its own axis, and the same programmes recur.} "
        "Human+Mouse model of each tissue; bone marrow, its three-species model. $\\Delta\\alpha$ centred within tissue. "
        f"\\textbf{{(a)}} Same scales in every tissue. $\\Delta\\alpha$ distribution, and $\\Delta\\alpha$ against protein identity and dN/dS (blue, count evidence; grey, imputed from features; orange, median trend). "
        f"Protein identity follows dN/dS in every tissue (Spearman ${a6.rho_pi_dNdS.max():.2f}$ to ${a6.rho_pi_dNdS.min():.2f}$); $\\Delta\\alpha$ follows neither (absolute Spearman with identity at most ${a6.rho_D_pi.abs().max():.2f}$, with dN/dS at most ${a6.rho_D_dNdS.abs().max():.2f}$), and neither does the count evidence $S_g$ on its own genes (at most ${a6s.rho_S_pi.abs().max():.2f}$ and ${a6s.rho_S_dNdS.abs().max():.2f}$). "
        f"On recipes that clamp the prior at $0.85$, $\\alpha$ cannot pass $1$, which gives the flat ceiling near $+0.15$ at high identity, and $\\Delta\\alpha$ taken against the unclamped prior follows identity by arithmetic (Spearman ${a6[a6.tissue != 'liver'].rho_D_raw_pi.max():.2f}$ to ${a6[a6.tissue != 'liver'].rho_D_raw_pi.min():.2f}$); the panels use the prior each model anchors on. "
        f"\\textbf{{(b)}} Programmes and gene classes across tissues (exploratory). Cell, $z$ of the set's mean rank of $\\Delta\\alpha$ on count-supported genes, against genes permuted within identity and expression bins of that tissue; *, $q<0.05$ within tissue; dash, fewer than {RU.MIN_SET} genes. "
        f"{n_rec} of {n_sets} sets recur (same direction in at least {RU.R_MIN} tissues, permutation test on that count, $q<0.05$); shown are the {n_strong} that also recur on $S_g$ alone, under a null that keeps protein-neighbourhood structure, and with genes matched on cell-type specificity ({n_ab} above the prior, {n_be} below; all {n_rec} in \\cref{{fig:supp-h1-recurrence-all}}). "
        "Below the first rule, sets called in one tissue only; below the second, sets with large opposite signs. Beside, the same count on $S_g$ and on pseudobulk profile correlation (B1; black if recurrent), and the set's mean $z$ on protein identity and dN/dS, the sequence baselines. "
        f"Over all {n_sets} sets, position on $\\Delta\\alpha$ agrees with $S_g$ (Spearman ${sa[('D', 'S')]:.2f}$) and with B1 (${sa[('D', 'B1')]:.2f}$), not with identity (${sa[('D', 'pi')]:.2f}$) or dN/dS (${sa[('D', 'dNdS')]:+.2f}$), which agree with each other (${sa[('pi', 'dNdS')]:.2f}$). "
        "\\textbf{(c)} What is a property of the gene and what of the tissue. \\textbf{(d)} Genes with the largest $\\Delta\\alpha$ whose count evidence backs the sign in every seed and whose evidence is at least the tissue median: top, largest above the prior; bold, the tissue's lineage genes (markers fixed in advance); bottom, largest below the prior. $\\wedge$, $\\alpha>0.95$ in a seed, so rank within the group is capped. Mitochondrially encoded genes are left out: they are measured in three of the six tissues and track RNA retention. "
        "(Counts of cells, genes and runs: \\cref{tab:harmonised-cohorts}.)}")
    lines.append(r"""    \label{fig:one-reading}
  \end{center}
  \vspace{-1em}
\end{figure*}
""")
    # ---- table
    lines.append(TAG.format(what="Cross-cohort table, one row per cohort"))
    lines.append(r"""\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\begin{landscape}
\scriptsize
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{1.75cm} >{\raggedright\arraybackslash}p{2.1cm} >{\raggedright\arraybackslash}p{1.65cm} >{\raggedright\arraybackslash}p{0.9cm} >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{0.75cm} >{\raggedright\arraybackslash}p{1.1cm} >{\raggedright\arraybackslash}p{0.7cm} >{\raggedright\arraybackslash}p{1.25cm} >{\raggedright\arraybackslash}p{1.25cm} >{\raggedright\arraybackslash}p{1.15cm} >{\raggedright\arraybackslash}p{1.9cm}@{}}""")
    lines.append(
        "\\caption[Supplementary table]{\\textbf{Every cohort, read the same way.} Types, cell-type classes in the cohort (shared by all species). Recipe, where the model settings come from (\\cref{tab:recipe-matrix}). Runs, runs kept of runs trained; failed runs are not analysed, and two or more healthy runs are enough for a new cohort. "
        "Count-supported, genes with a count-evidence margin, mean over runs. $\\Delta\\alpha$ SD and tails, consensus over runs, centred within cohort, share of genes beyond $\\pm0.05$, against the prior the model anchors on and against the unclamped prior. "
        "$\\rho$, Spearman of $\\Delta\\alpha$ with $S_g$ on count-supported genes per run, mean over runs (partial on expression). Mixing, kNN species entropy on the latent ($k=15$; in brackets its value under random mixing at the cohort's species proportions); label transfer, accuracy across species on the cell-type classes. $\\dagger$, the model used for the six-tissue reading (\\cref{fig:one-reading}).}")
    lines.append(r"""\label{tab:harmonised-cohorts}\\
\toprule
\textbf{Cohort} & \textbf{Species} & \textbf{Cells $\times$ genes} & \textbf{Types} & \textbf{Recipe} & \textbf{Runs} & \textbf{Count-supported} & \textbf{$\Delta\alpha$ SD} & \textbf{Tails, anchored prior (up / down)} & \textbf{Tails, unclamped prior (up / down)} & \textbf{$\rho(\Delta\alpha, S_g)$ (partial)} & \textbf{Mixing (random); label transfer} \\
\midrule
\endfirsthead
\multicolumn{12}{@{}l}{\textit{Supplementary \cref{tab:harmonised-cohorts}, continued}}\\
\toprule
\textbf{Cohort} & \textbf{Species} & \textbf{Cells $\times$ genes} & \textbf{Types} & \textbf{Recipe} & \textbf{Runs} & \textbf{Count-supported} & \textbf{$\Delta\alpha$ SD} & \textbf{Tails, anchored prior (up / down)} & \textbf{Tails, unclamped prior (up / down)} & \textbf{$\rho(\Delta\alpha, S_g)$ (partial)} & \textbf{Mixing (random); label transfer} \\
\midrule
\endhead
\bottomrule
\endlastfoot""")
    lines.append(cohort_rows(t, audit))
    lines.append(r"""\end{longtable}
\end{landscape}
\endgroup
""")
    # ---- rules
    aud = audit.set_index("cohort")
    tcl = lambda c: f"{aud.loc[c, 'n_cell_types']} ({aud.loc[c, 'n_cell_types_all_species']})"  # noqa: E731
    rows = [
        ("Counts", "Raw integer counts, no layers", "Raw integer counts, no layers", "Raw integer counts, no layers", "Raw integer counts, no layers", "Raw integer counts, no layers", "Raw integer counts, no layers"),
        ("Batch", "Source dataset within species: one per species, macaque two", "Source dataset within species: one per species", "Source dataset within species: one per species", "Source dataset within species: one per species", "Source dataset within species: chemistry in human, study in mouse, two per species", "Source dataset within species: chemistry, at most two per species"),
        ("Cell QC", "$\\geq200$ genes", "$\\geq200$ genes", "$\\geq200$ genes", "$\\geq200$ genes", "$\\geq200$ genes", "$\\geq200$ genes"),
        ("Mitochondrial cut", "Below $20\\%$", "Below $20\\%$", "None: mouse lemur and mouse matrices carry no mitochondrial genes", "None: no mitochondrial genes in the human matrix", "None: a cut would act unevenly by species and condition", "None: a prefix rule finds mitochondrial genes in two species only"),
        ("Mitochondrially encoded genes measured", "9", "11", "0", "0", "11", "0"),
        ("Types: classes (shared by all species)", tcl("liv_hmouse") + ", H+Mouse", tcl("test_hmouse"), tcl("bone_marrow"), tcl("kidney"), tcl("islet"), tcl("vz_hmouse") + ", H+Mouse"),
        ("Labels in training", "None", "Conditional MMD on classes (H+Macaque on finer types)", "Conditional MMD on classes", "Conditional MMD on classes", "Conditional MMD on classes", "Conditional MMD on classes"),
        ("Thin types", "None removed", "None removed", "None needed; smallest $302$", "Class removed if below $50$ in a species", "Class removed if below $50$ in a species", "Type removed from a species where below $20$"),
        ("Cap per species and type", "None", "None", "$1{,}150$", "$2{,}000$", "$2{,}000$", "None"),
        ("One-to-one orthologs", "Ensembl, four-species liver table", "Ensembl, three-species testis table", "Ensembl 110, human--lemur and human--mouse", "Ensembl 110, human--mouse", "Ensembl, three-species testis table", "Ensembl, five-species table, genes measured in every species"),
        ("Protein prior $\\pi$: identity with human, mean over", "macaque, mouse, zebrafish, in every liver model", "macaque, mouse, in every testis model", "mouse lemur, mouse", "mouse", "macaque, mouse", "\\textit{M.}~fascicularis, \\textit{M.}~mulatta, mouse, pig, in every eye model"),
        ("Clamp of $\\pi$ in training", "None", "$0.85$", "$0.85$", "$0.85$", "$0.85$", "$0.85$"),
        ("Anchor", "Symmetric", "Symmetric", "Negative side at half weight", "Symmetric", "Symmetric", "Symmetric"),
    ]
    lines.append(TAG.format(what="Same data rules, in the same words, across cohorts"))
    lines.append(r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{2.0cm} *{6}{>{\raggedright\arraybackslash}p{2.3cm}}@{}}""")
    lines.append(
        "\\caption[Supplementary table]{\\textbf{One data-rule sheet for all cohorts.} Same steps, same words; full reasons and counts in \\cref{tab:data-preparation}. Where rules differ, the reading is compared only after centring within cohort and matching within cohort on protein identity and expression. "
        "Matching on human--mouse identity instead of the cohort's own prior leaves the recurrent sets almost unchanged.}")
    lines.append(r"""\label{tab:harmonised-rules}\\
\toprule
\textbf{Step} & \textbf{Liver} & \textbf{Testis} & \textbf{Bone marrow} & \textbf{Kidney} & \textbf{Islet} & \textbf{Eye} \\
\midrule
\endfirsthead
\toprule
\textbf{Step} & \textbf{Liver} & \textbf{Testis} & \textbf{Bone marrow} & \textbf{Kidney} & \textbf{Islet} & \textbf{Eye} \\
\midrule
\endhead
\bottomrule
\endlastfoot""")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append(r"""\end{longtable}
\endgroup
""")
    # ---- supplementary figures
    STD = ("\\textbf{(a--c)} $\\alpha$ against the prior, $\\Delta\\alpha$ distribution, $\\alpha$ and prior densities. \\textbf{(d)} The identity-versus-$\\Delta\\alpha$ plane with the tissue's lineage genes that have count support. \\textbf{(e)} Gene categories (effect size $d$; $\\star$, $q<0.05$). "
           "\\textbf{(f)} Hallmark enrichment ranked on $\\Delta\\alpha$. \\textbf{(g)} Cell-type contributions to $S_g$ of labelled genes (weights in the legend). \\textbf{(h)} Top count-supported genes, with the cell type that carries most of the margin ($\\wedge$, $\\alpha>0.95$ in a seed). \\textbf{(i)} GIN hidden-state archetypes. \\textbf{(j)} $\\Delta\\alpha$ against count evidence. ")
    bm = RU.PRIMARY["bone marrow"]
    capt = {
        "bone marrow": (f"Bone marrow reads like liver and testis and adds immune lineages: lymphoid and myeloid lineage genes sit above their protein prior, and heme metabolism stands out only here.",
                        f"Three-species model. Lineage genes above the prior include {strict(bm, ['S100A8', 'CD3E', 'CD79A', 'CD4', 'CD3D', 'ELANE', 'CD19'])} (lineage sets $z$ {lz('bone marrow', 'T and NK', 'D'):.1f} to {lz('bone marrow', 'B and plasma', 'D'):.1f}; matched also on cell-type specificity {lz('bone marrow', 'T and NK', 'D_tau'):.1f} to {lz('bone marrow', 'myeloid', 'D_tau'):.1f}, so most of the signal is the pattern gain of genes with a lineage pattern). "
                        f"Hallmark heme metabolism $z={setz('hallmark|heme Metabolism', 'bone marrow'):.1f}$, called in no other tissue. Most of the negative tail is not backed by a negative margin ({t.set_index('cohort').loc[bm, 'margin_support_down']:.2f} of tail genes, panel k; positive tail {t.set_index('cohort').loc[bm, 'margin_support_up']:.2f}); this model is the only one trained with a negative-side anchor at half weight, so its lower tail is not read here."),
        "kidney": ("Kidney adds the nephron: tubule and glomerular genes sit above their protein prior, and this holds at equal cell-type specificity for proximal tubule and glomerulus.",
                   f"Named genes include {strict('kidney', ['PODXL', 'CUBN', 'UMOD', 'LRP2', 'TRPM6', 'SLC12A3', 'SLC12A1', 'AQP2', 'NPHS1'])} (proximal tubule $z={lz('kidney', 'proximal tubule', 'D'):.1f}$, loop and distal {lz('kidney', 'loop and distal tubule', 'D'):.1f}, collecting duct {lz('kidney', 'collecting duct', 'D'):.1f}, glomerulus {lz('kidney', 'glomerulus', 'D'):.1f}; matched also on specificity {lz('kidney', 'proximal tubule', 'D_tau'):.1f}, {lz('kidney', 'loop and distal tubule', 'D_tau'):.1f}, {lz('kidney', 'collecting duct', 'D_tau'):.1f}, {lz('kidney', 'glomerulus', 'D_tau'):.1f}). "
                   f"Mitochondrial proteins, below the prior in five tissues, stay at it here ($z={setz('class|mitochondrial', 'kidney'):.1f}$). The negative tail against the anchored prior is almost empty ({int(t.set_index('cohort').loc['kidney', 'n_tail_down_anchored'])} genes)."),
        "islet": ("Islet adds the endocrine programme: endocrine lineage genes sit above their protein prior; the exocrine and duct genes do not separate.",
                  f"Named genes include {strict('islet', ['PPY', 'CHGA', 'IAPP', 'GCG', 'SST', 'NKX6-1', 'MAFA', 'ARX', 'PECAM1', 'PLVAP'])} (endocrine $z={lz('islet', 'endocrine', 'D'):.1f}$, stroma {lz('islet', 'stroma', 'D'):.1f}, exocrine and duct {lz('islet', 'exocrine and duct', 'D'):.1f}; matched also on specificity {lz('islet', 'endocrine', 'D_tau'):.1f}, {lz('islet', 'stroma', 'D_tau'):.1f}, {lz('islet', 'exocrine and duct', 'D_tau'):.1f}). Acinar cells are not in the model (too few in mouse). "
                  f"Hallmark Pancreas Beta Cells has enough count-supported genes only here ($z={setz('hallmark|Pancreas Beta Cells', 'islet'):.1f}$). Mitochondrially encoded genes fill the bottom of panel h; they track RNA retention and are not read."),
        "eye": ("The eye adds the outflow tract: trabecular meshwork, ciliary muscle and pigment-cell genes sit above their protein prior in the Human+Mouse model.",
                f"Named genes include {strict('vz_hmouse', ['MYOC', 'MGP', 'DCN', 'DES', 'MYH11', 'TYR', 'PMEL', 'ANGPTL7'])} (trabecular meshwork and aqueous outflow $z={lz('eye', 'trabecular meshwork and aqueous outflow', 'D'):.1f}$, ciliary muscle and pigment {lz('eye', 'ciliary muscle and pigment', 'D'):.1f}; matched also on specificity {lz('eye', 'trabecular meshwork and aqueous outflow', 'D_tau'):.1f}, {lz('eye', 'ciliary muscle and pigment', 'D_tau'):.1f}). Heat-shock genes at the bottom of panel h ({strict('vz_hmouse', ['DNAJB1', 'HSPH1', 'BAG3'])}) may reflect tissue handling. "
                "Panel k, the other eye models; the two pairs are read for genes because the larger models follow protein features more than counts (\\cref{tab:harmonised-cohorts}). Glaucoma genes are treated in their own section."),
    }
    names = {"bone marrow": "supp_h1_bone_marrow", "kidney": "supp_h1_kidney", "islet": "supp_h1_islet", "eye": "supp_h1_eye"}
    lab = {"bone marrow": "fig:supp-h1-bone-marrow", "kidney": "fig:supp-h1-kidney", "islet": "fig:supp-h1-islet", "eye": "fig:supp-h1-eye"}
    tail = {"bone marrow": "\\textbf{(k)} Fraction of tail genes backed by count evidence of the same sign. ", "kidney": "\\textbf{(k)} Fraction of tail genes backed by count evidence of the same sign. ",
            "islet": "\\textbf{(k)} Fraction of tail genes backed by count evidence of the same sign. ", "eye": "\\textbf{(k)} $\\Delta\\alpha$ in each eye model. "}
    for tissue in RU.NEW_TISSUES:
        title, body = capt[tissue]
        lines.append(TAG.format(what=f"Standard panels for {tissue}, same axes and palette as liver and testis"))
        lines.append("\\begin{figure*}[!p]\n  \\begin{center}\n"
                     f"    \\includegraphics[width=\\textwidth,height=0.86\\textheight,keepaspectratio]{{figures/revision/{names[tissue]}.pdf}}\n"
                     f"    \\caption[Supplementary figure]{{\\textbf{{{title}}} {body} Panels as for liver and testis in the main figures: {STD}{tail[tissue]}}}\n"
                     f"    \\label{{{lab[tissue]}}}\n  \\end{{center}}\n\\end{{figure*}}\n")
    lines.append(TAG.format(what="All recurrent sets and the lineage panels"))
    lines.append("\\begin{figure*}[!p]\n  \\begin{center}\n    \\includegraphics[width=0.95\\textwidth,height=0.86\\textheight,keepaspectratio]{figures/revision/supp_h1_recurrence_all.pdf}\n"
                 f"    \\caption[Supplementary figure]{{\\textbf{{All {n_rec} sets that recur across tissues, of {n_sets} tested.}} As \\cref{{fig:one-reading}}b, without the further filters. The sequence baselines recur about as often on their own lists (identity {int(idx.pi.sum())}, dN/dS {int(idx.dNdS.sum())}; B1 {int(idx.B1.sum())}), so a set's recurrence alone is a low bar for coherent gene sets; what the panel adds is which sets recur, and on which side of the prior. Matched also on cell-type specificity, {int(idx.D_tau.sum())} of the sets recur on $\\Delta\\alpha$ and {int(idx.S_tau.sum())} on $S_g$: much of what repeats is having a cell-state pattern.}}\n"
                 "    \\label{fig:supp-h1-recurrence-all}\n  \\end{center}\n\\end{figure*}\n")
    lines.append("\\begin{figure*}[!t]\n  \\begin{center}\n    \\includegraphics[width=\\textwidth]{figures/revision/supp_h1_lineage.pdf}\n"
                 "    \\caption[Supplementary figure]{\\textbf{Lineage genes reach count support mostly in their own tissue, and sit above matched genes there.} Lineage marker sets fixed in advance (rows) in each tissue (columns); $z$ as in \\cref{fig:one-reading}b, on $\\Delta\\alpha$ and on $S_g$. Boxes, the set's own tissue. Dash, fewer than five count-supported genes. Exploratory, not a test of conservation: the margin rewards having a cell-state pattern, and these sets are chosen for one.}\n"
                 "    \\label{fig:supp-h1-lineage}\n  \\end{center}\n\\end{figure*}\n")
    txt = "\n".join(lines)
    (MS / "supplementary" / "harmonised_cohorts.tex").write_text(txt)
    print(MS / "supplementary" / "harmonised_cohorts.tex", len(txt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
