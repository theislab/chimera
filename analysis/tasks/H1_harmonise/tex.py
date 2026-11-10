"""H1: write the cohort tables and the figure captions of the tissue figures as LaTeX, from results/analysis/H1/.

Every number comes from a result table, so a re-run with new cohorts rewrites the text. The output is
``results/analysis/H1/cohort_tables_generated.tex`` (``--out`` for another place); nothing is written into the
manuscript repository, whose ``supplementary/cohort_tables.tex`` takes the tables from this file by hand.

    PYTHONPATH=<chimera repo> python analysis/tasks/H1_harmonise/tex.py [--out FILE]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.tasks.H1_harmonise import rules as RU

OUT = RU.OUT
GENERATED = OUT / "cohort_tables_generated.tex"
TAG = "% @@REV1:NEW@@ {what}. Why: harmonise all cohorts. [H1]"
T_SRC = {
    "liver": "Liver: \\textit{Guilliams et al.} 2022 (human, macaque, mouse) and \\textit{Wang et al.} 2023 (zebrafish), as processed by CAMEX",
    "testis": "Testis: \\textit{Murat et al.} 2023, as processed by CAMEX",
    "bone marrow": "Bone marrow: Tabula Sapiens, Tabula Microcebus and Tabula Muris Senis, bone marrow and spleen",
    "kidney": "Kidney: KPMP v2.0 human and mouse, multiome nuclei of healthy tissue",
    "islet": "Pancreatic islet: human 48-donor atlas and \\textit{Hrovatin et al.} 2023 mouse atlas, healthy donors and animals",
    "eye": "Eye: \\textit{van Zyl et al.} 2020, outflow pathway",
    "cortex": "Cortex: \\textit{Bakken et al.} 2021, primary motor cortex (M1); \\textit{Ma et al.} 2022, dorsolateral prefrontal cortex (DLPFC)",
    "lung": "Lung: \\textit{Travaglini et al.} 2020, Tabula Microcebus, Tabula Muris Senis",
    "skeletal muscle": "Skeletal muscle: \\textit{Kedlian et al.} 2024, cell fraction, human and mouse",
    "tumour myeloid": "Tumour myeloid: \\textit{Zilionis et al.} 2019, lung tumour-infiltrating immune cells, patients and KP mice",
}
ALL_TISSUES = RU.TISSUE_ORDER + ["skeletal muscle", "tumour myeloid"]


def num(x, nd=2):
    return f"{x:.{nd}f}"


def pct(x):
    return f"{100 * x:.0f}\\%"


def intc(x):
    return f"{int(x):,}".replace(",", "{,}")


def cohort_rows(t: pd.DataFrame, audit: pd.DataFrame) -> str:
    out = []
    a = audit.set_index("cohort")
    for tissue in ALL_TISSUES:
        g = t[t.tissue == tissue]
        if not len(g):
            continue
        out.append(f"\\multicolumn{{13}}{{@{{}}l}}{{\\textbf{{{T_SRC[tissue]}}}}}\\\\")
        order = list(g.cohort)
        if tissue == "eye":
            order = [c for c in ("vz_hmouse", "vz_hmmul", "vz_hmfas", "vz_hpig", "vz_primates", "vz_full") if c in order]
        if tissue == "cortex":
            order = [c for c in ("brain_m1_hmouse", "brain_m1", "brain_dlpfc", "brain_dlpfc_hchimp", "brain_dlpfc_hmac", "brain_dlpfc_hmarm", "brain_dlpfc_hmac_own") if c in order]
        for c in order:
            r = g.set_index("cohort").loc[c]
            sp = r.species.replace("M. ", "\\textit{M.}~")
            out.append(" & ".join([
                r.label.replace("Liver ", "").replace("Testis ", "").replace("Eye ", "").replace("Cortex ", "").replace("Lung ", "").replace("Muscle ", "").replace("Tumour myeloid ", "").replace("M. ", "\\textit{M.}~") + ("$^{\\dagger}$" if c == RU.PRIMARY.get(tissue) else ""),
                sp, f"{intc(r.cells)} $\\times$ {intc(r.genes)}", f"{r.cell_types} ({r.cell_types_all_species})", r.recipe_source, f"{r.runs_kept} of {r.runs_trained}" + ("$^{\\ddagger}$" if r.role == "new" and r.seed_sat_max > 0.05 else ""), pct(r.count_support_share) + ("$^{\\star}$" if c == "tumour_zilionis" else ""),
                num(r.sd_delta_alpha_anchored, 3), f"{pct(r.tail_up_D)} / {pct(r.tail_dn_D)}", f"{pct(r.tail_up_Draw)} / {pct(r.tail_dn_Draw)}", f"{num(r.rho_DS_z1)} ({num(r.rho_DS_partial_z1)})",
                f"{num(r.mixing_entropy)} ({num(r.mixing_entropy_random)}); {num(r.label_transfer)}",
                f"{num(r.scvi_mixing)}; {num(r.scvi_label_transfer)}" if pd.notna(r.scvi_mixing) else "--"]) + " \\\\")
        out.append("\\addlinespace")
    return "\n".join(out[:-1])


def main(out: Path = GENERATED) -> int:
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
    PRIM_ALL = {**RU.PRIMARY, **RU.SENS_EXTRA, "tumour myeloid": "tumour_zilionis"}
    Ltab = L[L.cohort.isin(PRIM_ALL.values())]

    def lz(tissue, s, rd):
        c = PRIM_ALL[tissue]
        return float(Ltab[(Ltab.cohort == c) & (Ltab.readout == rd) & (Ltab.key == f"lineage|{tissue}|{s}")].z.iloc[0])

    def setz(key, tissue, rd="D"):
        return float(R[(R.readout == rd) & (R.key == key)][f"z_{tissue}"].iloc[0])

    def strict(c, syms):
        g = pd.read_parquet(OUT / "genes" / f"{c}.parquet").set_index("symbol")
        bad = [s for s in syms if not bool(g.nameable.get(s, False))]
        if bad:
            raise AssertionError(f"{c}: named without strict support: {bad}")
        return ", ".join(f"\\textit{{{s}}}" for s in syms)

    tspec = ts[(ts.readout == "D") & ts.tissue_specific]
    lines = []
    lines.append(TAG.format(what="New file: harmonised table, data rules, main-figure candidate and one supplementary figure per new tissue"))
    # ---- main-figure candidate
    lines.append(TAG.format(what="Main-figure candidate, one reading across eight tissues"))
    EDIT = "% @@REV1:EDIT@@ {what}. Why: clean, aligned figure. [H1, F1]"
    six = idx.six_D & idx.six_D_tau & idx.six_D_block & idx.six_S & idx.six_S_block
    tf = R[(R.readout == "D") & (R.key == "class|TF")].iloc[0]
    lines.append(EDIT.format(what="Figure composed from cropped panels on one grid, in three floats; cortex and lung joined"))
    lines.append(r"""\begin{figure*}[!t]
  \begin{center}
    \centerline{\includegraphics{figures/revision/fig_h1_composed.pdf}}""")
    lines.append(
        "    \\caption[Main figure]{\\textbf{One reading across eight tissues: deployment runs on its own axis, and each tissue adds its own genes.} "
        "Human+Mouse model of each tissue; bone marrow and lung, their human, mouse lemur, mouse models. $\\Delta\\alpha$ centred within tissue. "
        f"\\textbf{{(a)}} $\\Delta\\alpha$ distribution, and $\\Delta\\alpha$ against protein identity and dN/dS on one scale (blue, count evidence; grey, imputed from features; orange, median trend). "
        f"Identity follows dN/dS in every tissue (Spearman ${a6.rho_pi_dNdS.max():.2f}$ to ${a6.rho_pi_dNdS.min():.2f}$); $\\Delta\\alpha$ follows neither (at most ${a6.rho_D_pi.abs().max():.2f}$ and ${a6.rho_D_dNdS.abs().max():.2f}$), nor does $S_g$ on its own genes (${a6s.rho_S_pi.abs().max():.2f}$ and ${a6s.rho_S_dNdS.abs().max():.2f}$). "
        f"With the prior clamped at $0.85$, $\\alpha$ meets a ceiling near $+0.15$, and $\\Delta\\alpha$ against the unclamped prior follows identity by arithmetic (Spearman ${a6[a6.tissue != 'liver'].rho_D_raw_pi.max():.2f}$ to ${a6[a6.tissue != 'liver'].rho_D_raw_pi.min():.2f}$); panels use the prior each model anchors on. "
        "\\textbf{(b)} Genes with the largest $\\Delta\\alpha$ whose count evidence backs the sign in every seed, at least the tissue median, with counts in every species, not quiet in any species and no mapping flag: top, largest above the prior; middle in bold, the tissue's lineage genes (markers fixed in advance); bottom, largest below the prior. "
        "$\\wedge$, $\\alpha>0.95$ in a seed, so rank within the group is capped; \\S, pair flagged by the ortholog audit on confidence, kept. Mitochondrially encoded genes are left out: measured in a few tissues, and they track RNA retention. "
        "The lowest cortex genes carry a level gap between the human and mouse libraries (long genes high in human, short housekeeping genes high in mouse), likely capture rather than biology. "
        "Programmes: Figure~\\ref{fig:one-reading} (continued); counts of cells, genes and runs: \\cref{tab:harmonised-cohorts}.}")
    lines.append(r"""    \label{fig:one-reading}
  \end{center}
  \vspace{-1em}
\end{figure*}
""")
    lines.append(EDIT.format(what="Second float of the one-reading figure: programmes and gene classes across eight tissues"))
    lines.append(r"""\begin{figure*}[!t]
  \begin{center}
    \centerline{\includegraphics{figures/revision/fig_h1_cont_composed.pdf}}""")
    lines.append(
        "    \\caption*{\\textit{Figure~\\ref{fig:one-reading} (continued).} "
        f"\\textbf{{(c)}} Programmes and gene classes (exploratory). Cell, $z$ of the set's mean rank of $\\Delta\\alpha$ on count-supported genes against genes permuted within identity and expression bins of that tissue; *, $q<0.05$ within tissue; dash, fewer than {RU.MIN_SET} genes. "
        f"{n_rec} of {n_sets} sets recur (same direction in at least {RU.R_MIN} tissues, permutation test, $q<0.05$); shown, the {n_strong} that also recur on $S_g$, under a null keeping protein neighbourhoods, and matched on cell-type specificity ({n_ab} above the prior, {n_be} below; all in \\cref{{fig:supp-h1-recurrence-all}}). "
        f"The {int(six.sum())} sets that passed the same filters on the first six tissues all pass on eight. "
        "Under the first rule, sets called in one tissue; under the second, large opposite signs. At right, the recurrence $q$; the same count on $S_g$ and on pseudobulk profile correlation (B1; dark if recurrent); the set's mean $z$ on identity and dN/dS. "
        f"Over all {n_sets} sets, position on $\\Delta\\alpha$ agrees with $S_g$ (${sa[('D', 'S')]:.2f}$) and B1 (${sa[('D', 'B1')]:.2f}$), not with identity (${sa[('D', 'pi')]:.2f}$) or dN/dS (${sa[('D', 'dNdS')]:+.2f}$).}}")
    lines.append(r"""  \end{center}
  \vspace{-1em}
\end{figure*}
""")
    lines.append(EDIT.format(what="Third float of the one-reading figure: gene by tissue (six tissues)"))
    lines.append(r"""\begin{figure*}[!t]
  \begin{center}
    \centerline{\includegraphics{figures/revision/fig_h1_cont2_composed.pdf}}""")
    lines.append(
        "    \\caption*{\\textit{Figure~\\ref{fig:one-reading} (continued).} \\textbf{(d--i)} What belongs to the gene and what to the tissue, on the first six tissues (exploratory). \\textbf{(d)} Variance of within-tissue $z$ of $S_g$, $\\Delta\\alpha$ and B1, $1{,}211$ genes: shared by the gene, gene by tissue, technical noise. "
        "\\textbf{(e)} Partial rank correlation of each tissue's readout with each independent gold; outlined, the same tissue. \\textbf{(f)} Same tissue minus other tissues; bars, $95\\%$ block CI over protein modules. \\textbf{(g)} Within a gene, tissue difference in the readout against tissue difference in the gold; filled, matched on variability (post hoc). "
        "\\textbf{(h)} What is low in the low tissue of genes that flip, from the direct split. \\textbf{(i)} Worked genes, $z$ in count-supported tissues, extreme tissues labelled.}")
    lines.append(r"""  \end{center}
  \vspace{-1em}
\end{figure*}
""")
    # ---- table
    lines.append(TAG.format(what="Cross-cohort table, one row per cohort"))
    flg = t[(t.role == "new") & (t.seed_sat_max > 0.05)]
    sat_note = "; leaving them out, $\\rho$ on the consensus moves " + ", ".join(f"from ${r.rho_DS_consensus:.2f}$ to ${r.rho_DS_no_saturated:.2f}$ ({r.label.replace('M. ', 'M.~')})" for r in flg.itertuples())
    lines.append(r"""\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\begin{landscape}
\scriptsize
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{1.75cm} >{\raggedright\arraybackslash}p{2.1cm} >{\raggedright\arraybackslash}p{1.65cm} >{\raggedright\arraybackslash}p{0.9cm} >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{0.75cm} >{\raggedright\arraybackslash}p{1.1cm} >{\raggedright\arraybackslash}p{0.7cm} >{\raggedright\arraybackslash}p{1.25cm} >{\raggedright\arraybackslash}p{1.25cm} >{\raggedright\arraybackslash}p{1.15cm} >{\raggedright\arraybackslash}p{1.9cm} >{\raggedright\arraybackslash}p{1.45cm}@{}}""")
    lines.append(
        "\\caption[Supplementary table]{\\textbf{Every cohort, read the same way.} Types, cell-type classes in the cohort (shared by all species). Recipe, where the model settings come from (\\cref{tab:recipe-matrix}). Runs, runs kept of runs trained; failed runs are not analysed, and two or more healthy runs are enough for a new cohort. "
        "Count-supported, genes with a count-evidence margin, mean over runs. $\\Delta\\alpha$ SD and tails, consensus over runs, centred within cohort, share of genes beyond $\\pm0.05$, against the prior the model anchors on and against the unclamped prior. "
        "$\\rho$, Spearman of $\\Delta\\alpha$ with $S_g$ on count-supported genes per run, mean over runs (partial on expression). Mixing, kNN species entropy on the latent ($k=15$; in brackets its value under random mixing at the cohort's species proportions); label transfer, accuracy across species on the cell-type classes. $\\dagger$, the model used for the six-tissue reading (\\cref{fig:one-reading}). $\\ddagger$, a kept run above $5\\%$ of genes at $\\alpha>0.95$ (the bar is $10\\%$): such genes are flagged in the per-gene tables" + sat_note + ". "
        "scVI, scVI(species) on the same object with the same $k$ and labels, for the cohorts added last; Chimera's label transfer sits below it in the DLPFC models and in lung, above it elsewhere. "
        f"$\\star$, count support {pct(t.set_index('cohort').loc['tumour_zilionis', 'count_support_share'])} at inDrop depth, the lowest of all cohorts: per-gene claims on tumour stay few. "
        "Muscle and tumour are outside the eight-tissue reading: muscle enters it only as a sensitivity, tumour is not healthy tissue.}")
    lines.append(r"""\label{tab:harmonised-cohorts}\\
\toprule
\textbf{Cohort} & \textbf{Species} & \textbf{Cells $\times$ genes} & \textbf{Types} & \textbf{Recipe} & \textbf{Runs} & \textbf{Count-supported} & \textbf{$\Delta\alpha$ SD} & \textbf{Tails, anchored prior (up / down)} & \textbf{Tails, unclamped prior (up / down)} & \textbf{$\rho(\Delta\alpha, S_g)$ (partial)} & \textbf{Mixing (random); label transfer} & \textbf{scVI, same object: mixing; label transfer} \\
\midrule
\endfirsthead
\multicolumn{13}{@{}l}{\textit{Supplementary \cref{tab:harmonised-cohorts}, continued}}\\
\toprule
\textbf{Cohort} & \textbf{Species} & \textbf{Cells $\times$ genes} & \textbf{Types} & \textbf{Recipe} & \textbf{Runs} & \textbf{Count-supported} & \textbf{$\Delta\alpha$ SD} & \textbf{Tails, anchored prior (up / down)} & \textbf{Tails, unclamped prior (up / down)} & \textbf{$\rho(\Delta\alpha, S_g)$ (partial)} & \textbf{Mixing (random); label transfer} & \textbf{scVI, same object: mixing; label transfer} \\
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
    aud2 = t.set_index("cohort")
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
        ("Count join", "Gene symbol, first-submission pipeline", "Gene symbol, first-submission pipeline", "Ensembl identifier", "Ensembl identifier", "Ensembl identifier", "Ensembl identifier first, symbol otherwise"),
        ("Pairs flagged by the ortholog audit, kept and marked: confidence; false zero or wrong row", *[f"{intc(aud2.loc[c, 'n_o1_T2'])}; {intc(aud2.loc[c, 'n_o1_map'])}" for c in [RU.PRIMARY[x] for x in RU.SIX]]),
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
        "\\caption[Supplementary table]{\\textbf{One data-rule sheet for all cohorts.} Same steps, same words; full reasons and counts in \\cref{tab:data-preparation}. Counts of flagged pairs are for the model of the six-tissue reading. Where rules differ, the reading is compared only after centring within cohort and matching within cohort on protein identity and expression. "
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
    # the four tissues added with the C1 cohorts, same steps and words (data_preparation_filters.md section 6)
    c4 = ["brain_m1_hmouse", "lung_healthy", "muscle_ageing", "tumour_zilionis"]
    rows2 = [
        ("Model", "M1 H+Mouse (and M1 3 species, DLPFC 4 primates and pairs)", "Human, mouse lemur, mouse", "H+Mouse", "H+Mouse"),
        ("Counts", *["Raw integer counts over the whole matrix, no layers"] * 4),
        ("Batch", *["Source dataset within species: one per species (one source, one chemistry)"] * 4),
        ("Cell QC", "$\\geq200$ genes", "$\\geq200$ genes", "$\\geq200$ genes", "$\\geq200$ genes; one stricter rule on top of the authors' species QC"),
        ("Mitochondrial cut", *["None in any species: one threshold cannot act evenly"] * 4),
        ("Types: classes (shared by all species)", *[tcl(c) for c in c4]),
        ("Labels in training", *["Conditional MMD on classes"] * 4),
        ("Thin types", *["Class kept only with two species and at least $50$ cells in each species that has it"] * 4),
        ("Cap per species and type", "$1{,}200$ (M1 3 species $600$, DLPFC $350$)", "$750$", "None binding", "$6{,}350$"),
        ("One-to-one orthologs", *["Ensembl 110 Compara, high confidence, protein coding, no retrocopy target, no name elsewhere (the audit's rule, applied at build)"] * 4),
        ("Count join", "Ensembl identifier", "Ensembl identifier", "Ensembl identifier", "Symbol through the matrix's own Ensembl release"),
        ("Genes quiet or zero in a species, flagged and not named", *[intc(int((lambda g: (g.o1_zero_any.fillna(False).astype(bool) | g.quiet_in_species).sum())(pd.read_parquet(OUT / "genes" / f"{c}.parquet")))) for c in c4]),
        ("Protein prior $\\pi$: identity with human, mean over", "mouse (DLPFC: chimpanzee, macaque, marmoset)", "mouse lemur, mouse", "mouse", "mouse"),
        ("Clamp of $\\pi$ in training", *["$0.85$"] * 4),
        ("Anchor", *["Symmetric, scaled by depth"] * 4),
        ("Embedding against scVI(species), same object", *[f"label transfer {aud2.loc[c, 'label_transfer']:.2f} against {aud2.loc[c, 'scvi_label_transfer']:.2f}" for c in c4]),
    ]
    lines.append(TAG.format(what="Same data rules for the cohorts added last: cortex, lung, muscle, tumour"))
    lines.append(r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{2.6cm} *{4}{>{\raggedright\arraybackslash}p{3.3cm}}@{}}""")
    lines.append("\\caption[Supplementary table]{\\textbf{The same data-rule sheet for cortex, lung, muscle and tumour.} As \\cref{tab:harmonised-rules}; full reasons and counts in \\cref{tab:data-preparation}. "
                 "DLPFC models: the lower tail of $\\Delta\\alpha$ is marked as reference artefacts and not read as biology (the genes at its bottom are single-species annotation gaps checked on raw counts).}")
    lines.append(r"""\label{tab:harmonised-rules-c1}\\
\toprule
\textbf{Step} & \textbf{Cortex} & \textbf{Lung} & \textbf{Skeletal muscle} & \textbf{Tumour myeloid} \\
\midrule
\endfirsthead
\toprule
\textbf{Step} & \textbf{Cortex} & \textbf{Lung} & \textbf{Skeletal muscle} & \textbf{Tumour myeloid} \\
\midrule
\endhead
\bottomrule
\endlastfoot""")
    for r in rows2:
        lines.append(" & ".join(r) + " \\\\")
    lines.append(r"""\end{longtable}
\endgroup
""")
    # ---- supplementary figures
    STD = ("\\textbf{(a--c)} $\\alpha$ against the prior, $\\Delta\\alpha$ distribution, $\\alpha$ and prior densities. \\textbf{(d)} The identity-versus-$\\Delta\\alpha$ plane with the tissue's lineage genes that have count support. \\textbf{(e)} Gene categories (effect size $d$; $\\star$, $q<0.05$). "
           "\\textbf{(f)} Hallmark enrichment ranked on $\\Delta\\alpha$. \\textbf{(g)} Cell-type contributions to $S_g$ of labelled genes (weights in the legend). \\textbf{(h)} Top count-supported genes, with the cell type that carries most of the margin ($\\wedge$, $\\alpha>0.95$ in a seed). \\textbf{(i)} GIN hidden-state archetypes. \\textbf{(j)} $\\Delta\\alpha$ against count evidence. ")
    bm = RU.PRIMARY["bone marrow"]
    ic = pd.read_csv(OUT / "identity_checks.csv").set_index("cohort")
    art = {c: (int(ic.loc[c, "n_reference_artefact"]), float(ic.loc[c, "frac_reference_artefact_10fold_gap"]), float(ic.loc[c, "frac_all_10fold_gap"])) for c in ic.index if pd.notna(ic.loc[c].get("n_reference_artefact", np.nan))}
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
                f"Panel k, the other eye models; the pairs are read for genes because the larger models follow protein features more than counts ($\\rho(\\Delta\\alpha, S_g)$ {t.set_index('cohort').loc['vz_full', 'rho_DS_z1']:.2f} on five species against {t.set_index('cohort').loc['vz_hmouse', 'rho_DS_z1']:.2f} on Human+Mouse). Glaucoma genes are treated in their own section."),
        "cortex": ("Cortex adds the neuron and its glia: excitatory, inhibitory and glial lineage genes sit above their protein prior, also at equal cell-type specificity.",
                   f"M1 human and mouse model. Named genes include {strict('brain_m1_hmouse', ['GAD1', 'GAD2', 'VIP', 'SATB2', 'SLC17A7', 'CUX2', 'MBP', 'PLP1', 'MOG'])} (excitatory $z={lz('cortex', 'excitatory', 'D'):.1f}$, inhibitory {lz('cortex', 'inhibitory', 'D'):.1f}, glia {lz('cortex', 'glia', 'D'):.1f}; "
                   f"matched also on specificity {lz('cortex', 'excitatory', 'D_tau'):.1f}, {lz('cortex', 'inhibitory', 'D_tau'):.1f}, {lz('cortex', 'glia', 'D_tau'):.1f}). "
                   "The lowest genes of panel h carry a level gap between the two libraries (long genes high in human, short housekeeping genes high in mouse), likely capture, not biology. "
                   f"Panel k, every cortex model. In the DLPFC models the lower tail is marked as reference artefacts and never named: its bottom genes are single-species annotation gaps checked on raw counts, and {pct(art['brain_dlpfc'][1])} of the {intc(art['brain_dlpfc'][0])} lower-tail genes of the four-primate model carry a 10-fold single-species level gap, against {pct(art['brain_dlpfc'][2])} of all genes. "
                   f"Label transfer {aud2.loc['brain_m1_hmouse', 'label_transfer']:.2f} against {aud2.loc['brain_m1_hmouse', 'scvi_label_transfer']:.2f} for scVI on the same object; on DLPFC below scVI (\\cref{{tab:harmonised-cohorts}})."),
        "lung": ("Lung adds the alveolus and its vessels: stromal, immune and alveolar genes sit above their protein prior; stroma holds at equal specificity.",
                 f"Human, mouse lemur, mouse model. Named genes include {strict('lung_healthy', ['CA4', 'EMCN', 'PECAM1', 'SFTPB', 'SFTPC', 'AGER', 'HOPX', 'CD79A', 'COL1A1'])} (stroma $z={lz('lung', 'stroma', 'D'):.1f}$, immune {lz('lung', 'immune', 'D'):.1f}, alveolar epithelium {lz('lung', 'alveolar epithelium', 'D'):.1f}; "
                 f"matched also on specificity {lz('lung', 'stroma', 'D_tau'):.1f}, {lz('lung', 'immune', 'D_tau'):.1f}, {lz('lung', 'alveolar epithelium', 'D_tau'):.1f}; endothelium has fewer than five count-supported markers). "
                 f"Mitochondrial proteins sit lower here than in any tissue ($z={setz('class|mitochondrial', 'lung'):.1f}$). Label transfer {aud2.loc['lung_healthy', 'label_transfer']:.2f}, below the canonical three-species floor and below scVI on the same object ({aud2.loc['lung_healthy', 'scvi_label_transfer']:.2f})."),
        "skeletal muscle": ("Muscle reads the same way: vessel and stromal genes sit above their protein prior.",
                            f"Human and mouse, cell fraction. Named genes include {strict('muscle_ageing', ['PECAM1', 'CDH5', 'DCN', 'MYH11', 'PAX7', 'MYF5'])} (stroma and vessel $z={lz('skeletal muscle', 'stroma and vessel', 'D'):.1f}$; myofibre and satellite markers have fewer than five count-supported genes). "
                            "Muscle fits the rule of the eight-tissue reading and enters it as a sensitivity only."),
        "tumour myeloid": ("Tumour myeloid cells: the reading holds where counts allow, and counts allow little.",
                           f"Count support {pct(aud2.loc['tumour_zilionis', 'count_support_share'])} at inDrop depth, the lowest of all cohorts, so per-gene claims stay few. Named genes include {strict('tumour_zilionis', ['S100A8', 'CSF3R', 'CD14', 'C1QA', 'APOE'])} (myeloid lineage $z={lz('tumour myeloid', 'myeloid', 'D'):.1f}$). "
                           "\\textit{LYZ} is not named: the one-to-one pair misses \\textit{Lyz2}, the mouse lysozyme that is expressed. Tumour is not healthy tissue and stays outside the eight-tissue reading."),
    }
    names = {t_: f"supp_h1_{ {'bone marrow': 'bone_marrow', 'skeletal muscle': 'muscle', 'tumour myeloid': 'tumour'}.get(t_, t_)}" for t_ in RU.NEW_TISSUES}
    lab = {t_: names[t_].replace("supp_h1_", "fig:supp-h1-").replace("_", "-") for t_ in RU.NEW_TISSUES}
    tail = {"bone marrow": "\\textbf{(k)} Fraction of tail genes backed by count evidence of the same sign. ", "kidney": "\\textbf{(k)} Fraction of tail genes backed by count evidence of the same sign. ",
            "islet": "\\textbf{(k)} Fraction of tail genes backed by count evidence of the same sign. ", "eye": "\\textbf{(k)} $\\Delta\\alpha$ in each eye model. "}
    for i, tissue in enumerate(RU.NEW_TISSUES):
        title, body = capt[tissue]
        lines.append(TAG.format(what=f"Standard panels for {tissue}, same axes and palette as liver and testis"))
        lines.append(EDIT.format(what="Figure composed from cropped panels on one grid"))
        key = (f"Panels as for liver and testis in the main figures: {STD}{tail[tissue]}" if i == 0 else
               f"Panels as in \\cref{{{lab[RU.NEW_TISSUES[0]]}}}" + (f", except (k), $\\Delta\\alpha$ in each {tissue} model." if tissue in ("eye", "cortex") else "."))
        lines.append("\\begin{figure*}[!p]\n  \\begin{center}\n"
                     f"    \\centerline{{\\includegraphics{{figures/revision/{names[tissue]}_composed.pdf}}}}\n"
                     f"    \\caption[Supplementary figure]{{\\textbf{{{title}}} {body} {key}}}\n"
                     f"    \\label{{{lab[tissue]}}}\n  \\end{{center}}\n\\end{{figure*}}\n")
    lines.append(TAG.format(what="All recurrent sets and the lineage panels"))
    lines.append(EDIT.format(what="Figure composed at its size on the page"))
    lines.append("\\begin{figure*}[!p]\n  \\begin{center}\n    \\centerline{\\includegraphics{figures/revision/supp_h1_recurrence_all_composed.pdf}}\n"
                 f"    \\caption[Supplementary figure]{{\\textbf{{All {n_rec} sets that recur across tissues, of {n_sets} tested.}} As \\cref{{fig:one-reading}}c, eight tissues, without the further filters. On the first six tissues {int(idx.six_D.sum())} recur; at least four of eight tissues, {int(idx.D_r4.sum())}; with muscle as a ninth, {int(idx.D_with_muscle.sum())}. The sequence baselines recur about as often on their own lists (identity {int(idx.pi.sum())}, dN/dS {int(idx.dNdS.sum())}; B1 {int(idx.B1.sum())}), so a set's recurrence alone is a low bar for coherent gene sets; what the panel adds is which sets recur, and on which side of the prior. Matched also on cell-type specificity, {int(idx.D_tau.sum())} of the sets recur on $\\Delta\\alpha$ and {int(idx.S_tau.sum())} on $S_g$: much of what repeats is having a cell-state pattern. Without genes at $\\alpha>0.95$, {int(idx.D_nosat.sum())} recur on $\\Delta\\alpha$ and {int(idx.S_nosat.sum())} on $S_g$.}}\n"
                 "    \\label{fig:supp-h1-recurrence-all}\n  \\end{center}\n\\end{figure*}\n")
    lines.append(EDIT.format(what="Figure composed at its size on the page"))
    lines.append("\\begin{figure*}[!t]\n  \\begin{center}\n    \\centerline{\\includegraphics{figures/revision/supp_h1_lineage_composed.pdf}}\n"
                 "    \\caption[Supplementary figure]{\\textbf{Lineage genes reach count support mostly in their own tissue, and sit above matched genes there.} Lineage marker sets fixed in advance (rows) in each tissue (columns); $z$ as in \\cref{fig:one-reading}c, on $\\Delta\\alpha$ and on $S_g$. Boxes, the set's own tissue. Dash, fewer than five count-supported genes. Exploratory, not a test of conservation: the margin rewards having a cell-state pattern, and these sets are chosen for one.}\n"
                 "    \\label{fig:supp-h1-lineage}\n  \\end{center}\n\\end{figure*}\n")
    txt = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    print(out, len(txt))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=GENERATED, help="file to write (default: results/analysis/H1/cohort_tables_generated.tex)")
    raise SystemExit(main(ap.parse_args().out))
