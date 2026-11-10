"""H1: write the cohort table and the two data-rule sheets of the Supplement as LaTeX, and the numbers that the figure
captions on the tissues state, from results/analysis/H1/.

Every number comes from a result table, so a re-run with new cohorts rewrites the rows. Two files are written under
``results/analysis/H1/`` (or beside ``--out``) and nothing into the manuscript repository:

  cohort_tables_generated.tex   the three tables of ``supplementary/cohort_tables.tex`` of the manuscript, line for line
                                (the manuscript file also carries comment lines and takes the tables from this file by hand)
  caption_numbers.tsv           figure label, quantity, value: the numbers and the named genes in the captions of the
                                figure on the eight tissues and of the per-tissue and recurrence figures of the Supplement;
                                a gene is listed only if it may be named (strict count support in its cohort)

    PYTHONPATH=<chimera repo> python -m analysis.tasks.H1_harmonise.tex [--out FILE] [--compare FILE]

``--compare FILE`` prints every line of the generated tables that differs from FILE (comment lines of FILE are ignored)
and exits 1 if there is one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.tasks.H1_harmonise import rules as RU
from figures import style

OUT = RU.OUT
GENERATED = OUT / "cohort_tables_generated.tex"
NUMBERS = "caption_numbers.tsv"  # written beside the tables
T_SRC = {  # tissue as in rules.py -> heading of its block of rows
    "liver": "Liver: \\textit{Guilliams et al.} 2022 (human, macaque, mouse) and \\textit{Wang et al.} 2023 (zebrafish), as processed by CAMEX",
    "testis": "Testis: \\textit{Murat et al.} 2023, as processed by CAMEX",
    "bone marrow": "Bone marrow: Tabula Sapiens, Tabula Microcebus and Tabula Muris Senis, bone marrow and spleen",
    "kidney": "Kidney: KPMP v2.0 human and mouse, multiome nuclei of healthy tissue",
    "islet": "Pancreatic islet: human 48-donor atlas and \\textit{Hrovatin et al.} 2023 mouse atlas, healthy donors and animals",
    "eye": "Eye: \\textit{van Zyl et al.} 2020, outflow pathway",
    "cortex": "Cortex: \\textit{Bakken et al.} 2021, primary motor cortex (M1); \\textit{Ma et al.} 2022, dorsolateral prefrontal cortex (DLPFC)",
    "lung": "Lung: \\textit{Travaglini et al.} 2020, Tabula Microcebus, Tabula Muris Senis",
    "skeletal muscle": "Skeletal muscle: \\textit{Kedlian et al.} 2024, cell fraction, human and mouse",
    "tumour myeloid": "Tumor myeloid: \\textit{Zilionis et al.} 2019, lung tumor-infiltrating immune cells, patients and KP mice",
}
ALL_TISSUES = RU.TISSUE_ORDER + ["skeletal muscle", "tumour myeloid"]
ROW_ORDER = {  # tissues whose rows are not in the order of cohort_table.csv
    "eye": ("vz_hmouse", "vz_hmmul", "vz_hmfas", "vz_hpig", "vz_primates", "vz_full"),
    "cortex": ("brain_m1_hmouse", "brain_m1", "brain_dlpfc", "brain_dlpfc_hchimp", "brain_dlpfc_hmac", "brain_dlpfc_hmarm", "brain_dlpfc_hmac_own"),
}
TISSUE_PREFIX = ("Liver ", "Testis ", "Eye ", "Cortex ", "Lung ", "Muscle ", "Tumor myeloid ")  # dropped from a row name: the block heading says it
# the recipe of the cortex, lung, muscle and tumor cohorts, in the words of the Methods
GRID_RECIPE = "row of a grid fixed before training, chosen by the health rule and count dependence"
COHORT_HEAD = (r"\textbf{Cohort} & \textbf{Species} & \textbf{Cells $\times$ genes} & \textbf{Types} & \textbf{Recipe} & \textbf{Runs} & \textbf{Count-supported} & "
               r"\textbf{$\Delta\alpha$ SD} & \textbf{Tails, anchored prior (up / down)} & \textbf{Tails, unclamped prior (up / down)} & \textbf{$\rho(\Delta\alpha, S_g)$ (partial)} & "
               r"\textbf{Mixing (random); label transfer} & \textbf{scVI, same object: mixing; label transfer} \\")
COHORT_COLUMNS = (r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{1.75cm} >{\raggedright\arraybackslash}p{2.1cm} >{\raggedright\arraybackslash}p{1.65cm} "
                  r">{\raggedright\arraybackslash}p{0.9cm} >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{0.75cm} >{\raggedright\arraybackslash}p{1.1cm} "
                  r">{\raggedright\arraybackslash}p{0.7cm} >{\raggedright\arraybackslash}p{1.25cm} >{\raggedright\arraybackslash}p{1.25cm} >{\raggedright\arraybackslash}p{1.15cm} "
                  r">{\raggedright\arraybackslash}p{1.9cm} >{\raggedright\arraybackslash}p{1.45cm}@{}}")


def num(x, nd=2):
    return f"{x:.{nd}f}"


def pct(x):
    return f"{100 * x:.0f}\\%"


def intc(x):
    return f"{int(x):,}".replace(",", "{,}")


def name(label: str) -> str:
    """A cohort label of cohort_table.csv as the paper writes it: species sets in full, American spelling."""
    return style.cohort_name(label).replace(", own table", ", own ortholog table")


def recipe(source: str) -> str:
    return GRID_RECIPE if source.startswith("blind grid of the plan") else source


def cohort_rows(t: pd.DataFrame) -> str:
    out = []
    for tissue in ALL_TISSUES:
        g = t[t.tissue == tissue]
        if not len(g):
            continue
        out.append(f"\\multicolumn{{13}}{{@{{}}l}}{{\\textbf{{{T_SRC[tissue]}}}}}\\\\")
        order = [c for c in ROW_ORDER[tissue] if c in set(g.cohort)] if tissue in ROW_ORDER else list(g.cohort)
        for c in order:
            r = g.set_index("cohort").loc[c]
            row_name = name(r.label)
            for prefix in TISSUE_PREFIX:
                row_name = row_name.replace(prefix, "")
            out.append(" & ".join([
                row_name.replace("M. ", "\\textit{M.}~") + ("$^{\\dagger}$" if c == RU.PRIMARY.get(tissue) else ""),
                r.species.replace("M. ", "\\textit{M.}~"), f"{intc(r.cells)} $\\times$ {intc(r.genes)}", f"{r.cell_types} ({r.cell_types_all_species})", recipe(r.recipe_source),
                f"{r.runs_kept} of {r.runs_trained}" + ("$^{\\ddagger}$" if r.role == "new" and r.seed_sat_max > 0.05 else ""), pct(r.count_support_share) + ("$^{\\star}$" if c == "tumour_zilionis" else ""),
                num(r.sd_delta_alpha_anchored, 3), f"{pct(r.tail_up_D)} / {pct(r.tail_dn_D)}", f"{pct(r.tail_up_Draw)} / {pct(r.tail_dn_Draw)}", f"{num(r.rho_DS_z1)} ({num(r.rho_DS_partial_z1)})",
                f"{num(r.mixing_entropy)} ({num(r.mixing_entropy_random)}); {num(r.label_transfer)}",
                f"{num(r.scvi_mixing)}; {num(r.scvi_label_transfer)}" if pd.notna(r.scvi_mixing) else "--"]) + " \\\\")
        out.append("\\addlinespace")
    return "\n".join(out[:-1])


def tables() -> str:
    """The cohort table and the two data-rule sheets, as in ``supplementary/cohort_tables.tex`` without its comment lines."""
    t = pd.read_csv(OUT / "cohort_table.csv")
    aud = pd.read_csv(OUT / "audit_cohorts.csv").set_index("cohort")
    tc = t.set_index("cohort")
    lines = []
    # ---- cohort table, one row per cohort
    flg = t[(t.role == "new") & (t.seed_sat_max > 0.05)]
    sat_note = "; leaving them out, $\\rho$ on the consensus moves " + ", ".join(
        f"from ${r.rho_DS_consensus:.2f}$ to ${r.rho_DS_no_saturated:.2f}$ ({name(r.label).replace('M. ', 'M.~')})" for r in flg.itertuples())
    lines.append("\\begingroup\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.08}\n\\begin{landscape}\n\\scriptsize\n" + COHORT_COLUMNS)
    lines.append(
        "\\caption[Supplementary table]{Every cohort, read the same way, one row per cohort. Types, cell-type classes in the cohort (shared by all species). "
        "Recipe, where the model settings come from (Supplementary~\\cref{tab:recipe-matrix}). Runs, runs kept of runs trained; failed runs are not analyzed, and two or more healthy runs "
        "are enough for a cohort beyond liver and testis; no model in this table uses the latent guard (\\cref{par:latent-guard}), and neither does any per-gene fit. "
        "Count-supported, genes with a count-evidence margin, mean over runs. $\\Delta\\alpha$ SD and tails, consensus over runs, centered within cohort, share of genes beyond $\\pm0.05$, "
        "against the prior the model anchors on and against the unclamped prior. "
        "$\\rho$, Spearman of $\\Delta\\alpha$ with $S_g$ on count-supported genes per run, mean over runs (partial on expression). Mixing, kNN species entropy on the latent ($k=15$; "
        "in parentheses its value under random mixing at the cohort's species proportions); label transfer, accuracy across species on the cell-type classes. "
        "$\\dagger$, the model used in \\cref{fig:one-reading}. $\\ddagger$, a kept run above $5\\%$ of genes at $\\alpha>0.95$ (the bar is $10\\%$): such genes are flagged in the per-gene tables"
        + sat_note + ". "
        "The DLPFC pair models are subsets of the four-primate object, except DLPFC Human+Macaque, own ortholog table, which is built on its own human--macaque ortholog table with the prior from human--macaque identity. "
        "scVI, scVI(species) on the same object with the same $k$ and labels, for cortex, lung, skeletal muscle and tumor myeloid cells; "
        "Chimera's label transfer sits below it in the DLPFC models and in lung, above it elsewhere. "
        f"$\\star$, count evidence for {pct(tc.loc['tumour_zilionis', 'count_support_share'])} of genes at inDrop depth, the lowest of all cohorts: per-gene claims on tumor stay few. "
        "Muscle and tumor are outside the eight-tissue reading: muscle enters it only as a sensitivity, tumor is not healthy tissue.}")
    lines.append("\\label{tab:harmonised-cohorts}\\\\\n\\toprule\n" + COHORT_HEAD + "\n\\midrule\n\\endfirsthead\n"
                 "\\multicolumn{13}{@{}l}{\\textit{Supplementary \\cref{tab:harmonised-cohorts}, continued}}\\\\\n\\toprule\n" + COHORT_HEAD + "\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot")
    lines.append(cohort_rows(t))
    lines.append("\\end{longtable}\n\\end{landscape}\n\\endgroup\n")
    # ---- data rules of liver, testis, bone marrow, kidney, islet and eye
    tcl = lambda c: f"{aud.loc[c, 'n_cell_types']} ({aud.loc[c, 'n_cell_types_all_species']})"  # noqa: E731
    rows = [
        ("Counts", *["Raw integer counts, no layers"] * 6),
        ("Batch", "Source dataset within species: one per species, macaque two", "Source dataset within species: one per species", "Source dataset within species: one per species", "Source dataset within species: one per species", "Source dataset within species: chemistry in human, study in mouse, two per species", "Source dataset within species: chemistry, at most two per species"),
        ("Cell QC", *["$\\geq200$ genes"] * 6),
        ("Mitochondrial cut", "Below $20\\%$", "Below $20\\%$", "None: mouse lemur and mouse matrices carry no mitochondrial genes", "None: no mitochondrial genes in the human matrix", "None: a cut would act unevenly by species and condition", "None: a prefix rule finds mitochondrial genes in two species only"),
        ("Mitochondrially encoded genes measured", "9", "11", "0", "0", "11", "0"),
        ("Types: classes (shared by all species)", tcl("liv_hmouse") + ", Human+Mouse", tcl("test_hmouse"), tcl("bone_marrow"), tcl("kidney"), tcl("islet"), tcl("vz_hmouse") + ", Human+Mouse"),
        ("Labels in training", "None", "Conditional MMD on classes (Human+Macaque on finer types)", "Conditional MMD on classes", "Conditional MMD on classes", "Conditional MMD on classes", "Conditional MMD on classes"),
        ("Thin types", "None removed", "None removed", "None needed; smallest $302$", "Class removed if below $50$ in a species", "Class removed if below $50$ in a species", "Type removed from a species where below $20$"),
        ("Cap per species and type", "None", "None", "$1{,}150$", "$2{,}000$", "$2{,}000$", "None"),
        ("One-to-one orthologs", "Ensembl, four-species liver table", "Ensembl, three-species testis table", "Ensembl 110, human--lemur and human--mouse", "Ensembl 110, human--mouse", "Ensembl, three-species testis table", "Ensembl, five-species table, genes measured in every species"),
        ("Count join", "Gene symbol, canonical pipeline", "Gene symbol, canonical pipeline", "Ensembl identifier", "Ensembl identifier", "Ensembl identifier", "Ensembl identifier first, symbol otherwise"),
        ("Pairs flagged by the ortholog audit, kept and marked: confidence; false zero or wrong row", *[f"{intc(tc.loc[c, 'n_o1_T2'])}; {intc(tc.loc[c, 'n_o1_map'])}" for c in [RU.PRIMARY[x] for x in RU.SIX]]),
        ("Protein prior $\\pi$: identity with human, mean over", "macaque, mouse, zebrafish, in every liver model", "macaque, mouse, in every testis model", "mouse lemur, mouse", "mouse", "macaque, mouse", "\\textit{M.}~fascicularis, \\textit{M.}~mulatta, mouse, pig, in every eye model"),
        ("Clamp of $\\pi$ in training", "None", "$0.85$", "$0.85$", "$0.85$", "$0.85$", "$0.85$"),
        ("Anchor", "Symmetric", "Symmetric", "Negative side at half weight", "Symmetric", "Symmetric", "Symmetric"),
    ]
    head = r"\textbf{Step} & \textbf{Liver} & \textbf{Testis} & \textbf{Bone marrow} & \textbf{Kidney} & \textbf{Islet} & \textbf{Eye} \\"
    lines.append("\\begingroup\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
                 "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{2.0cm} *{6}{>{\\raggedright\\arraybackslash}p{2.3cm}}@{}}")
    lines.append(
        "\\caption[Supplementary table]{One data-rule sheet for liver, testis, bone marrow, kidney, islet and eye: the same steps in the same words. "
        "Full reasons and counts are in Supplementary~\\cref{tab:data-preparation}; cortex, lung, skeletal muscle and tumor myeloid cells follow in Supplementary~\\cref{tab:harmonised-rules-c1}. "
        "Counts of flagged pairs are for the model used in \\cref{fig:one-reading}. Where rules differ, the reading is compared only after centering within cohort and matching within cohort "
        "on protein identity and expression. Matching on human--mouse identity instead of the cohort's own prior leaves the recurrent sets almost unchanged.}")
    lines.append("\\label{tab:harmonised-rules}\\\\\n\\toprule\n" + head + "\n\\midrule\n\\endfirsthead\n\\toprule\n" + head + "\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot")
    lines.extend(" & ".join(r) + " \\\\" for r in rows)
    lines.append("\\end{longtable}\n\\endgroup\n")
    # ---- data rules of cortex, lung, skeletal muscle and tumor myeloid cells, same steps and words
    c4 = ["brain_m1_hmouse", "lung_healthy", "muscle_ageing", "tumour_zilionis"]

    def quiet_or_zero(c):
        g = pd.read_parquet(OUT / "genes" / f"{c}.parquet")
        return intc(int((g.o1_zero_any.fillna(False).astype(bool) | g.quiet_in_species).sum()))

    rows2 = [
        ("Model", "M1 Human+Mouse (and M1 3 species, DLPFC 4 primates and pairs)", "Human, mouse lemur, mouse", "Human+Mouse", "Human+Mouse"),
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
        ("Genes quiet or zero in a species, flagged and not named", *[quiet_or_zero(c) for c in c4]),
        ("Protein prior $\\pi$: identity with human, mean over", "mouse (DLPFC: chimpanzee, macaque, marmoset)", "mouse lemur, mouse", "mouse", "mouse"),
        ("Clamp of $\\pi$ in training", *["$0.85$"] * 4),
        ("Anchor", *["Symmetric, scaled by depth"] * 4),
        ("Embedding against scVI(species), same object", *[f"label transfer {tc.loc[c, 'label_transfer']:.2f} against {tc.loc[c, 'scvi_label_transfer']:.2f}" for c in c4]),
    ]
    head = r"\textbf{Step} & \textbf{Cortex} & \textbf{Lung} & \textbf{Skeletal muscle} & \textbf{Tumor myeloid} \\"
    lines.append("\\begingroup\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
                 "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{2.6cm} *{4}{>{\\raggedright\\arraybackslash}p{3.3cm}}@{}}")
    lines.append("\\caption[Supplementary table]{The same data-rule sheet for cortex, lung, skeletal muscle and tumor myeloid cells, as Supplementary~\\cref{tab:harmonised-rules}. "
                 "These cohorts were built by one pipeline with the rules of the pre-specified analysis plan (its section 13; \\cref{sssec:rev-plan,sssec:rev-dataprep}), and their build records are in the deposit. "
                 "DLPFC models: the lower tail of $\\Delta\\alpha$ is marked as reference artifacts and not read as biology (the genes at its bottom are single-species annotation gaps checked on raw counts).}")
    lines.append("\\label{tab:harmonised-rules-c1}\\\\\n\\toprule\n" + head + "\n\\midrule\n\\endfirsthead\n\\toprule\n" + head + "\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot")
    lines.extend(" & ".join(r) + " \\\\" for r in rows2)
    lines.append("\\end{longtable}\n\\endgroup\n")
    return "\n".join(lines)


def caption_numbers() -> pd.DataFrame:
    """figure label, quantity, value: what the captions of the tissue figures state, each value read from a result table
    and formatted as the caption prints it. Raises if a gene that a caption names has no strict count support."""
    t = pd.read_csv(OUT / "cohort_table.csv").set_index("cohort")
    ax = pd.read_csv(OUT / "axis.csv")
    R = pd.read_csv(OUT / "recurrence.csv")
    sa = pd.read_csv(OUT / "set_axis.csv").set_index(["a", "b"]).spearman_over_sets
    L = pd.read_csv(OUT / "lineage_z.csv")
    ic = pd.read_csv(OUT / "identity_checks.csv").set_index("cohort")
    idx = R.pivot(index="key", columns="readout", values="recurrent")
    D = R[R.readout == "D"].set_index("key")
    prim = [RU.PRIMARY[x] for x in RU.TISSUE_ORDER]
    a8 = ax[ax.cohort.isin(prim) & (ax.universe == "all")]
    a8s = ax[ax.cohort.isin(prim) & (ax.universe == "count_support")]
    strong = idx.D & idx.D_tau & idx.D_block & idx.S & idx.S_block
    six = idx.six_D & idx.six_D_tau & idx.six_D_block & idx.six_S & idx.six_S_block
    direction = D.direction[strong[strong].index]
    model = {**RU.PRIMARY, **RU.SENS_EXTRA, "tumour myeloid": "tumour_zilionis"}  # the model a tissue's figure reads
    Lm = L[L.cohort.isin(model.values())]
    rows = []

    def add(figure, quantity, value):
        rows.append((figure, quantity, value))

    def lz(tissue, s, rd="D"):
        """z of a lineage set of the tissue in the tissue's model; rd "D" for Δα, "D_tau" for Δα matched also on specificity."""
        return f"{float(Lm[(Lm.cohort == model[tissue]) & (Lm.readout == rd) & (Lm.key == f'lineage|{tissue}|{s}')].z.iloc[0]):.1f}"

    def setz(key, tissue):
        return f"{float(R[(R.readout == 'D') & (R.key == key)][f'z_{tissue}'].iloc[0]):.1f}"

    def named(c, syms):
        g = pd.read_parquet(OUT / "genes" / f"{c}.parquet").set_index("symbol")
        bad = [s for s in syms if not bool(g.nameable.get(s, False))]
        if bad:
            raise AssertionError(f"{c}: named without strict support: {bad}")
        return ", ".join(syms)

    def lineages(fig, tissue, sets):
        for s in sets:
            add(fig, f"lineage set {s}: z on Δα", lz(tissue, s))
            add(fig, f"lineage set {s}: z on Δα, matched also on cell-type specificity", lz(tissue, s, "D_tau"))

    # ---- the figure on the eight tissues (model of each tissue as in rules.PRIMARY)
    f = "fig:one-reading"
    add(f, "Spearman of protein identity with dN/dS over the eight models: from", f"{a8.rho_pi_dNdS.max():.2f}")
    add(f, "Spearman of protein identity with dN/dS over the eight models: to", f"{a8.rho_pi_dNdS.min():.2f}")
    add(f, "Spearman of Δα with protein identity: largest absolute value", f"{a8.rho_D_pi.abs().max():.2f}")
    add(f, "Spearman of Δα with dN/dS: largest absolute value", f"{a8.rho_D_dNdS.abs().max():.2f}")
    add(f, "Spearman of S_g with protein identity on count-supported genes: largest absolute value", f"{a8s.rho_S_pi.abs().max():.2f}")
    add(f, "Spearman of S_g with dN/dS on count-supported genes: largest absolute value", f"{a8s.rho_S_dNdS.abs().max():.2f}")
    add(f, "Spearman of Δα against the unclamped prior with protein identity, tissues other than liver: from", f"{a8[a8.tissue != 'liver'].rho_D_raw_pi.max():.2f}")
    add(f, "Spearman of Δα against the unclamped prior with protein identity, tissues other than liver: to", f"{a8[a8.tissue != 'liver'].rho_D_raw_pi.min():.2f}")
    add(f, "smallest number of count-supported genes for a set to be scored in a tissue", RU.MIN_SET)
    add(f, "sets tested", len(D))
    add(f, f"sets that recur (same direction in at least {RU.R_MIN} tissues, permutation test, q < 0.05)", int(D.recurrent.sum()))
    add(f, "of these, sets that also recur on S_g, under the null that keeps protein neighborhoods, and matched on cell-type specificity", int(strong.sum()))
    add(f, "of these, above the prior", int((direction == "above").sum()))
    add(f, "of these, below the prior", int((direction == "below").sum()))
    add(f, "sets that pass the same filters on the first six tissues", int(six.sum()))
    for b, what in (("S", "S_g"), ("B1", "profile correlation"), ("pi", "protein identity"), ("dNdS", "dN/dS")):
        add(f, f"Spearman over all sets of the position on Δα with the position on {what}", f"{sa[('D', b)]:+.2f}" if b == "dNdS" else f"{sa[('D', b)]:.2f}")
    # ---- all recurrent sets
    f = "fig:supp-h1-recurrence-all"
    add(f, "sets that recur, of the sets tested", f"{int(D.recurrent.sum())} of {len(D)}")
    for col, what in (("six_D", "on the first six tissues"), ("D_r4", "in at least four of eight tissues"), ("D_with_muscle", "with muscle as a ninth tissue"),
                      ("pi", "on protein identity"), ("dNdS", "on dN/dS"), ("B1", "on profile correlation"),
                      ("D_tau", "on Δα, matched also on cell-type specificity"), ("S_tau", "on S_g, matched also on cell-type specificity"),
                      ("D_nosat", "on Δα, without genes at α > 0.95"), ("S_nosat", "on S_g, without genes at α > 0.95")):
        add(f, f"sets that recur {what}", int(idx[col].sum()))
    # ---- one figure per tissue
    f, c = "fig:supp-h1-bone-marrow", model["bone marrow"]
    add(f, "genes named", named(c, ["S100A8", "CD3E", "CD79A", "CD4", "CD3D", "ELANE", "CD19"]))
    lineages(f, "bone marrow", ["T and NK", "B and plasma", "myeloid"])  # the caption gives the range over the three sets
    add(f, "Hallmark heme metabolism: z on Δα", setz("hallmark|heme Metabolism", "bone marrow"))
    add(f, "share of genes of the negative tail backed by a negative margin", f"{t.loc[c, 'margin_support_down']:.2f}")
    add(f, "share of genes of the positive tail backed by a positive margin", f"{t.loc[c, 'margin_support_up']:.2f}")
    f = "fig:supp-h1-kidney"
    add(f, "genes named", named("kidney", ["PODXL", "CUBN", "UMOD", "LRP2", "TRPM6", "SLC12A3", "SLC12A1", "AQP2", "NPHS1"]))
    lineages(f, "kidney", ["proximal tubule", "loop and distal tubule", "collecting duct", "glomerulus"])
    add(f, "mitochondrial proteins: z on Δα", setz("class|mitochondrial", "kidney"))
    add(f, "genes in the negative tail against the anchored prior", int(t.loc["kidney", "n_tail_down_anchored"]))
    f = "fig:supp-h1-islet"
    add(f, "genes named", named("islet", ["PPY", "CHGA", "IAPP", "GCG", "SST", "NKX6-1", "MAFA", "ARX", "PECAM1", "PLVAP"]))
    lineages(f, "islet", ["endocrine", "stroma", "exocrine and duct"])
    add(f, "Hallmark pancreas beta cells: z on Δα", setz("hallmark|Pancreas Beta Cells", "islet"))
    f = "fig:supp-h1-eye"
    add(f, "genes named", named("vz_hmouse", ["MYOC", "MGP", "DCN", "DES", "MYH11", "TYR", "PMEL", "ANGPTL7"]))
    add(f, "heat-shock genes named", named("vz_hmouse", ["DNAJB1", "HSPH1", "BAG3"]))
    lineages(f, "eye", ["trabecular meshwork and aqueous outflow", "ciliary muscle and pigment"])
    add(f, "Spearman of Δα with S_g, eye 5 species", f"{t.loc['vz_full', 'rho_DS_z1']:.2f}")
    add(f, "Spearman of Δα with S_g, eye Human+Mouse", f"{t.loc['vz_hmouse', 'rho_DS_z1']:.2f}")
    f = "fig:supp-h1-cortex"
    add(f, "genes named", named("brain_m1_hmouse", ["GAD1", "GAD2", "VIP", "SATB2", "SLC17A7", "CUX2", "MBP", "PLP1", "MOG"]))
    lineages(f, "cortex", ["excitatory", "inhibitory", "glia"])
    if "n_reference_artefact" in ic.columns and pd.notna(ic.loc["brain_dlpfc", "n_reference_artefact"]):
        add(f, "genes of the lower tail of the DLPFC 4 primates model", int(ic.loc["brain_dlpfc", "n_reference_artefact"]))
        add(f, "share of them with a 10-fold level gap in one species", f"{100 * float(ic.loc['brain_dlpfc', 'frac_reference_artefact_10fold_gap']):.0f}%")
        add(f, "share of all genes with a 10-fold level gap in one species", f"{100 * float(ic.loc['brain_dlpfc', 'frac_all_10fold_gap']):.0f}%")
    add(f, "label transfer, cortex M1 Human+Mouse", f"{t.loc['brain_m1_hmouse', 'label_transfer']:.2f}")
    add(f, "label transfer of scVI on the same object", f"{t.loc['brain_m1_hmouse', 'scvi_label_transfer']:.2f}")
    f = "fig:supp-h1-lung"
    add(f, "genes named", named("lung_healthy", ["CA4", "EMCN", "PECAM1", "SFTPB", "SFTPC", "AGER", "HOPX", "CD79A", "COL1A1"]))
    lineages(f, "lung", ["stroma", "immune", "alveolar epithelium"])
    add(f, "mitochondrial proteins: z on Δα", setz("class|mitochondrial", "lung"))
    add(f, "label transfer, lung 3 species", f"{t.loc['lung_healthy', 'label_transfer']:.2f}")
    add(f, "label transfer of scVI on the same object", f"{t.loc['lung_healthy', 'scvi_label_transfer']:.2f}")
    f = "fig:supp-h1-muscle"
    add(f, "genes named", named("muscle_ageing", ["PECAM1", "CDH5", "DCN", "MYH11", "PAX7", "MYF5"]))
    add(f, "lineage set stroma and vessel: z on Δα", lz("skeletal muscle", "stroma and vessel"))
    f = "fig:supp-h1-tumour"
    add(f, "share of genes with count evidence", f"{100 * t.loc['tumour_zilionis', 'count_support_share']:.0f}%")
    add(f, "genes named", named("tumour_zilionis", ["S100A8", "CSF3R", "CD14", "C1QA", "APOE"]))
    add(f, "lineage set myeloid: z on Δα", lz("tumour myeloid", "myeloid"))
    return pd.DataFrame(rows, columns=["figure", "quantity", "value"])


def compare(txt: str, other: Path) -> int:
    """Lines of the generated tables that differ from ``other`` (its comment lines ignored); 0 if there is none."""
    mine = [line for line in txt.split("\n") if not line.startswith("%")]
    theirs = [line for line in other.read_text().split("\n") if not line.startswith("%")]
    n = 0
    for i in range(max(len(mine), len(theirs))):
        a, b = (mine[i] if i < len(mine) else None), (theirs[i] if i < len(theirs) else None)
        if a != b:
            n += 1
            print(f"line {i + 1}\n  generated: {a}\n  {other.name}: {b}")
    print(f"{other}: {n} lines differ" if n else f"{other}: the tables are the same")
    return int(n > 0)


def main(out: Path = GENERATED, other: Path | None = None) -> int:
    txt = tables()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    print(out, len(txt))
    numbers = caption_numbers()
    numbers.to_csv(out.with_name(NUMBERS), sep="\t", index=False)
    print(out.with_name(NUMBERS), len(numbers))
    return compare(txt, other) if other is not None else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=GENERATED, help="file of the tables (default: results/analysis/H1/cohort_tables_generated.tex)")
    ap.add_argument("--compare", type=Path, default=None, help="a LaTeX file to compare the generated tables with, comment lines ignored")
    a = ap.parse_args()
    raise SystemExit(main(a.out, a.compare))
