"""Statistics that depend on gene identity in testis and on the match of genes between tissues.

The liver and the testis cohorts have different ortholog tables, so a column name (``ortho_N``)
stands for different genes in the two tissues. Every statistic here takes gene symbols from the
cohort's own table and matches genes between cohorts on the human Ensembl gene ID
(``annotate_genes_with_names``, ``shared_genes`` and ``ensembl_ids`` of
``notebooks/_paper_utils.py``). The formulas are those of the notebooks named in the ``source``
column. The values are computed here from the archived runs, or read from the cached tables of
the notebooks, so that every quantity has one row:

- agreement of program-level Δα between cohorts, within testis and between testis and liver;
- sign quadrants of the mean Δα of the liver cohorts against the mean Δα of the testis cohorts;
- standardized effects of the broad and the narrow gene categories on testis Human+Macaque;
- gene-set enrichment (GSEA) tables of the testis cohorts and of the liver-minus-testis
  contrast: number of significant gene sets, and the Hallmark sets named in the paper;
- row counts and named genes of three notebook summaries.

Δα is alpha minus the prior stored with the run (the clamped prior).

Output ``results/analysis/T2_2/values.csv``, one row per quantity: ``cohort``, ``quantity``,
``value`` (floats with four significant digits), ``source`` (the runs or the cached table the
value comes from), ``paper`` (LaTeX labels of the figures and sections that print the value,
with the panel letter where there is one; empty when the value is not printed) and ``note``.

Run as a module from the repository root; the command is in the README of this folder.
``--out FILE`` writes to another place.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon

REPO = Path(os.environ.get("CHIMERA_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO / "notebooks"))
import _gene_categories as gce  # noqa: E402
import _gene_categories_broad as gcb  # noqa: E402
import _paper_utils as pu  # noqa: E402

OUT = REPO / "results" / "analysis" / "T2_2"
RUNS = REPO / "training" / "runs"
DATA = REPO / "data" / "processed"

# cohort key -> (canonical run, processed object, tissue)
COHORTS = {
    "liv_hmonkey": ("iter90-150ep-s42", "camex_human_monkey.h5ad", "liver"),
    "liv_hmouse": ("xds-hmouse-iter90-150ep-s42", "camex_human_mouse.h5ad", "liver"),
    "liv_hzeb": ("xds-hzeb-iter85-s42", "camex_human_zebrafish.h5ad", "liver"),
    "liv_4sp": ("xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42", "camex_pancreas_unified.h5ad", "liver"),
    "test_hmac": ("testis-hmac-iter109-pclamp085-200ep-s42", "camex_testis_human_macaque.h5ad", "testis"),
    "test_hmouse": ("testis-hmouse-iter111-pclamp-coc-200ep-s42", "camex_testis_human_mouse.h5ad", "testis"),
    "test_3sp": ("testis-unified-iter115-pclamp-coc-100ep-s42", "camex_testis_unified.h5ad", "testis"),
}
LABEL = {"liv_hmonkey": "liver H+Monkey", "liv_hmouse": "liver H+Mouse", "liv_hzeb": "liver H+Zeb",
         "liv_4sp": "liver 4-sp", "test_hmac": "testis H+Macaque", "test_hmouse": "testis H+Mouse",
         "test_3sp": "testis 3-sp"}
LIVER = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp"]
TESTIS = ["test_hmac", "test_hmouse", "test_3sp"]
# Name of each testis cohort in the summary table of notebooks/08_ortholog_qc.ipynb.
QC_NAME = {"test_hmac": "test H+Mac", "test_hmouse": "test H+Mouse", "test_3sp": "test 3-sp"}
# Broad categories whose shift q-values are listed one by one.
Q_LISTED = ["Mitochondria & OXPHOS", "Translation & ribosome", "Metabolism", "Protein homeostasis", "Cell cycle"]
# Hallmark gene sets of the liver-minus-testis contrast that the paper names.
HALLMARK_NAMED = ["Bile Acid Metabolism", "Xenobiotic Metabolism", "Fatty Acid Metabolism", "Cholesterol Homeostasis",
                  "Epithelial Mesenchymal Transition"]
MT_DNA = "mitochondrial_dna_encoded"

# LaTeX labels of the places in the paper that print these values.
FIG_ROBUST = "fig:supp-alpha-robustness"
FIG_BROAD = "fig:supp-dalpha-broad-categories"
FIG_NARROW = "fig:supp-dalpha-testis-detailed"
FIG_HALLMARK = "fig:supp-dalpha-hallmark"
FIG_RESIDUE = "fig:supp-main-gin-protein"

_C: dict[str, dict] = {}


def cohort(key: str) -> dict:
    """Column names, tissue, run name and Δα of a cohort's canonical run."""
    if key not in _C:
        run, h5, tissue = COHORTS[key]
        fs = dict(np.load(RUNS / run / "final_state.npz", allow_pickle=False))
        a = ad.read_h5ad(DATA / h5, backed="r")
        vn = np.asarray(a.var_names)
        a.file.close()
        _C[key] = {"vn": vn, "tissue": tissue, "run": run, "delta": fs["alpha"] - fs["prior"]}
    return _C[key]


def state(key: str) -> str:
    return f"training/runs/{COHORTS[key][0]}/final_state.npz"


def symbols(key: str) -> pd.Series:
    """Human gene symbol of every column, from the cohort's own ortholog table."""
    c = cohort(key)
    return pu.annotate_genes_with_names(c["vn"], c["tissue"])


def join(kA: str, kB: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions of the genes two cohorts share, matched on the human Ensembl gene ID."""
    A, B = cohort(kA), cohort(kB)
    return pu.shared_genes(A["vn"], A["tissue"], B["vn"], B["tissue"])


def program_rho(kA: str, kB: str) -> dict:
    """Program means of Δα in two cohorts on their shared genes (notebook 03, ``category_means_aligned`` and
    ``_category_pair_panel``): number of genes and of programs, Spearman and Pearson over programs, Spearman over genes."""
    iA, iB = join(kA, kB)
    vA, vB = cohort(kA)["delta"][iA], cohort(kB)["delta"][iB]
    common = cohort(kA)["vn"][iA]
    sym = symbols(kA).values[iA]
    cats = gce.annotate_extended(common, pd.Series(sym, index=common))["category"].values
    df = pd.DataFrame({"cat": cats, "vA": vA, "vB": vB})
    sizes = df.groupby("cat").size()
    keep = [c for c in gce.CATEGORY_ORDER_EXTENDED if c != "other" and sizes.get(c, 0) >= 5]
    mA = np.array([df.loc[df.cat == c, "vA"].mean() for c in keep])
    mB = np.array([df.loc[df.cat == c, "vB"].mean() for c in keep])
    return {"n": len(iA), "k": len(keep), "rho": stats.spearmanr(mA, mB).correlation,
            "r": stats.pearsonr(mA, mB).statistic,
            "gene_rho": stats.spearmanr(vA, vB).correlation}


def category_effects(key: str, broad: bool) -> pd.DataFrame:
    """Effect sizes per gene category (notebooks 04 and 04_simpler, ``plot_effect_size_heatmap_for_dataset``):
    shift d, amplitude d_amp, and Benjamini-Hochberg q over both columns together. Categories with fewer than five genes are left out."""
    c = cohort(key)
    sym = symbols(key)
    ann = (gcb.annotate_broad if broad else gce.annotate_extended)(c["vn"], sym)
    order = gcb.CATEGORY_ORDER_BROAD if broad else gce.CATEGORY_ORDER_EXTENDED
    cat = ann["category"].values
    v = c["delta"].astype(float)
    a = np.abs(v)
    sd, sd_abs, mabs = np.std(v, ddof=1), np.std(a, ddof=1), a.mean()
    rows = []
    for k in order:
        if k == "other":
            continue
        m = cat == k
        if m.sum() < 5:
            continue
        d = v[m].mean() / sd
        d_amp = (a[m].mean() - mabs) / sd_abs * (1 if d >= 0 else -1)
        p_d = wilcoxon(v[m], alternative="two-sided", zero_method="wilcox").pvalue
        p_a = mannwhitneyu(a[m], a[~m], alternative="greater").pvalue
        rows.append({"category": k, "n": int(m.sum()), "d": d, "d_amp": d_amp, "p_d": p_d, "p_amp": p_a})
    df = pd.DataFrame(rows)
    q = pu.bh_fdr(np.concatenate([df.p_d.values, df.p_amp.values]))
    df["q_d"], df["q_amp"] = q[: len(df)], q[len(df):]
    return df.set_index("category")


def quadrants() -> dict:
    """Mean Δα per tissue over its cohorts, tissues matched on the human Ensembl gene ID, sign quadrants (notebook 08_ortholog_qc)."""
    def pool(keys):
        frames = []
        for k in keys:
            c = cohort(k)
            frames.append(pd.DataFrame({"g": pu.ensembl_ids(c["vn"], c["tissue"]).values, "d": c["delta"]}))
        return pd.concat(frames).dropna(subset=["g"]).groupby("g")["d"].mean()
    p, t = pool(LIVER), pool(TESTIS)
    xt = pd.concat([p.rename("p"), t.rename("t")], axis=1, join="inner")
    out = {"n": len(xt),
           "neg_neg": int(((xt.p < 0) & (xt.t < 0)).sum()), "pos_pos": int(((xt.p > 0) & (xt.t > 0)).sum()),
           "neg_pos": int(((xt.p < 0) & (xt.t > 0)).sum()), "pos_neg": int(((xt.p > 0) & (xt.t < 0)).sum())}
    out["opposite"] = out["neg_pos"] + out["pos_neg"]
    out["opposite_share"] = out["opposite"] / out["n"]
    out["gene_rho"] = stats.spearmanr(xt.p, xt.t).correlation
    return out


def table(rel: str) -> pd.DataFrame:
    """A cached table of a run or a notebook, by its path in the repository."""
    return pd.read_csv(REPO / rel)


def n_significant(d: pd.DataFrame) -> int:
    return int((pd.to_numeric(d["FDR q-val"], errors="coerce") < 0.05).sum())


ROWS: list[dict] = []


def row(cohort_label, quantity, value, source, paper="", note=""):
    ROWS.append({"cohort": cohort_label, "quantity": quantity, "value": value, "source": source, "paper": paper, "note": note})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT / "values.csv", help="file to write (default: results/analysis/T2_2/values.csv)")
    out = ap.parse_args().out

    # ---- program-level agreement within testis (panels g to i of the robustness figure)
    for (kA, kB), panel in {("test_3sp", "test_hmac"): "g", ("test_3sp", "test_hmouse"): "h", ("test_hmac", "test_hmouse"): "i"}.items():
        v = program_rho(kA, kB)
        lab = f"{LABEL[kA]} vs {LABEL[kB]}"
        src = f"{state(kA)} ; {state(kB)} (program means as in notebooks/03_alpha_master.ipynb)"
        fig = f"{FIG_ROBUST} {panel}"
        row(lab, "n shared genes", v["n"], src, fig + (" ; ssec:er-model" if panel == "i" else ""))
        row(lab, "k programs", v["k"], src, f"{FIG_ROBUST} ; ssec:er-model")
        row(lab, "program Spearman rho", v["rho"], src, fig + (" ; ssec:er-model" if panel == "h" else ""))
        row(lab, "program Pearson r", v["r"], src, fig)

    # ---- program-level agreement between testis and liver (panels j and k)
    for (kA, kB), panel in {("liv_hmonkey", "test_hmac"): "j", ("liv_hmouse", "test_hmouse"): "k"}.items():
        v = program_rho(kA, kB)
        lab = f"{LABEL[kB]} vs {LABEL[kA]}"
        src = f"{state(kB)} ; {state(kA)} (program means as in notebooks/03_alpha_master.ipynb)"
        fig = f"{FIG_ROBUST} {panel}"
        row(lab, "n shared genes", v["n"], src, f"{fig} ; ssec:er-model")
        row(lab, "k programs", v["k"], src, FIG_ROBUST)
        row(lab, "program Spearman rho", v["rho"], src, f"{fig} ; ssec:er-model")
        row(lab, "program Pearson r", v["r"], src, fig)
        row(lab, "gene-level Spearman rho (clamped prior)", v["gene_rho"], src)

    # ---- four-species liver against three-species testis
    v = program_rho("liv_4sp", "test_3sp")
    src = f"{state('liv_4sp')} ; {state('test_3sp')} (program means as in notebooks/03_alpha_master.ipynb)"
    row("liver 4-sp vs testis 3-sp", "n shared genes", v["n"], src)
    row("liver 4-sp vs testis 3-sp", "gene-level Spearman rho (clamped prior)", v["gene_rho"], src)
    row("liver 4-sp vs testis 3-sp", "program Spearman rho", v["rho"], src)

    # ---- sign quadrants of the tissue means
    q = quadrants()
    lab = "mean of 4 liver vs mean of 3 testis"
    src = "final_state.npz of the four liver and three testis canonical runs (sign quadrants as in notebooks/08_ortholog_qc.ipynb)"
    panel = f"{FIG_RESIDUE} e"
    row(lab, "sign quadrants: n shared genes", q["n"], src, "ssec:er-residue")
    row(lab, "sign quadrants: liver negative and testis negative", q["neg_neg"], src, panel)
    row(lab, "sign quadrants: liver positive and testis positive", q["pos_pos"], src, panel)
    row(lab, "sign quadrants: liver negative and testis positive", q["neg_pos"], src, panel)
    row(lab, "sign quadrants: liver positive and testis negative", q["pos_neg"], src, panel)
    row(lab, "sign quadrants: opposite sign in liver and testis", q["opposite"], src, "ssec:er-residue")
    row(lab, "share of genes changing sign class", q["opposite_share"], src, f"{panel} ; ssec:er-residue")
    row(lab, "gene-level Spearman rho of tissue means", q["gene_rho"], src)

    # ---- broad gene categories, testis Human+Macaque
    b = category_effects("test_hmac", True)
    liv_b = category_effects("liv_4sp", True)
    src = f"{state('test_hmac')} (effect sizes as in notebooks/04_delta_alpha_biology_simpler.ipynb)"
    in_text = {"Mitochondria & OXPHOS", "Metabolism", "Cell cycle", "Translation & ribosome"}
    for cat in b.index:
        row("testis H+Macaque", f"broad d (shift): {cat}", b.d[cat], src, f"{FIG_BROAD} d" + (" ; ssec:er-axis" if cat in in_text else ""))
        row("testis H+Macaque", f"broad n genes: {cat}", b.n[cat], src)
    row("testis H+Macaque", "broad d_amp: Mitochondria & OXPHOS", b.d_amp["Mitochondria & OXPHOS"], src)
    neg = [c for c in liv_b.index if liv_b.d[c] < 0]
    row("liver 4-sp vs testis H+Macaque", "liver-negative umbrellas positive on testis",
        f"{sum(b.d.get(c, -1) > 0 for c in neg)}/{len(neg)}", f"{state('liv_4sp')} ; {src}", "ssec:er-axis")
    row("testis H+Macaque", "max shift q over the 5 named umbrellas", b.q_d.reindex(Q_LISTED).max(), src, note="the five: " + ", ".join(Q_LISTED))
    for cat in Q_LISTED:
        row("testis H+Macaque", f"broad shift q: {cat}", b.q_d[cat], src,
            f"{FIG_BROAD} d ; ssec:er-axis" if cat == "Translation & ribosome" else "",
            "printed as not significant" if cat == "Translation & ribosome" else "")

    # ---- narrow gene categories, testis Human+Macaque
    n = category_effects("test_hmac", False)
    src = f"{state('test_hmac')} (effect sizes as in notebooks/04_delta_alpha_biology.ipynb)"
    for cat in n.index:
        row("testis H+Macaque", f"narrow d (shift): {cat}", n.d[cat], src, f"{FIG_NARROW} ; ssec:er-axis" if cat == MT_DNA else "")
    pooled = "Benjamini-Hochberg q over the shift and the amplitude column of the panel together, as in the notebook"
    row("testis H+Macaque", f"narrow n: {MT_DNA}", n.n[MT_DNA], src, f"{FIG_NARROW} ; ssec:er-axis")
    row("testis H+Macaque", f"narrow q_d: {MT_DNA}", n.q_d[MT_DNA], src, f"{FIG_NARROW} ; ssec:er-axis", pooled)
    row("testis H+Macaque", f"narrow q_amp: {MT_DNA}", n.q_amp[MT_DNA], src, "ssec:er-axis", pooled)

    # ---- GSEA tables of the testis cohorts (cached by notebooks/06_pathway_analysis.ipynb)
    for k in TESTIS:
        run = COHORTS[k][0]
        for lib, metric in (("MSigDB_Hallmark_2020", "delta"), ("MSigDB_Hallmark_2020", "prior"), ("Reactome_Pathways_2024", "delta")):
            rel = f"training/runs/{run}/cache/gsea__{lib}__{metric}.csv"
            row(LABEL[k], f"GSEA {lib} {metric}: pathways at FDR<0.05", n_significant(table(rel)), rel)
        rel = f"training/runs/{run}/tables/gsea_chimera_unique.csv"
        row(LABEL[k], "rows in gsea_chimera_unique", len(table(rel)), rel)

    # ---- GSEA of the liver-minus-testis contrast (four-species liver, three-species testis)
    run4 = COHORTS["liv_4sp"][0]
    lab = "liver 4-sp minus testis 3-sp"
    for lib in ("MSigDB_Hallmark_2020", "Reactome_Pathways_2024"):
        rel = f"training/runs/{run4}/cache/gsea_crosstissue__{lib}.csv"
        d = table(rel)
        row(lab, f"cross-tissue GSEA {lib}: pathways at FDR<0.05", n_significant(d), rel)
        if not lib.startswith("MSigDB"):
            continue
        for term in HALLMARK_NAMED:
            hit = d[d.Term.str.endswith(term)]
            undefined = term.startswith("Epithelial") and len(hit) > 0 and float(hit["NES"].iloc[0]) == 1.0
            for col in ("NES", "FDR q-val"):
                value = float(hit[col].iloc[0]) if len(hit) else np.nan
                paper = f"{FIG_HALLMARK} c ; ssec:er-axis"
                note = ""
                if undefined:
                    paper = "sssec:flip-test" if col == "NES" else ""
                    note = ("no defined enrichment: the enrichment score is +0.20, but no permuted gene set of size 80 reaches a "
                            "score of 0 or more (72% of the rank metric is negative), so gseapy reports NES = 1 and p = 1")
                row(lab, f"cross-tissue Hallmark {term}: {col}", value, rel, paper, note)

    # ---- summaries of the confidence and the ortholog QC notebooks
    for rel in ("notebooks/cache/alpha_confidence/stable_high_confidence_genes.csv",
                "notebooks/cache/alpha_confidence/variable_confidence_genes.csv"):
        row("7 cohorts", f"rows in {Path(rel).name}", len(table(rel)), rel)
    rel = "notebooks/cache/ortholog_qc_summary.csv"
    summary = table(rel).set_index("dataset")
    for k in TESTIS:
        for col in ("top_M", "top_F"):
            cell = summary.loc[QC_NAME[k], col]
            if cell != "—":  # the table writes a dash when no gene carries the flag
                row(LABEL[k], f"ortholog QC summary: {col}", cell, rel)

    df = pd.DataFrame(ROWS)
    df["value"] = df["value"].map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) and x == x else
                                  ("" if isinstance(x, float) else x))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows")


if __name__ == "__main__":
    main()
