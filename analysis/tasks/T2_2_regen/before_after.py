"""Before/after table for the T2.2 regeneration (testis gene identity and cross-tissue joins).

One row per affected published number. ``old_value`` is the submitted value: read from the
stale copy (results/analysis/T2_2/stale/) when the number lives in a file, otherwise the
manuscript value. ``old_reproduced`` recomputes it here with the old behaviour (liver
ortholog table on testis var_names, cross-tissue joins on var_names), as a check that the
old number is explained by the bug. ``new_value`` uses the corrected behaviour
(cohort-correct table, joins on hsapiens_gene_id) with the same formulas as the notebooks,
or is read from the regenerated files.

Run from the repository root:
    python analysis/tasks/T2_2_regen/before_after.py
"""

from __future__ import annotations

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
STALE = OUT / "stale"
RUNS = REPO / "training" / "runs"
DATA = REPO / "data" / "processed"

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
SCRIPT = "analysis/tasks/T2_2_regen/before_after.py"

_C: dict[str, dict] = {}


def cohort(key: str) -> dict:
    if key not in _C:
        run, h5, tissue = COHORTS[key]
        fs = dict(np.load(RUNS / run / "final_state.npz", allow_pickle=False))
        a = ad.read_h5ad(DATA / h5, backed="r")
        vn = np.asarray(a.var_names)
        a.file.close()
        _C[key] = {"vn": vn, "tissue": tissue, "run": run, "delta": fs["alpha"] - fs["prior"]}
    return _C[key]


def symbols(key: str, old: bool) -> pd.Series:
    c = cohort(key)
    return pu.annotate_genes_with_names(c["vn"], "liver" if old else c["tissue"])


def join(kA: str, kB: str, old: bool) -> tuple[np.ndarray, np.ndarray]:
    """Positions of shared genes. Old: var_names intersection. New: hsapiens_gene_id."""
    A, B = cohort(kA), cohort(kB)
    if old:
        posB = {v: i for i, v in enumerate(B["vn"])}
        iA = np.array([i for i, v in enumerate(A["vn"]) if v in posB])
        iB = np.array([posB[v] for v in A["vn"][iA]])
        return iA, iB
    return pu.shared_genes(A["vn"], A["tissue"], B["vn"], B["tissue"])


def program_rho(kA: str, kB: str, old: bool) -> dict:
    """03 `category_means_aligned` + `_category_pair_panel`: k programs, Spearman, Pearson."""
    iA, iB = join(kA, kB, old)
    vA, vB = cohort(kA)["delta"][iA], cohort(kB)["delta"][iB]
    common = cohort(kA)["vn"][iA]
    sym = symbols(kA, old).values[iA]
    cats = gce.annotate_extended(common, pd.Series(sym, index=common))["category"].values
    df = pd.DataFrame({"cat": cats, "vA": vA, "vB": vB})
    sizes = df.groupby("cat").size()
    keep = [c for c in gce.CATEGORY_ORDER_EXTENDED if c != "other" and sizes.get(c, 0) >= 5]
    mA = np.array([df.loc[df.cat == c, "vA"].mean() for c in keep])
    mB = np.array([df.loc[df.cat == c, "vB"].mean() for c in keep])
    return {"n": len(iA), "k": len(keep), "rho": stats.spearmanr(mA, mB).correlation,
            "r": stats.pearsonr(mA, mB).statistic,
            "gene_rho": stats.spearmanr(vA, vB).correlation}


def category_effects(key: str, old: bool, broad: bool) -> pd.DataFrame:
    """04 / 04_simpler `plot_effect_size_heatmap_for_dataset`: d (shift), d_amp, BH q."""
    c = cohort(key)
    sym = symbols(key, old)
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


def quadrants(old: bool) -> dict:
    """08_ortholog_qc §I.3: mean Δα per tissue, joined across tissues, sign quadrants."""
    def pool(keys):
        frames = []
        for k in keys:
            c = cohort(k)
            if old:
                gid = c["vn"]  # old join key: ortho_N
            else:
                gid = pu.ensembl_ids(c["vn"], c["tissue"]).values
            frames.append(pd.DataFrame({"g": gid, "d": c["delta"]}))
        return pd.concat(frames).dropna(subset=["g"]).groupby("g")["d"].mean()
    p, t = pool(LIVER), pool(TESTIS)
    xt = pd.concat([p.rename("p"), t.rename("t")], axis=1, join="inner")
    out = {"n": len(xt),
           "M|M": int(((xt.p < 0) & (xt.t < 0)).sum()), "F|F": int(((xt.p > 0) & (xt.t > 0)).sum()),
           "M liver/F testis": int(((xt.p < 0) & (xt.t > 0)).sum()),
           "F liver/M testis": int(((xt.p > 0) & (xt.t < 0)).sum())}
    out["flip_share"] = (out["M liver/F testis"] + out["F liver/M testis"]) / out["n"]
    out["gene_rho"] = stats.spearmanr(xt.p, xt.t).correlation
    return out


ROWS: list[dict] = []


def row(claim, location, cohort_label, quantity, old, old_src, new, new_src, old_rep=np.nan, note=""):
    ROWS.append({"claim": claim, "location": location, "cohort": cohort_label, "quantity": quantity,
                 "old_value": old, "old_source": old_src, "new_value": new, "new_source": new_src,
                 "old_reproduced": old_rep, "note": note})


def stale_or_none(rel: str) -> pd.DataFrame | None:
    p = STALE / rel
    return pd.read_csv(p) if p.exists() else None


def main() -> None:
    ms = "chimera-manuscript"
    # ---- Fig 2 Supp (supp-alpha-robustness) g-i: testis within-tissue program rho
    published_gi = {("test_3sp", "test_hmac"): ("g", 14640, 0.911, 0.990),
                    ("test_3sp", "test_hmouse"): ("h", 14731, 0.821, 0.975),
                    ("test_hmac", "test_hmouse"): ("i", 14551, 0.939, 0.993)}
    for (kA, kB), (panel, n_pub, rho_pub, r_pub) in published_gi.items():
        o, n = program_rho(kA, kB, True), program_rho(kA, kB, False)
        loc = f"{ms} supplementary_figures:88, Fig supp-alpha-robustness ({panel})"
        lab = f"{LABEL[kA]} vs {LABEL[kB]}"
        src = f"{SCRIPT} (03 category_means_aligned)"
        row("C01", loc, lab, "n shared genes", n_pub, "manuscript", n["n"], src, o["n"])
        row("C01", loc, lab, "k programs", np.nan, "", n["k"], src, o["k"])
        row("C01", loc, lab, "program Spearman rho", rho_pub, "manuscript", n["rho"], src, o["rho"])
        row("C01", loc, lab, "program Pearson r", r_pub, "manuscript", n["r"], src, o["r"])

    # ---- Fig 2 Supp j-k and extended_results:8: cross-tissue program rho
    published_jk = {("liv_hmonkey", "test_hmac"): ("j", 7810, 0.016, -0.709),
                    ("liv_hmouse", "test_hmouse"): ("k", 7702, -0.352, -0.847)}
    for (kA, kB), (panel, n_pub, rho_pub, r_pub) in published_jk.items():
        o, n = program_rho(kA, kB, True), program_rho(kA, kB, False)
        loc = f"{ms} extended_results:8; supplementary_figures:88 ({panel})"
        lab = f"{LABEL[kB]} vs {LABEL[kA]}"
        src = f"{SCRIPT} (03 §B.4)"
        row("C02", loc, lab, "n shared genes", n_pub, "manuscript", n["n"], src, o["n"])
        row("C02", loc, lab, "k programs", np.nan, "", n["k"], src, o["k"])
        row("C02", loc, lab, "program Spearman rho", rho_pub, "manuscript", n["rho"], src, o["rho"])
        row("C02", loc, lab, "program Pearson r", r_pub, "manuscript", n["r"], src, o["r"])
        row("C02", f"03 §B.4 panel B4_crosstissue_*_delta", lab, "gene-level Spearman rho (clamped prior)",
            o["gene_rho"], "old join, recomputed", n["gene_rho"], src, o["gene_rho"],
            "not quoted in text; panel annotation")

    # ---- 05 §C.2 / flip table: gene-level rho liver 4-sp vs testis 3-sp
    flip = stale_or_none(f"training/runs/{COHORTS['liv_4sp'][0]}/tables/cross_tissue_delta_alpha_flip.csv")
    o, n = program_rho("liv_4sp", "test_3sp", True), program_rho("liv_4sp", "test_3sp", False)
    old_rho = stats.spearmanr(flip.delta_alpha_panc, flip.delta_alpha_test).correlation if flip is not None else np.nan
    row("C02", "05 §C.2 text ('rho = -0.016 on ~9K shared genes'); flip table", "liver 4-sp vs testis 3-sp",
        "n shared genes", len(flip) if flip is not None else np.nan, "stale cross_tissue_delta_alpha_flip.csv",
        n["n"], SCRIPT, o["n"])
    row("C02", "05 §C.2 text; flip table", "liver 4-sp vs testis 3-sp", "gene-level Spearman rho (clamped prior)",
        old_rho, "stale cross_tissue_delta_alpha_flip.csv", n["gene_rho"], SCRIPT, o["gene_rho"],
        "05 is not in the T2.2 rerun list; the flip table is regenerated when 05 reruns")
    row("C02", "05 §C.2 program level", "liver 4-sp vs testis 3-sp", "program Spearman rho",
        np.nan, "", n["rho"], SCRIPT, o["rho"])

    # ---- Fig 7e quadrants (08_ortholog_qc §I.3)
    qo, qn = quadrants(True), quadrants(False)
    loc7 = f"{ms} supplementary_figures:350 (Fig 7e); main.tex:336"
    src7 = "08_ortholog_qc.ipynb §I.3 printed counts (rerun); checked by " + SCRIPT
    pub7 = {"n": 8879, "M|M": 1900, "F|F": 1833, "M liver/F testis": 4267, "F liver/M testis": 879}
    for k, v in pub7.items():
        row("C02", loc7, "mean of 4 liver vs mean of 3 testis", f"Fig 7e {k}" if k != "n" else "Fig 7e n shared",
            v, "manuscript", qn[k], src7, qo[k])
    row("C02", loc7, "mean of 4 liver vs mean of 3 testis", "share of genes changing sign class",
        np.nan, "main.tex:336 'more than half'", qn["flip_share"], SCRIPT, qo["flip_share"])
    row("C02", loc7, "mean of 4 liver vs mean of 3 testis", "gene-level Spearman rho of tissue means",
        np.nan, "", qn["gene_rho"], SCRIPT, qo["gene_rho"])

    # ---- Broad umbrellas, testis H+Macaque (Fig supp-dalpha-broad-categories d; extended_results:18)
    pub_broad = {"Mitochondria & OXPHOS": 1.17, "Metabolism": 0.75, "Protein homeostasis": 0.68,
                 "Cytoskeleton & ECM": 0.68, "Signaling pathways": 0.62, "Immunity & inflammation": 0.57,
                 "Chromatin & DNA processing": 0.55, "Cell cycle": 0.53, "Transcription factors": 0.52,
                 "Translation & ribosome": 0.51}
    bo, bn = category_effects("test_hmac", True, True), category_effects("test_hmac", False, True)
    liv_b = category_effects("liv_4sp", False, True)
    locb = f"{ms} supplementary_figures:167 (d); extended_results:18"
    srcb = f"{SCRIPT} (04_simpler §A.3 effect-size heatmap)"
    for cat, v in pub_broad.items():
        row("C01", locb, "testis H+Macaque", f"broad d (shift): {cat}", v, "manuscript",
            bn.d.get(cat, np.nan), srcb, bo.d.get(cat, np.nan))
        row("C01", locb, "testis H+Macaque", f"broad n genes: {cat}", np.nan, "",
            bn.n.get(cat, np.nan), srcb, bo.n.get(cat, np.nan))
    row("C01", locb, "testis H+Macaque", "broad d_amp: Mitochondria & OXPHOS", 0.62, "manuscript",
        bn.d_amp.get("Mitochondria & OXPHOS"), srcb, bo.d_amp.get("Mitochondria & OXPHOS"))
    neg = [c for c in liv_b.index if liv_b.d[c] < 0]
    row("C05", locb, "liver 4-sp vs testis H+Macaque", "liver-negative umbrellas positive on testis",
        "all", "manuscript ('every broad umbrella')",
        f"{sum(bn.d.get(c, -1) > 0 for c in neg)}/{len(neg)}", srcb,
        f"{sum(bo.d.get(c, -1) > 0 for c in neg)}/{len(neg)}")
    named = ["Mitochondria & OXPHOS", "Translation & ribosome", "Metabolism", "Protein homeostasis", "Cell cycle"]
    row("C01", "extended_results:18", "testis H+Macaque", "max shift q over the 5 named umbrellas", 1.16e-7,
        "manuscript", bn.q_d.reindex(named).max(), srcb, bo.q_d.reindex(named).max(),
        "published value equals the old Mitochondria & OXPHOS q (1.159e-7); old Cell cycle q was 4.3e-6")
    for cat in named:
        row("C01", "extended_results:18", "testis H+Macaque", f"broad shift q: {cat}", np.nan, "",
            bn.q_d.get(cat, np.nan), srcb, bo.q_d.get(cat, np.nan))

    # ---- 38 narrow categories, testis H+Macaque (Fig supp-dalpha-testis-detailed, line 198)
    pub_narrow = {
        "mtdna-encoded": 4.36, "metabolism-glycolysis": 0.89, "cytoskeleton-actin": 0.68,
        "ribosomal-structural": 0.64, "cytoskeleton-intermediate": 0.99, "cytoskeleton-microtubule": 0.84,
        "housekeeping": 0.70, "ecm-collagen": 0.61, "glutathione-redox": 0.86, "cytokines-chemokines": 0.63,
        "cytochrome-p450": 0.69, "splicing-factors": 0.63, "kinases-signaling": 0.74, "tf-homeobox": 0.57,
        "chaperones-hsp": 0.63, "oxphos-nuclear-encoded": 0.70, "tgfb-bmp-signaling": 0.49, "proteasome": 0.54,
        "complement": 0.51, "nuclear-receptors": 0.59, "hedgehog-signaling": 0.71, "cell-cycle": 0.53,
        "metabolism-lipid": 0.63, "autophagy": 1.00, "dna-replication-repair": 0.53, "ribosomal-mt": 0.50,
        "apoptosis": 0.67, "tf-zinc-finger": 0.54, "immediate-early": 0.56, "ecm-other": 0.47,
        "translation-elongation": 0.28, "notch-signaling": 0.24, "tf-forkhead-etc": 0.38,
        "interferon-response": 0.68, "ribosome-biogenesis": 0.49, "translation-initiation": 0.43,
        "wnt-signaling": 0.49, "coagulation-serum": 0.62}
    no, nn = category_effects("test_hmac", True, False), category_effects("test_hmac", False, False)
    locn = f"{ms} supplementary_figures:192-198; extended_results:18"
    srcn = f"{SCRIPT} (04 §A.3 effect-size heatmap)"
    mt_key = "mitochondrial_dna_encoded"
    for cat, v in pub_narrow.items():
        key = mt_key if cat == "mtdna-encoded" else cat.replace("-", "_")
        row("C01", locn, "testis H+Macaque", f"narrow d (shift): {key}", v, "manuscript",
            nn.d.get(key, np.nan), srcn, no.d.get(key, np.nan))
    for col, pub in (("n", np.nan), ("q_d", 0.0234), ("q_amp", 3.26e-4)):
        row("C01", locn, "testis H+Macaque", f"narrow {col}: {mt_key}", pub,
            "manuscript" if pub == pub else "", nn[col].get(mt_key, np.nan), srcn, no[col].get(mt_key, np.nan),
            "BH q pooled over both columns of the panel as in 04; the published q is not reproduced exactly "
            "with the old labels (see old_reproduced), d is" if col != "n" else "")

    # ---- GSEA, testis cohorts (Fig 4 Supp; 06_pathway_analysis)
    for k in TESTIS:
        run = COHORTS[k][0]
        for lib, metric in (("MSigDB_Hallmark_2020", "delta"), ("MSigDB_Hallmark_2020", "prior"),
                            ("Reactome_Pathways_2024", "delta")):
            rel = f"training/runs/{run}/cache/gsea__{lib}__{metric}.csv"
            o = stale_or_none(rel)
            n_ = pd.read_csv(REPO / rel) if (REPO / rel).exists() else None
            cnt = lambda d: int((pd.to_numeric(d["FDR q-val"], errors="coerce") < 0.05).sum()) if d is not None else np.nan  # noqa: E731
            row("C01", "Fig 4 Supp (GSEA); 06_pathway_analysis", LABEL[k],
                f"GSEA {lib} {metric}: pathways at FDR<0.05", cnt(o), f"stale {rel}", cnt(n_), rel)
            if o is not None and n_ is not None:
                m = o.merge(n_, on="Term", suffixes=("_o", "_n"))
                row("C01", "06_pathway_analysis", LABEL[k], f"GSEA {lib} {metric}: NES Spearman old vs new",
                    np.nan, "", stats.spearmanr(m["NES_o"], m["NES_n"]).correlation, rel)
        rel = f"training/runs/{run}/tables/gsea_chimera_unique.csv"
        o, n_ = stale_or_none(rel), (pd.read_csv(REPO / rel) if (REPO / rel).exists() else None)
        row("C01", "06_pathway_analysis §C (Chimera-unique pathways)", LABEL[k], "rows in gsea_chimera_unique",
            len(o) if o is not None else np.nan, f"stale {rel}", len(n_) if n_ is not None else np.nan, rel)

    # ---- Cross-tissue GSEA (Fig supp-dalpha-hallmark c; extended_results:20)
    run4 = COHORTS["liv_4sp"][0]
    for lib in ("MSigDB_Hallmark_2020", "Reactome_Pathways_2024"):
        rel = f"training/runs/{run4}/cache/gsea_crosstissue__{lib}.csv"
        o, n_ = stale_or_none(rel), (pd.read_csv(REPO / rel) if (REPO / rel).exists() else None)
        cnt = lambda d: int((pd.to_numeric(d["FDR q-val"], errors="coerce") < 0.05).sum()) if d is not None else np.nan  # noqa: E731
        row("C02", f"{ms} supplementary_figures:206 (c); extended_results:20", "liver 4-sp minus testis 3-sp",
            f"cross-tissue GSEA {lib}: pathways at FDR<0.05", cnt(o), f"stale {rel}", cnt(n_), rel)
        if lib.startswith("MSigDB") and o is not None and n_ is not None:
            for term, (nes_pub, q_pub) in {"Bile Acid Metabolism": (-1.679, 0.00278),
                                           "Xenobiotic Metabolism": (-1.663, 0.00324),
                                           "Fatty Acid Metabolism": (-1.544, 0.0148),
                                           "Cholesterol Homeostasis": (-1.519, 0.0169),
                                           "Epithelial Mesenchymal Transition": (1.664, 0.0234)}.items():
                g = lambda d, c: float(d.loc[d.Term.str.endswith(term), c].iloc[0]) if d.Term.str.endswith(term).any() else np.nan  # noqa: E731
                for col, pub in (("NES", nes_pub), ("FDR q-val", q_pub)):
                    note = ""
                    if term.startswith("Epithelial") and g(n_, "NES") == 1.0:
                        note = ("undefined on the corrected ranking: ES=+0.20 but no permuted gene set of size 80 "
                                "reaches ES>=0 (72% of the rank metric is negative), so gseapy reports NES=1, p=1. "
                                "Old NES rested on very few positive nulls (1.66 with gseapy 1.1.13, 1.82 with 1.3.2)")
                    row("C02", f"{ms} supplementary_figures:206 (c)", "liver 4-sp minus testis 3-sp",
                        f"cross-tissue Hallmark {term}: {col}", g(o, col), f"stale {rel} (manuscript {pub})",
                        g(n_, col), rel, note=note)

    # ---- Testis top/bottom-30 gene tables (04 §C): same genes, labels corrected
    for k in TESTIS:
        run = COHORTS[k][0]
        for tail in ("top30", "bot30"):
            rel = f"training/runs/{run}/tables/delta_alpha_{tail}.csv"
            o, n_ = stale_or_none(rel), (pd.read_csv(REPO / rel) if (REPO / rel).exists() else None)
            if o is None or n_ is None:
                continue
            col = "symbol" if "symbol" in o.columns else o.columns[0]
            same = len(set(o[col].astype(str)) & set(n_[col].astype(str)))
            row("C01", "04 §C tables; Fig 4 Supp gene lists", LABEL[k], f"{tail}: gene labels unchanged (of 30)",
                30, f"stale {rel}", same, rel)

    # ---- Ortholog QC and confidence summaries (notebooks/cache)
    for rel, key in (("notebooks/cache/alpha_confidence/stable_high_confidence_genes.csv", None),
                     ("notebooks/cache/alpha_confidence/variable_confidence_genes.csv", None)):
        o, n_ = stale_or_none(rel), (pd.read_csv(REPO / rel) if (REPO / rel).exists() else None)
        row("C02", "08_alpha_integration_confidence §D.3", "7 cohorts", f"rows in {Path(rel).name}",
            len(o) if o is not None else np.nan, f"stale {rel}", len(n_) if n_ is not None else np.nan, rel)

    # ---- 08_ortholog_qc summary table: report every testis cell that changed
    rel = "notebooks/cache/ortholog_qc_summary.csv"
    o, n_ = stale_or_none(rel), (pd.read_csv(REPO / rel) if (REPO / rel).exists() else None)
    if o is not None and n_ is not None and o.shape == n_.shape:
        for i in range(len(o)):
            for c in o.columns:
                if str(o.iloc[i][c]) != str(n_.iloc[i][c]):
                    row("C01", "08_ortholog_qc §G.1 summary table", str(o.iloc[i, 0]), c,
                        o.iloc[i][c], f"stale {rel}", n_.iloc[i][c], rel)

    df = pd.DataFrame(ROWS)
    for c in ("old_value", "new_value", "old_reproduced"):
        df[c] = df[c].map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) and x == x else
                          ("" if isinstance(x, float) else x))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "before_after.csv", index=False)
    print(f"wrote {OUT / 'before_after.csv'}: {len(df)} rows")


if __name__ == "__main__":
    main()
