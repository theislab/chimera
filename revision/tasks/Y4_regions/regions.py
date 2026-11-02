"""Y4 (EXPLORATORY): the list of regions, as functions of a cohort's gene table. No gold value enters a region.
Quantile bands are computed on all genes of the cohort, both splits. Fixed before any association of this task.

Axes and mechanism-guided hypotheses (``MECH``). Direction and reasoning are stated in the ledger before the screen.
``SEEN`` marks axes whose strata were looked at on all genes in X1-X6 or T4.3, so they are not naive on even genes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from revision.tasks.Y4_regions import rules as RU


def band(x: pd.Series, k: int) -> pd.Series:
    """1..k by rank over defined values. Ties share a band, so a heavy tie stays together."""
    r = x.rank(method="average") / x.notna().sum()
    return np.ceil(r * k).clip(1, k)


def _bands(t, col, k, prefix):
    if col not in t or t[col].notna().sum() < 100:
        return {}
    b = band(t[col], k)
    return {f"{prefix}_{'Q' if k == 5 else 'T'}{i}": (b == i).to_numpy() for i in range(1, k + 1)}


def regions(t: pd.DataFrame, cohort: str) -> list[dict]:
    """List of dict(name, axis, mech, seen, mask)."""
    out: list[dict] = []
    tissue = "testis" if cohort.startswith("test") else "liver"

    def add(axis, d, mech=False, seen=False):
        for k, m in d.items():
            out.append(dict(name=k, axis=axis, mech=mech, seen=seen, mask=np.asarray(m, bool)))

    pi = t["prior_raw"]
    add("all", {"all": np.ones(len(t), bool)}, seen=True)
    # sequence similarity: M1, M2
    d = _bands(t, "prior_raw", 5, "pi")
    b10 = band(pi, 10)
    d.update(pi_D1=b10 == 1, pi_D10=b10 == 10, pi_ge95=pi >= 0.95, pi_ge99=pi >= 0.99, pi_ge85=pi >= 0.85,
             pi_85to95=(pi >= 0.85) & (pi < 0.95), pi_lt70=pi < 0.70, pi_lt50=pi < 0.50)
    add("pi", d, mech=True, seen=True)
    # pi_raw is capped at 0.95: inside the cap the prior is one number. True identity still varies there.
    idt = t["id_true"]
    capped = pi >= 0.95
    add("identity", dict(id_ge90=idt >= 0.90, id_ge95=idt >= 0.95, id_ge98=idt >= 0.98, id_ge99=idt >= 0.99, id_eq100=idt >= 0.9999,
                         id_lt60=idt < 0.60, id_lt40=idt < 0.40, cap__id_95to98=capped & (idt < 0.98),
                         cap__id_98to99=capped & (idt >= 0.98) & (idt < 0.99), cap__id_ge99=capped & (idt >= 0.99)), mech=True)
    if capped.sum() > 200:
        for name, col in (("expr", "log_cp10k_spmean"), ("tau", "tau_shared"), ("spread", "spread_y")):
            bb = band(t[col], 3)
            for i in (1, 2, 3):
                add("identity", {f"cap__{name}_T{i}": capped & (bb == i)}, mech=True)
        add("identity", {"cap__unmasked": capped & ~t["masked"], "cap__margin": capped & t["margin_support"].eq(True)}, mech=True)
    # expression and detection: M3, M6
    add("expr", {**_bands(t, "log_cp10k_spmean", 5, "expr"),
                 "expr_D10": band(t["log_cp10k_spmean"], 10) == 10}, mech=True)
    add("detection", _bands(t, "detection_spmean", 5, "det"))
    add("detection", _bands(t, "det_gap_y", 3, "detgap"))
    # variability and specificity: M4, M7
    add("variability", {**_bands(t, "spread_y", 3, "spread"), **_bands(t, "hv_resid", 3, "hvresid"),
                        **_bands(t, "logvar_within", 3, "wvar"), **_bands(t, "logvar_total", 3, "tvar")}, mech=True)
    add("specificity", {**_bands(t, "tau_shared", 5, "tau"), **_bands(t, "peak_margin", 3, "peakmargin")}, mech=True)
    # margin support: M9
    ms = t["margin_support"].eq(True).to_numpy()
    add("margin", dict(masked=t["masked"], unmasked=~t["masked"], margin_support=ms,
                       lacks_margin_support=t["lacks_margin_support"].eq(True)), mech=True, seen=True)
    um = t["S"].abs().where(~t["masked"])
    add("margin", _bands(pd.DataFrame(dict(absS=um)), "absS", 3, "absS"), seen=True)
    if t["n_seeds"].max() > 1:
        add("margin", dict(seedstable=t["sign_consistency"] >= 0.999, seedunstable=t["sign_consistency"] < 0.999))
    # projection fidelity: M8
    add("projection", _bands(t, "moran_S", 3, "moranS"), mech=True)
    # level gap: M5
    add("levelgap", {**_bands(t, "abs_shift_y", 3, "absshift"), **_bands(t, "frac_level_y", 3, "fraclevel"),
                     **_bands(t, "shape_y", 3, "shape")}, mech=True)
    # sequence constraint
    add("constraint", {**_bands(t, "dNdS", 3, "dNdS"), **_bands(t, "LOEUF", 3, "LOEUF")}, seen=True)
    # classes
    cls = [c for c in t.columns if c.startswith("cls_")]
    x3 = {"cls_TF", "cls_TF_homeodomain", "cls_TF_C2H2_ZF", "cls_TF_bHLH", "cls_TF_bZIP", "cls_TF_nuclear_rec",
          "cls_TF_forkhead", "cls_TF_other", "cls_protein_kinase", "cls_kinase_any", "cls_receptor", "cls_GPCR",
          "cls_enzyme", "cls_EC1", "cls_EC2", "cls_EC3", "cls_EC4", "cls_EC5", "cls_EC6", "cls_EC7", "cls_ion_channel",
          "cls_SLC_transporter", "cls_chromatin_reg", "cls_mitochondrial"}
    for c in cls:
        add("class", {c: t[c]}, seen=c in x3)
    for c in ("C1", "C2", "V1", "V2", "V3"):
        add("class", {f"set_{c}": t[c].eq(True)}, seen=True)
    for pre, ax in (("kw_", "keyword"), ("hm_", "hallmark"), ("fam_", "family"), ("grp_", "genegroup")):
        for c in [c for c in t.columns if c.startswith(pre)]:
            add(ax, {c: t[c]})
    # age and paralogs: M10
    for name, _ in RU.AGE_CLADES:
        add("age", {f"age_{name}": t["age_class"].eq(name)}, mech=True)
    add("age", {"age_old": t["age_class"].isin(["opisthokont", "bilaterian"]),
                "age_young": t["age_class"].isin(["tetrapod", "mammal"])}, mech=True)
    add("paralog", dict(no_paralog=t["n_paralog"] == 0, has_paralog=t["n_paralog"] > 0,
                        paralog_close=t["max_paralog_id"] >= 50, paralog_far=t["max_paralog_id"] < 50,
                        paralog_many=t["n_paralog"] >= 10, paralog_tandem=t["paralog_same_chrom"]), mech=True)
    # predominant cell type: M5b
    if "peak_y" in t:
        for ty in sorted(t["peak_y"].dropna().unique()):
            add("celltype", {f"peak_{ty}": t["peak_y"].eq(ty)}, mech=True, seen=True)
            add("celltype", {f"peakstrong_{ty}": t["peak_y"].eq(ty) & (band(t["peak_margin"], 3) == 3)}, mech=True)
        if "peak_agree" in t:
            add("celltype", dict(peak_agree=t["peak_agree"].eq(True), peak_disagree=t["peak_agree"].eq(False)))
    # baseline availability: M6b
    add("baseline_gap", dict(B1b_undefined=t["B1b_raw"].isna(), B1b_defined=t["B1b_raw"].notna()), mech=True, seen=True)
    # two-way grid, mechanism-guided on pi
    Q = band(pi, 5)
    two = {"expr": band(t["log_cp10k_spmean"], 3), "tau": band(t["tau_shared"], 3), "spread": band(t["spread_y"], 3)}
    for lab, q in (("pihi", Q == 5), ("pilo", Q == 1), ("pimid", Q == 3)):
        for name, bb in two.items():
            for i in (1, 2, 3):
                add("pi_grid", {f"{lab}__{name}_T{i}": q & (bb == i)}, mech=True)
        add("pi_grid", {f"{lab}__unmasked": q & ~t["masked"], f"{lab}__margin": q & ms,
                        f"{lab}__old": q & t["age_class"].isin(["opisthokont", "bilaterian"]),
                        f"{lab}__young": q & t["age_class"].isin(["tetrapod", "mammal"]),
                        f"{lab}__paralog": q & (t["n_paralog"] > 0), f"{lab}__noparalog": q & (t["n_paralog"] == 0),
                        f"{lab}__TF": q & t["cls_TF"], f"{lab}__enzyme": q & t["cls_enzyme"],
                        f"{lab}__membrane": q & t["cls_membrane"], f"{lab}__smallshift": q & (band(t["abs_shift_y"], 3) == 1)
                        if "abs_shift_y" in t else q & False}, mech=True)
    e3, ta3, sp3 = two["expr"], two["tau"], two["spread"]
    for i in (1, 2, 3):
        for j in (1, 3):
            add("expr_grid", {f"expr_T{i}__tau_T{j}": (e3 == i) & (ta3 == j),
                              f"expr_T{i}__spread_T{j}": (e3 == i) & (sp3 == j)}, mech=True)
    if "abs_shift_y" in t:
        ab3 = band(t["abs_shift_y"], 3)
        for i in (1, 3):
            for j in (1, 3):
                add("levelgap_grid", {f"absshift_T{i}__spread_T{j}": (ab3 == i) & (sp3 == j),
                                      f"absshift_T{i}__expr_T{j}": (ab3 == i) & (e3 == j)}, mech=True)
    return out


def usable(t: pd.DataFrame, reg: dict, gold: str, covs: list[str]) -> bool:
    """Coverage only: enough genes with gold and covariates on both splits, and enough ESM blocks."""
    ok = t[[gold] + covs].notna().all(1).to_numpy() & reg["mask"]
    for s in ("odd", "even"):
        m = ok & t["split"].eq(s).to_numpy()
        if m.sum() < RU.MIN_GENES or t.loc[m, "block"].nunique() < RU.MIN_BLOCKS:
            return False
    return True
