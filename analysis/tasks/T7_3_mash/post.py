"""Liver MASH: post-hoc reading of H-D4, descriptive, no Chimera readout. Added after h4.tsv was seen.

Two explanations of the primary calls are checked.

1. Tissue source. Gribben's healthy tissue is deceased-donor or explant tissue, the MASH tissue is
   needle biopsy. h4_programs.tsv puts each program's NES side by side: primary, NASH vs NAFLD
   (biopsy against biopsy), NASH vs biopsy-type controls, bulk GSE135251, biopsy-type vs donor
   tissue, mouse. ``source_confounded``: the primary human call is significant, the source contrast
   (q < 0.05) has the same sign, and bulk NASH F0-F3 does not agree (opposite sign or q >= 0.05).

2. Non-hepatocyte RNA in hepatocyte nuclei. A program built of stromal genes can move in hepatocyte
   nuclei through ambient RNA or doublets. h4_intrinsic.tsv repeats GSEA on genes detected in >= 5%
   of hepatocyte nuclei in both species, so genes that hepatocyte nuclei do not express leave.

Calibration. Gene-set permutation ignores correlation between genes, and h4_null.tsv shows it: with
permuted mouse labels it still calls about as many "transfers" as the real labels. h4_sample_perm.tsv
therefore gives each program sample-label empirical p-values, from the 50 permuted fits of each
species (NES against NES of the same program under permuted labels). Mouse draws keep only labelings
orthogonal to diet (3 of the 6 true SD mice among the 6 labeled SD).

    PYTHONPATH=<chimera repo> python analysis/tasks/T7_3_mash/post.py [--n-perm 2000]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analysis.tasks.T7_3_mash import hd4 as H4

OUT = H4.OUT
DE = H4.DE
SEED = 20261103


def _nes(args):
    name, seed = args
    return name, H4._gsea((name, 500, seed))[1].set_index(["library", "set"])["nes"]


def robust() -> pd.DataFrame:
    """Programs that hold on every check. Human sign the same in the primary, NASH vs NAFLD, biopsy-type
    controls and bulk (bulk q < 0.05); the source contrast does not explain it (q >= 0.05 or the other
    sign); mouse sample-label p at its floor region (<= 0.05) with or against the human direction."""
    w = pd.read_csv(OUT / "h4_programs.tsv", sep="\t").merge(
        pd.read_csv(OUT / "h4_sample_perm.tsv", sep="\t")[["library", "set", "p_sample_h", "p_sample_m_oriented", "p_sample_m_against"]],
        on=["library", "set"])
    i = pd.read_csv(OUT / "h4_intrinsic.tsv", sep="\t")
    i = i[i["human_contrast"] == "human_primary"][["library", "set", "h_nes", "m_nes"]].rename(
        columns={"h_nes": "intrinsic_h_nes", "m_nes": "intrinsic_m_nes"})
    w = w.merge(i, on=["library", "set"], how="left")
    sg = np.sign(w["h_nes"])
    same = (sg == np.sign(w["nes_nash_vs_nafld"])) & (sg == np.sign(w["nes_biopsy_ctrl"])) & (sg == np.sign(w["nes_bulk"])) & (w["q_bulk"] < 0.05)
    src_ok = (w["q_source"] >= 0.05) | (sg != np.sign(w["nes_source"]))
    k = same & src_ok
    w["reading"] = np.where(k & (w["p_sample_m_against"] <= 0.05), "robust_opposite",
                            np.where(k & (w["p_sample_m_oriented"] <= 0.05), "robust_transfer", ""))
    w["hepatocyte_intrinsic"] = (np.sign(w["intrinsic_h_nes"]) == sg) & (np.sign(w["intrinsic_m_nes"]) == np.sign(w["m_nes"]))
    r = w[w["reading"] != ""].sort_values(["reading", "h_nes"])
    r.to_csv(OUT / "h4_robust.tsv", sep="\t", index=False)
    return r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--robust-only", action="store_true")
    a = p.parse_args()
    if a.robust_only:
        r = robust()
        print(r[["library", "set", "h_nes", "nes_nash_vs_nafld", "nes_biopsy_ctrl", "nes_bulk", "nes_source", "m_nes", "reading",
                 "hepatocyte_intrinsic", "intrinsic_h_nes", "intrinsic_m_nes"]].round(2).to_string())
        return 0
    g = pd.read_csv(OUT / "h4.tsv", sep="\t")
    lg = pd.read_csv(OUT / "h4_human_contrasts.tsv", sep="\t")
    w = g[["library", "set", "size", "direction_h", "h_nes", "h_q", "m_nes", "m_q_any", "call"]].copy()
    for c, tag in [("human_nash_vs_nafld", "nash_vs_nafld"), ("human_biopsy_controls", "biopsy_ctrl"), ("human_nafld", "nafld"),
                   ("bulk_nash_f0f3", "bulk"), ("source_biopsy_vs_donor", "source")]:
        x = lg[lg["contrast"] == c][["library", "set", "nes", "q"]].rename(columns={"nes": f"nes_{tag}", "q": f"q_{tag}"})
        w = w.merge(x, on=["library", "set"], how="left")
    rob = pd.read_csv(OUT / "h4_sensitivity.tsv", sep="\t")
    keep = ["library", "set", "bulk_nash_f0f3", "human_nash_vs_nafld", "human_biopsy_controls", "lohd_same_call_frac", "lomo_same_call_frac"]
    w = w.merge(rob[keep].rename(columns={"bulk_nash_f0f3": "call_bulk", "human_nash_vs_nafld": "call_nash_vs_nafld",
                                          "human_biopsy_controls": "call_biopsy_ctrl"}), on=["library", "set"], how="left")
    sig = w["h_q"] < 0.05
    same_src = (np.sign(w["h_nes"]) == np.sign(w["nes_source"])) & (w["q_source"] < 0.05)
    bulk_ok = (np.sign(w["h_nes"]) == np.sign(w["nes_bulk"])) & (w["q_bulk"] < 0.05)
    w["source_confounded"] = sig & same_src & ~bulk_ok
    w.to_csv(OUT / "h4_programs.tsv", sep="\t", index=False)
    print(w.groupby("call")["source_confounded"].agg(["size", "sum"]).to_string())

    # intrinsic universe
    h = pd.read_csv(DE / "human_primary.tsv", sep="\t").dropna(subset=["gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
    m = pd.read_csv(DE / "mouse_primary.tsv", sep="\t").dropna(subset=["gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
    uni = H4.universe()
    d_h = h["det_frac"].reindex(uni["human_id"]).to_numpy() >= 0.05
    d_m = m["det_frac"].reindex(uni["mouse_id"]).to_numpy() >= 0.05
    uni = uni[d_h & d_m]
    H4._S.update(uni=uni, sets={k: H4.G.read_gmt(v) for k, v in H4.G.SETS.items()})
    res = {}
    for i, n in enumerate(["human_primary", "bulk_nash_f0f3", "human_nash_vs_nafld", "mouse_primary"]):
        res[n] = H4._gsea((n, a.n_perm, SEED + i))[1]
    out = []
    for n in ("human_primary", "bulk_nash_f0f3", "human_nash_vs_nafld"):
        c = H4.combine(res[n], res["mouse_primary"])
        c.insert(0, "human_contrast", n)
        out.append(c)
    r = pd.concat(out)
    r.insert(1, "n_universe", len(uni))
    r.to_csv(OUT / "h4_intrinsic.tsv", sep="\t", index=False)
    print(r.groupby(["human_contrast", "call"]).size().to_string())

    # sample-label calibration, on the primary universe
    H4._S.update(uni=H4.universe())
    perm = pd.read_csv(DE / "permutations.tsv", sep="\t")
    ph = perm.loc[perm["name"].str.startswith("perm_human"), "name"].tolist()
    pm = perm.loc[perm["name"].str.startswith("perm_mouse") & (perm["n_true_controls_kept"] == 3), "name"].tolist()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        nes = dict(ex.map(_nes, [(n, SEED + 100 + i) for i, n in enumerate(ph + pm)], chunksize=1))
    H = pd.DataFrame({n: nes[n] for n in ph})
    M = pd.DataFrame({n: nes[n] for n in pm})
    s = g.set_index(["library", "set"])[["size", "h_nes", "h_q", "m_nes", "call", "direction_h"]].copy()
    sign = np.sign(s["h_nes"])
    s["p_sample_h"] = (1 + (H.abs().ge(s["h_nes"].abs(), axis=0)).sum(1)) / (H.notna().sum(1) + 1)
    s["p_sample_m_oriented"] = (1 + (M.mul(sign, axis=0).ge(s["m_nes"] * sign, axis=0)).sum(1)) / (M.notna().sum(1) + 1)
    s["p_sample_m_against"] = (1 + (M.mul(-sign, axis=0).ge(-s["m_nes"] * sign, axis=0)).sum(1)) / (M.notna().sum(1) + 1)
    s["n_perm_human"], s["n_perm_mouse"] = len(ph), len(pm)
    s.reset_index().to_csv(OUT / "h4_sample_perm.tsv", sep="\t", index=False)
    print(len(ph), len(pm), s.groupby("call")[["p_sample_h", "p_sample_m_oriented", "p_sample_m_against"]].median().to_string())
    robust()
    return 0


if __name__ == "__main__":
    sys.exit(main())
