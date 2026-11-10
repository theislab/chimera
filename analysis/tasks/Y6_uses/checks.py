"""Mechanics checks and the tidy table in the benchmark schema.

    PYTHONPATH=$PWD python -m analysis.tasks.Y6_uses.checks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from analysis.tasks.Y6_uses import rules as RU
from analysis.tasks.Y6_uses.use1 import OUT as U1, ap_setup, wap

METH = {"S_row": "Chimera species margin", "S_gene": "Chimera S_g", "D_neg": "Chimera delta-alpha", "zero_s": "Zero fraction", "zero_gap": "Zero-fraction gap", "B1_neg": "B1 (undefined as worst)", "id_neg": "Identity",
        "C_chimera": "Chimera species margin", "C_pairfits": "Chimera, pair fits", "C_Dpair": "Chimera delta-alpha, pair fits", "C_B1": "B1", "C_id": "Identity", "C_own": "Own-species margin", "C_dir": "Direct margin",
        "S_mouse": "Chimera species margin", "E_mouse": "B1", "id_mouse": "Identity", "S": "Chimera S_g", "D": "Chimera delta-alpha", "B1": "B1", "identity": "Identity", "random": "Random",
        "S_ema": "Chimera S_g", "S_row_eye": "Chimera species margin", "E_own": "B1", "id_own": "Identity", "gap_neg": "Level gap", "identity_s": "Identity"}


def checks():
    d = pd.read_parquet(U1 / "scored_rows.parquet")
    rows = []
    rng = np.random.default_rng(RU.SEED)
    for fit, lab, label in (("bone_marrow", "mouse_lemur", "T_later"), ("test_3sp", "Macaque", "T_newasm"), ("vz_full", "pig", "T_lowconf")):
        x = d[(d.fit == fit) & (d.species_label == lab) & (d.det_other >= RU.DET_EXPR) & (d.split == "even") & d[label].notna()]
        y, s = x[label].to_numpy(float), x.S_row.to_numpy(float)
        mine, sk = wap(y, np.ones(len(y)), ap_setup(s)), average_precision_score(y, s)
        perm = [wap(y, np.ones(len(y)), ap_setup(rng.permutation(s))) / y.mean() for _ in range(200)]
        rows.append(dict(fit=fit, species=lab, label=label, ap_here=mine, ap_sklearn=sk, ap_lift_permuted_mean=np.mean(perm), ap_lift_permuted_q975=np.percentile(perm, 97.5)))
    pd.DataFrame(rows).to_csv(RU.OUT / "checks_use1.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())
    # Y4_regions saw UBE2G2 near absent in mouse at identity 1.00.
    print(d[d.symbol == "UBE2G2"][["fit", "species_label", "identity_s", "S_gs", "det_s", "det_other", "goc", "wga", "high_conf", "homology_type_116", "T_lowconf", "T_later"]].round(3).to_string())


def tidy():
    out = []
    m = pd.read_csv(U1 / "metrics.csv")
    m = m[m.score.notna() & m.note.isna()]
    for r in m.itertuples():
        base = dict(track="Y6", task=f"Y6|annotation|{r.label}|{r.fit}|{r.species_label}", setting=f"U_{r.universe}; even autosomes; threshold tuned on odd", method=METH[r.score], n=r.n, rank_direction="higher")
        out.append(dict(base, metric="AP lift", value=r.ap_lift, ci_lo=r.ap_lift_lo, ci_hi=r.ap_lift_hi))
        out.append(dict(base, metric="precision", value=r.precision, ci_lo=r.precision_lo, ci_hi=r.precision_hi))
        out.append(dict(base, metric="recall", value=r.recall, ci_lo=r.recall_lo, ci_hi=r.recall_hi))
    u2 = pd.read_csv(RU.OUT / "use2/rho.csv")
    for r in u2[u2.score.isin(METH)].itertuples():
        out.append(dict(track="Y6", task=f"Y6|species_pick|{r.test}", setting=f"{r.split} autosomes", method=METH["S_row_eye" if r.score == "S_row" else r.score], metric="rho_p", value=r.rho_p, ci_lo=r.ci_lo, ci_hi=r.ci_hi, n=r.n, rank_direction="higher"))
    for r in pd.read_csv(RU.OUT / "use2/pick.csv").itertuples():
        out.append(dict(track="Y6", task=f"Y6|species_pick|{r.test}", setting="clear cases", method=METH["S_row_eye" if r.score == "S_row" else r.score], metric="pick accuracy", value=r.pick_acc, ci_lo=r.lo, ci_hi=r.hi, n=r.n_clear, rank_direction="higher"))
    u3 = pd.read_csv(RU.OUT / "use3/ladder.csv")
    for r in u3[u3.ladder.isin(["C0", "C5"])].itertuples():
        out.append(dict(track="Y6", task=f"Y6|drug_targets|{r.y}|{r.tissue}", setting=f"held fixed: {r.ladder}", method=METH[r.score], metric="rho_p", value=r.rho_p, ci_lo=r.ci_lo, ci_hi=r.ci_hi, n=r.n, rank_direction="" if r.score == "random" else "higher"))
    u4 = pd.read_csv(RU.OUT / "use4/rho.csv")
    for r in u4.itertuples():
        out.append(dict(track="Y6", task=f"Y6|cis_hybrid|{r.fit}|{r.gold}", setting="proxy: human-macaque or human-mouse readout, human-chimpanzee cis reference", method=METH[r.score], metric="rho_p", value=r.rho_p, ci_lo=r.ci_lo, ci_hi=r.ci_hi, n=r.n, rank_direction="higher"))
    t = pd.DataFrame(out)[RU.BENCH]
    t.to_csv(RU.OUT / "benchmark_rows.csv", index=False)
    print(t.shape, t.isna().sum().to_dict())


if __name__ == "__main__":
    checks()
    tidy()
