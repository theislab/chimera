"""05 section 6, Steps 1-4, applied mechanically to the scoreboard tables. Prints the row and
the modifiers and writes ``row.json``. Only D enters (section 6, last rule).

Re-runnable: when T5.6 adds tool columns, ``run.py --tools <csv>`` adds the ``F1_tools`` rows
(Delta_tool = rho_p(D_3) - rho_p(tool) on the paired universe) and this script then evaluates
TOOL-LOSS; until then TOOL-LOSS is "pending" (it can only reduce a claim, 05 section 5).

Usage: python -m revision.tasks.T4_3_scoreboard.decide <outdir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HYP = ["H1", "H2", "H3", "H4"]
J = ["B1b", "B1c", "B2b", "B3a", "B1z"]
SOI = 0.03  # smallest difference of interest

ROWS = [
    (1, "Not supported", "Delta-alpha is a model diagnostic: which genes the shared decoder explains relative to the prior. Drop gene-level biological-regime claims"),
    (2, "Weak only", "As row 1, weak estimates stated"),
    (3, "Wins", "The readout adds information beyond protein identity and expression conservation. Still a statistic, not a rate"),
    (4, "Loses but valid", "Direct baselines are more accurate where labels are rich. Recommend them alongside"),
    (5, "Mixed", "Per-gold sentences; no general claim of winning or losing"),
    (6, "No detectable difference", "Chimera recovers what direct analyses recover, in one joint fit; any single Loss is disclosed. \"Equivalent\" only under its section 5 rule"),
    (7, "Partial", "Claim limited to that gold's construct and tissue"),
]


def status(rho: float, lo: float, hi: float, p_holm: float) -> str:
    """Step 1, first match wins."""
    if rho <= -SOI and hi < 0:
        return "R"
    if rho >= 0.10 and lo > 0.03 and p_holm < 0.05:
        return "V"
    if rho >= SOI and lo > 0:
        return "W"
    return "N"


def head_to_head(d: pd.DataFrame) -> tuple[str, bool, dict]:
    """Step 2 for one H_i over J; and the section 5 'Equivalent' rule."""
    d = d.set_index("comparator").reindex(J)
    if d["delta"].isna().any():
        missing = d.index[d["delta"].isna()].tolist()
        raise ValueError(f"comparators without a value on U_pair: {missing}")
    win = bool(((d["delta"] >= SOI) & (d["pc_lo"] > 0)).all())
    loss = bool(((d["delta"] <= -SOI) & (d["sim95_hi"] < 0)).any())
    equiv = bool(((d["sim90_lo"] > -SOI) & (d["sim90_hi"] < SOI)).all())
    out = "Win" if win else ("Loss" if loss else "Tie")
    return out, equiv, d[["delta", "pc_lo", "pc_hi", "sim95_lo", "sim95_hi", "sim90_lo", "sim90_hi"]].round(4).to_dict("index")


def decide(outdir: Path) -> dict:
    sb = pd.read_csv(outdir / "scoreboard.csv")
    h2h = pd.read_csv(outdir / "head_to_head.csv")
    sets = pd.read_csv(outdir / "sets.csv")
    f3 = pd.read_csv(outdir / "family3.csv")
    gm_f = outdir.parent / "T4_3" / "golds" / "gold_meta.json"
    if not gm_f.exists():
        gm_f = Path(__file__).resolve().parents[3] / "results/revision/T4_3/golds/gold_meta.json"
    K_h3 = json.loads(gm_f.read_text())["H3"]["K_ge30_both"]

    # Step 1
    prim = sb[(sb.family == "F1") & (sb.score == "D")].set_index("hyp").reindex(HYP)
    st = {}
    for h in HYP:
        r = prim.loc[h]
        if h == "H3" and K_h3 < 3:
            st[h] = "N"  # 05 section 4: K < 3 gives p = 1 and status N
            continue
        st[h] = status(r.rho_p, r.ci_lo, r.ci_hi, r.p_holm)
    # Step 2
    hh, eq, hdet = {}, {}, {}
    for h in HYP:
        d = h2h[(h2h.id == f"{h}:h2h")]
        hh[h], eq[h], hdet[h] = head_to_head(d)
    # Step 3
    nV = sum(st[h] == "V" for h in HYP)
    nWk = sum(st[h] == "W" for h in HYP)
    a = sum(st[h] == "V" and hh[h] == "Win" for h in HYP)
    b = sum(hh[h] == "Loss" for h in HYP)
    conds = [nV == 0 and nWk == 0, nV == 0 and nWk >= 1, nV >= 2 and a >= 2 and b == 0,
             nV >= 1 and b >= 2 and b > a, nV >= 1 and a >= 1 and b >= 1, nV >= 2, nV == 1]
    k = next(i for i, c in enumerate(conds) if c)
    row_no, row_name, row_claim = ROWS[k]
    # Step 4
    common = sb[(sb.family == "F1_common") & (sb.score == "D")].set_index("hyp").reindex(HYP)
    stc = {h: status(common.loc[h].rho_p, common.loc[h].ci_lo, common.loc[h].ci_hi, common.loc[h].p_holm) for h in HYP}
    VW = ("V", "W")
    mods = {}
    mods["PATTERN-ONLY"] = bool((st["H3"] == "V" or st["H4"] == "V") and st["H1"] not in VW and st["H2"] not in VW
                                and (stc["H3"] == "V" or stc["H4"] == "V") and stc["H1"] not in VW and stc["H2"] not in VW)
    mods["LEVEL-ONLY"] = bool((st["H1"] == "V" or st["H2"] == "V") and st["H3"] not in VW and st["H4"] not in VW
                              and (stc["H1"] == "V" or stc["H2"] == "V") and stc["H3"] not in VW and stc["H4"] not in VW)
    mods["REVERSED-LIVER"] = st["H1"] == "R" or st["H3"] == "R"
    mods["REVERSED-TESTIS"] = st["H2"] == "R" or st["H4"] == "R"
    # FEATURES-ONLY (05 section 5): on more than half of V golds, rho_p(D | C + 64 PCs) < 0.03
    # or rho_p(B3b, G) >= rho_p(D, G) - 0.03
    fo = {}
    for h in HYP:
        pcs = f3[(f3.id == f"{h}:beyond_pcs")].iloc[0]
        b3 = sb[(sb.id == f"{h}:B3b")].iloc[0]
        fo[h] = dict(rho_p_D_given_C_plus_PCs=float(pcs.rho_p), rho_p_B3b=float(b3.rho_p),
                     rho_p_D=float(prim.loc[h].rho_p),
                     meets=bool(pcs.rho_p < SOI or b3.rho_p >= prim.loc[h].rho_p - SOI))
    vg = [h for h in HYP if st[h] == "V"]
    mods["FEATURES-ONLY"] = bool(len(vg) > 0 and sum(fo[h]["meets"] for h in vg) > len(vg) / 2)
    # COVERAGE: genes under 1% detection in shared types, on a gold defined there: n >= 300,
    # rho_p >= 0.10 with CI lower > 0.03 given C + 64 PCs, and D beats B3a and B3b there
    cov = {}
    for h in HYP:
        p = f3[f3.id == f"{h}:cov_pcs"].iloc[0]
        dd = h2h[h2h.id == f"{h}:cov_vs_B3"].set_index("comparator")
        beats = {c: bool(dd.loc[c, "delta"] >= SOI and dd.loc[c, "pc_lo"] > 0) if c in dd.index else False
                 for c in ("B3a", "B3b")}
        n = int(p.n) if np.isfinite(p.n) else 0
        ok = bool(n >= 300 and p.rho_p >= 0.10 and p.ci_lo > 0.03 and all(beats.values()))
        cov[h] = dict(n=n, rho_p=float(p.get("rho_p", np.nan)), ci_lo=float(p.get("ci_lo", np.nan)), beats=beats, meets=ok)
    mods["COVERAGE"] = any(v["meets"] for v in cov.values())
    tools = sb[sb.family == "F1_tools"]
    if len(tools):
        tl = h2h[h2h.id.str.contains(":tool:")]
        mods["TOOL-LOSS"] = bool(((tl["delta"] <= -SOI) & (tl["pc_hi"] < 0)).any())
    else:
        mods["TOOL-LOSS"] = "pending"
    f2 = sets[sets.family == "F2"]
    npass = int(f2["pass_predicted_direction"].astype(bool).sum())
    mods["PATHWAY-LEVEL"] = npass >= 5
    applied = [m for m, v in mods.items() if v is True]
    claim = row_claim
    if mods["PATHWAY-LEVEL"] and row_no in (1, 2):
        claim += ". PATHWAY-LEVEL: supported at the level of predefined gene sets, not gene by gene"
    equivalent = row_no == 6 and all(eq.values())
    res = dict(
        step1_status=st, step1_inputs={h: dict(rho_p=float(prim.loc[h].rho_p), ci_lo=float(prim.loc[h].ci_lo),
                                               ci_hi=float(prim.loc[h].ci_hi), p_perm=float(prim.loc[h].p_perm),
                                               p_holm=float(prim.loc[h].p_holm)) for h in HYP},
        step2_head_to_head=hh, step2_equivalent_rule=eq, step2_details=hdet,
        step3=dict(nV=nV, nWk=nWk, a=a, b=b, row_number=row_no, row=row_name, paper_claim=claim,
                   equivalent=equivalent),
        step4_modifiers=mods, modifiers_applied=applied, common_universe_status=stc,
        features_only_inputs=fo, coverage_inputs=cov, family2_passes=npass, family2_tests=int(len(f2)),
        h3_K=K_h3, h3_K_limited=K_h3 < 5,
    )
    return res


def main():
    outdir = Path(sys.argv[1])
    res = decide(outdir)
    tmp = outdir / "row.json.tmp"
    tmp.write_text(json.dumps(res, indent=2, default=str))
    tmp.replace(outdir / "row.json")
    s = res["step3"]
    print("Step 1 status:", res["step1_status"])
    print("Step 2 head-to-head:", res["step2_head_to_head"], "| Equivalent rule:", res["step2_equivalent_rule"])
    print(f"Step 3: nV={s['nV']} nWk={s['nWk']} a={s['a']} b={s['b']} -> row {s['row_number']}: {s['row']}")
    print("  claim:", s["paper_claim"])
    print("Step 4 modifiers:", res["step4_modifiers"])
    print("Applied:", res["modifiers_applied"] or "none", "| TOOL-LOSS:", res["step4_modifiers"]["TOOL-LOSS"])


if __name__ == "__main__":
    main()
