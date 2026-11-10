"""Readouts in the format of H1_harmonise: free-gate per-gene tables built by its own build_cohort on the kept free seeds, and per cohort
the comparison with the canonical reading (H1_harmonise): count dependence, agreement of the two gates, expression, masked genes, separate axis.

    PYTHONPATH=$PWD python -m analysis.tasks.Z4_free_gate_all.readouts [--jobs 4] [--force]
Inputs: results/analysis/Z4/kept.json (health.py), results/analysis/H1/genes/<cohort>.parquet (canonical).
Output: results/analysis/Z4/h1/genes/<cohort>.parquet (H1_harmonise schema, free gate), h1/identity_checks.csv, cohort_compare.csv, axis_free.csv,
axis_compare.csv, genes_compare/<cohort>.parquet (both gates side by side).
"""

from __future__ import annotations

import os

os.environ["H1_OUT"] = str(__import__("analysis.common.gene_ids", fromlist=["x"]).REPO_ROOT / "results/analysis/Z4/h1")  # before H1_harmonise is imported

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from analysis.tasks.H1_harmonise import recurrence as H1REC  # noqa: E402
from analysis.tasks.H1_harmonise import rules as H1R  # noqa: E402
from analysis.tasks.H1_harmonise import tables as H1T  # noqa: E402
from analysis.tasks.Z4_free_gate_all import rules as RU  # noqa: E402

FREE = RU.OUT / "h1"
CAN = RU.RES / "H1"
CMP = RU.OUT / "genes_compare"
assert H1R.OUT == FREE and H1T.GENES == FREE / "genes"


def build(args) -> dict:
    c, kept = args
    try:
        ck = H1T.build_cohort((c, {int(s): r for s, r in kept.items()}))
    except AssertionError as e:  # eye: H1 asserts that D equals the canonical tidy table, which a different gate cannot; the table is written first
        if "T11 tidy table" not in str(e) or not (H1T.GENES / f"{c}.parquet").exists():
            raise
        ck = dict(cohort=c, note="tidy-table identity check skipped (free gate differs from canonical by design)")
    return ck


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 20 else np.nan


def compare(c: str) -> dict:
    f = pd.read_parquet(FREE / "genes" / f"{c}.parquet")
    k = pd.read_parquet(CAN / "genes" / f"{c}.parquet").reindex(f.index)
    assert (f.hsapiens_gene_id.fillna("") == k.hsapiens_gene_id.fillna("")).all() and np.allclose(f.pi_raw, k.pi_raw)
    sup = (f.count_support & k.count_support).to_numpy(bool)
    seeds = [x for x in f.columns if x.startswith("alpha_s")]
    r = dict(cohort=c, tissue=f.tissue.iloc[0], n_genes=len(f), n_seeds_free=len(seeds), n_seeds_can=len([x for x in k.columns if x.startswith("alpha_s")]),
             share_support_free=float(f.count_support.mean()), share_support_can=float(k.count_support.mean()), share_support_both=float(sup.mean()))
    e = f.expr.to_numpy(float)
    for tag, g in (("free", f), ("can", k)):
        D, S = g.D.to_numpy(float), g.S_ema.to_numpy(float)
        r[f"rho_DS_{tag}"] = rho(D[sup], S[sup])
        r[f"rho_DS_partial_{tag}"] = H1REC.partial_rho(D[sup], S[sup], [e[sup]])
        r[f"rho_D_expr_{tag}"] = rho(D[sup], e[sup])
        r[f"rho_S_expr_{tag}"] = rho(S[sup], e[sup])
        r[f"rho_D_expr_all_{tag}"] = rho(D, e)
        r[f"sd_D_{tag}"] = float(np.std(D))
        r[f"sd_D_support_{tag}"] = float(np.std(D[sup]))
        r[f"tail_up_D_{tag}"], r[f"tail_dn_D_{tag}"] = float((D > H1R.TAIL).mean()), float((D < -H1R.TAIL).mean())
        Dr = g.D_raw.to_numpy(float)
        r[f"tail_up_Draw_{tag}"], r[f"tail_dn_Draw_{tag}"] = float((Dr > H1R.TAIL).mean()), float((Dr < -H1R.TAIL).mean())
        r[f"frac_S_pos_{tag}"] = float((S[sup] > 0).mean())
        r[f"n_strict_{tag}"], r[f"n_nameable_{tag}"] = int(g.strict_support.sum()), int(g.nameable.sum())
        r[f"n_saturated_{tag}"] = int(g.saturated_flag.sum())
    r["rho_Dfree_Dcan_all"] = rho(f.D, k.D)
    r["rho_Dfree_Dcan_support"] = rho(f.D[sup], k.D[sup])
    r["rho_Draw_free_can_all"] = rho(f.D_raw, k.D_raw)
    r["rho_Sfree_Scan_support"] = rho(f.S_ema[sup], k.S_ema[sup])
    r["rho_Dfree_Scan_support"] = rho(f.D[sup], k.S_ema[sup])
    r["rho_Dcan_Sfree_support"] = rho(k.D[sup], f.S_ema[sup])
    masked = ~f.count_support.to_numpy(bool)
    A = f[seeds].to_numpy(float)
    off = np.abs(A - f.pi_clamped.to_numpy(float)[:, None]).max(1)
    r["n_masked_free"] = int(masked.sum())
    r["masked_at_prior_free"] = float((off[masked] < 0.005).mean()) if masked.any() else np.nan
    r["max_abs_alpha_minus_prior_masked_free"] = float(off[masked].max()) if masked.any() else np.nan
    r["sd_D_masked_free"] = float(np.std(f.D.to_numpy(float)[masked])) if masked.any() else np.nan
    r["sd_D_masked_can"] = float(np.std(k.D.to_numpy(float)[masked])) if masked.any() else np.nan
    r["n_nameable_both"] = int((f.nameable & k.nameable).sum())
    r["sign_agree_nameable_both"] = float((np.sign(f.D[f.nameable & k.nameable]) == np.sign(k.D[f.nameable & k.nameable])).mean()) if (f.nameable & k.nameable).any() else np.nan
    CMP.mkdir(parents=True, exist_ok=True)
    keep = ["hsapiens_gene_id", "symbol", "pi_raw", "pi_clamped", "expr", "detection", "dNdS", "B1"]
    side = f[keep].copy()
    for col in ("alpha", "D", "D_raw", "S_ema", "count_support", "strict_support", "nameable", "saturated_flag"):
        side[f"{col}_free"], side[f"{col}_can"] = f[col], k[col]
    side["support_both"] = sup
    side.to_parquet(CMP / f"{c}.parquet")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    kept = json.loads((RU.OUT / "kept.json").read_text())
    reg = H1R.cohorts()
    ready = {c: v for c, v in kept.items() if len(v) >= 2 and c in reg}
    print("cohorts with >= 2 kept free seeds:", len(ready), "missing:", [c for c in reg if c not in ready], flush=True)
    (FREE / "genes").mkdir(parents=True, exist_ok=True)
    meta_f = FREE / "kept_used.json"
    used = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    todo = [(c, v) for c, v in ready.items() if a.force or used.get(c) != v or not (FREE / "genes" / f"{c}.parquet").exists()]
    with ProcessPoolExecutor(a.jobs) as ex:
        cks = list(ex.map(build, todo))
    for c, v in todo:
        used[c] = v
    meta_f.write_text(json.dumps(used, indent=1))
    ckf = FREE / "identity_checks.csv"
    old = pd.read_csv(ckf) if ckf.exists() else pd.DataFrame(columns=["cohort"])
    pd.concat([old[~old.cohort.isin([x["cohort"] for x in cks])], pd.DataFrame(cks)]).to_csv(ckf, index=False)
    order = [c for c in reg if c in ready]
    T = pd.DataFrame([compare(c) for c in order])
    T.to_csv(RU.OUT / "cohort_compare.csv", index=False)
    ax = H1REC.axis_table({c: reg[c] for c in order})
    ax.to_csv(RU.OUT / "axis_free.csv", index=False)
    can = pd.read_csv(CAN / "axis.csv")
    m = ax.merge(can, on=["cohort", "universe"], suffixes=("_free", "_can"))
    cols = ["cohort", "universe", "n_free", "rho_pi_dNdS_free"] + [f"{k}_{s}" for k in ("rho_D_pi", "rho_D_dNdS", "rhop_D_dNdS", "rhop_D_pi", "rho_S_pi", "rho_S_dNdS", "rho_D_expr") for s in ("free", "can")]
    m[[x for x in cols if x in m]].to_csv(RU.OUT / "axis_compare.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    show = ["cohort", "n_seeds_free", "share_support_both", "rho_DS_free", "rho_DS_can", "rho_DS_partial_free", "rho_DS_partial_can", "rho_Dfree_Dcan_support",
            "rho_Dfree_Dcan_all", "rho_Sfree_Scan_support", "masked_at_prior_free", "sd_D_free", "sd_D_can"]
    print(T[show].round(3).to_string())
    print(m[[x for x in cols if x in m]].round(3).to_string())


if __name__ == "__main__":
    main()
