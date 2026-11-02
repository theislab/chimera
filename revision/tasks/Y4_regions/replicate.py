"""Y4 (EXPLORATORY): open even autosomes for the carried units only, after carried.json is committed to the ledger.

Sections, for each carried unit: replication (T1-T4 on even genes, BOOT_FINAL draws, also odd at the same draws);
sibling cohorts of the tissue; the other tissue; X holdout; sensitivities (Murat primates, D_raw, uncapped identity, leave-one-seed-
out); matched controls; paralog leak control; pattern and level terms; named margin-supported genes.
"""

from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from revision.tasks.T4_3_scoreboard import stats as ST
from revision.tasks.Y4_regions import rules as RU
from revision.tasks.Y4_regions import screen as SC

OUT = SC.OUT
LEDGER = SC.REPO.parent / "chimera-private/reviews/cell_system_review_1/response/ledgers/Y4_regions.md"
MARK = "CARRIED LIST COMMITTED BEFORE EVEN GENES"
RES = SC.REPO / "results/revision"


def carried() -> list[dict]:
    if MARK not in LEDGER.read_text():
        raise SystemExit("The carried list is not in the ledger yet.")
    return json.loads((OUT / "carried.json").read_text())


def mask_of(c: str, region: str):
    for r in SC.regs(c):
        if r["name"] == region:
            return r["mask"]
    return None


def cov_of(c: str, gold: str) -> list[str]:
    return RU.GOLDS[SC.tissue(c)][gold][1]


def _bundle(c, gold, region, split, draws, tag):
    m = mask_of(c, region)
    if m is None:
        return []
    res = SC.bundle(c, gold, cov_of(c, gold), region, m, split, draws)
    for r in res:
        r["section"] = tag
    return res


def twin_gold(gold: str, other: str) -> str:
    level = gold in ("G_H1", "G_H2")
    return {"liver": "G_H1" if level else "G_H3", "testis": "G_H2" if level else "G_H4"}[other]


def simple(c, rows, gold, covs, scores, draws=RU.BOOT_FINAL, label=None, seed_extra=""):
    t = SC.table(c)
    m = np.asarray(rows, bool) & t[[gold] + covs + scores].replace([np.inf, -np.inf], np.nan).notna().all(1).to_numpy()
    base = dict(label or {})
    base.update(cohort=c, gold=gold, n=int(m.sum()))
    if m.sum() < 30:
        return [dict(base, score=s, rho=np.nan) for s in scores]
    T = t.loc[m]
    R = SC.ranks(T, [gold] + scores)
    Z = SC.zdesign(T, covs)
    est = ST.rho_p_matrix(R, Z)[1:]
    rng = np.random.default_rng(SC.useed(c, gold, *scores, seed_extra, int(m.sum())))
    B = ST.bootstrap(R, Z, draws, rng, T["block"].to_numpy())[:, 1:]
    return [dict(base, score=s, rho=float(est[k]), lo=ST.pct_ci(B[:, k])[0], hi=ST.pct_ci(B[:, k])[1]) for k, s in enumerate(scores)]


def even_rows(t, m):
    return np.asarray(m, bool) & t["split"].eq("even").to_numpy()


def extra_cov(c: str) -> list[str]:
    add = ["logvar_total", "logvar_within", "spread_y", "peak_margin"]
    if not c.startswith("test"):
        add.append("hep_enrich_y")
    return add


def sensitivities(u: dict) -> list[dict]:
    c, g, x, reg = u["cohort"], u["gold"], u["readout"], u["region"]
    t = SC.table(c)
    m = even_rows(t, mask_of(c, reg))
    cv = cov_of(c, g)
    tag = dict(region=reg, unit=f"{c}|{g}|{x}|{reg}")
    out = []
    out += simple(c, m, g, cv, [x, "D_raw", "S"], label=dict(tag, what="base"))
    out += simple(c, m, g, cv + extra_cov(c), [x], label=dict(tag, what="given_variability_block"))
    out += simple(c, m, g, cv + ["id_true"], [x], label=dict(tag, what="given_uncapped_identity"))
    if g in RU.SECONDARY_GOLD:
        g2, cv2 = RU.SECONDARY_GOLD[g]
        out += simple(c, m, g2, cv2, [x, "B1b_raw"], label=dict(tag, what="Murat_primates"))
    leak = m & ~t["paralog_on_odd"].to_numpy()
    out += simple(c, leak, g, cv, [x], label=dict(tag, what="no_paralog_on_odd"))
    uns = m & ~t["masked"].to_numpy()
    out += simple(c, uns, g, cv, [x, "B1b_raw"], label=dict(tag, what="unmasked"))
    return out


def seed_range(u: dict) -> dict:
    """rho_p of each seed's readout on even genes of the region (D_raw and S per seed from X3)."""
    c, g, x, reg = u["cohort"], u["gold"], u["readout"], u["region"]
    t = SC.table(c)
    s = pd.read_parquet(RES / f"X3/seeds/{c}.parquet").set_index("var_name").reindex(t["var_name"])
    pre = "D_raw_s" if x == "D" else "S_s"
    cols = [k for k in s.columns if k.startswith(pre)]
    m = even_rows(t, mask_of(c, reg))
    cv = cov_of(c, g)
    m &= t[[g] + cv].notna().all(1).to_numpy()
    T = t.loc[m]
    Z = SC.zdesign(T, cv)
    rho = []
    for k in cols:
        R = np.column_stack([ST.midrank(T[g].to_numpy(float)), ST.midrank(s.loc[T["var_name"], k].to_numpy(float))])
        rho.append(float(ST.rho_p_matrix(R, Z)[1]))
    return dict(unit=f"{c}|{g}|{x}|{reg}", n_seeds=len(cols), seed_min=min(rho), seed_max=max(rho), seed_mean=float(np.mean(rho)),
                seeds_positive=int(sum(r > 0 for r in rho)))


def matched(u: dict, draws: int = 1000) -> list[dict]:
    """Region against nearest genes outside it on expression, detection, residual variance, spread, tau and pi (even genes)."""
    c, g, x, reg = u["cohort"], u["gold"], u["readout"], u["region"]
    t = SC.table(c)
    cv = cov_of(c, g)
    J = [b for b in u["J"].split(",") if b]
    scores = [x] + J + ["Sdir"]
    mm = ["log_cp10k_spmean", "detection_spmean", "hv_resid", "spread_y", "tau_shared", "prior_raw"]
    ok = t[[g] + cv + mm + scores].notna().all(1).to_numpy() & t["split"].eq("even").to_numpy()
    rm = mask_of(c, reg)
    fi, pi = np.flatnonzero(ok & rm), np.flatnonzero(ok & ~rm)
    if len(fi) < 30 or len(pi) < 200:
        return [dict(unit=f"{c}|{g}|{x}|{reg}", note="too few genes", n=len(fi))]
    U = np.column_stack([ST.midrank(t.loc[np.r_[fi, pi], k].to_numpy(float)) / (len(fi) + len(pi)) for k in mm])
    Uf, Up = U[: len(fi)], U[len(fi):]
    k = min(8, len(pi))
    _, nb = NearestNeighbors(n_neighbors=k).fit(Up).kneighbors(Uf)
    rng = np.random.default_rng(SC.useed("matched", c, g, x, reg))
    Tf = t.iloc[fi]
    Rf, Zf = SC.ranks(Tf, [g] + scores), SC.zdesign(Tf, cv)
    est_f = ST.rho_p_matrix(Rf, Zf)[1:]
    Bf, Bm = [], []
    bf = Tf["block"].to_numpy()
    ests_m = []
    for b in range(draws):
        pick = pi[nb[np.arange(len(fi)), rng.integers(0, k, len(fi))]]
        Tm = t.iloc[pick]
        Rm, Zm = SC.ranks(Tm, [g] + scores), SC.zdesign(Tm, cv)
        ests_m.append(ST.rho_p_matrix(Rm, Zm)[1:])
        wb = ST.block_weights(bf, rng)
        Bf.append(ST.weighted_rho(Rf, Zf, wb)[1:])
        Bm.append(ST.weighted_rho(Rm, Zm, ST.block_weights(Tm["block"].to_numpy(), rng))[1:])
    est_m = np.nanmean(ests_m, 0)
    Bf, Bm = np.array(Bf), np.array(Bm)
    out = []
    for j, s in enumerate(scores):
        dl, dh = ST.pct_ci(Bf[:, j] - Bm[:, j])
        out.append(dict(unit=f"{c}|{g}|{x}|{reg}", score=s, n=len(fi), rho_region=float(est_f[j]), rho_matched=float(est_m[j]),
                        diff=float(est_f[j] - est_m[j]), diff_lo=dl, diff_hi=dh))
    return out


def terms(u: dict) -> list[dict]:
    """Where the region sits on S, pattern gain, level cost, |level shift| against matched genes outside it (even and odd, residual on C)."""
    c, g, x, reg = u["cohort"], u["gold"], u["readout"], u["region"]
    t = SC.table(c)
    rm = mask_of(c, reg)
    out = []
    cols = ["S", "D", "Sdir", "Sdir_pattern", "Sdir_level", "abs_shift_y", "spread_y", "B1b_raw", g]
    for sp in ("odd", "even"):
        ok = t[RU.C].notna().all(1).to_numpy() & t["split"].eq(sp).to_numpy()
        T = t.loc[ok]
        Z = SC.zdesign(T, RU.C)
        for k in cols:
            d = T[k].notna().to_numpy()
            e = ST.resid(ST.midrank(T.loc[d, k].to_numpy(float)), Z[d])
            e = e / e.std()
            r = rm[ok][d]
            if r.sum() < 10:
                continue
            bl = T.loc[d, "block"].to_numpy()[r]
            ub, inv = np.unique(bl, return_inverse=True)
            su = np.bincount(inv, weights=e[r])
            cn = np.bincount(inv).astype(float)
            rng = np.random.default_rng(SC.useed("terms", c, reg, k, sp))
            pk = rng.integers(0, len(ub), (2000, len(ub)))
            boot = su[pk].sum(1) / np.maximum(cn[pk].sum(1), 1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            out.append(dict(unit=f"{c}|{g}|{x}|{reg}", split=sp, term=k, n=int(r.sum()), excess_sd=float(e[r].mean()), lo=float(lo), hi=float(hi)))
    return out


def named(u: dict) -> pd.DataFrame:
    """Margin-supported, S unmasked, counts in every species, seed-stable on D and S, gold and readout agree on the residual scale."""
    c, g, x, reg = u["cohort"], u["gold"], u["readout"], u["region"]
    t = SC.table(c)
    cv = cov_of(c, g)
    rm = mask_of(c, reg)
    s = pd.read_parquet(RES / f"X3/seeds/{c}.parquet").set_index("var_name").reindex(t["var_name"])
    Sg = s[[k for k in s.columns if k.startswith("S_s")]].to_numpy(float)
    Dg = s[[k for k in s.columns if k.startswith("D_raw_s")]].to_numpy(float)
    stable_s = (np.sign(Sg) == np.sign(t["S"].to_numpy())[:, None]).all(1)
    stable_d = (np.sign(Dg) == np.sign(t["D_raw"].to_numpy())[:, None]).all(1)
    ok = rm & t[[g] + cv].notna().all(1).to_numpy() & t["split"].isin(["odd", "even"]).to_numpy()
    T = t.loc[ok].copy()
    Z = SC.zdesign(T, cv)
    for k in (x, g):
        e = ST.resid(ST.midrank(T[k].to_numpy(float)), Z)
        T["r_" + k] = e / e.std()
    T["stable_S"], T["stable_D"] = stable_s[ok], stable_d[ok]
    T = T[T.margin_support.eq(True) & ~T.masked & T.nonzero_all_species & T.stable_S & T.stable_D]
    if c.startswith("liv"):
        T = T[~T.symbol.astype(str).str.startswith("MT-")]
    T = T[np.sign(T["r_" + x]) == np.sign(T["r_" + g])]
    T["score"] = np.abs(T["r_" + x]) + np.abs(T["r_" + g])
    y = [k for k in T.columns if k.startswith("y_")]
    keep = ["symbol", "hsapiens_gene_id", "chrom", "split", "prior_raw", "id_true", "log_cp10k_spmean", "tau_shared", "peak_y", "S", "D", "D_raw", "Sdir",
            "Sdir_pattern", "Sdir_level", "abs_shift_y", "shift_y", "B1b_raw", "B1c_raw", g, "r_" + x, "r_" + g, "score"] + y
    T = T.sort_values("score", ascending=False)[keep]
    T.insert(0, "unit", f"{c}|{g}|{x}|{reg}")
    return T


def main():
    car = carried()
    tasks = []
    for u in car:
        c, g, reg = u["cohort"], u["gold"], u["region"]
        tasks.append((c, g, reg, "even", RU.BOOT_FINAL, "replication"))
        tasks.append((c, g, reg, "odd", RU.BOOT_FINAL, "discovery_final"))
        tis = SC.tissue(c)
        for sib in RU.COHORTS:
            if SC.tissue(sib) == tis and sib != c:
                tasks.append((sib, g, reg, "even", RU.BOOT_SCREEN, "sibling"))
        other = "testis" if tis == "liver" else "liver"
        tasks.append((RU.PRIMARY[other], twin_gold(g, other), reg, "even", RU.BOOT_FINAL, "other_tissue"))
        tasks.append((c, g, reg, "X", RU.BOOT_SCREEN, "X_holdout"))
    tasks = sorted(set(tasks), key=str)
    res = Parallel(n_jobs=SC.N_JOBS, backend="loky")(delayed(_bundle)(*x) for x in tasks)
    df = pd.DataFrame([r for x in res for r in x])
    df.to_csv(OUT / "replication_units.csv", index=False)
    keys = {(u["cohort"], u["gold"], u["readout"], u["region"]): u for u in car}
    prim = df[df.section.eq("replication")].copy()
    prim = prim[[(a, b, r, g) in keys for a, b, r, g in zip(prim.cohort, prim.gold, prim.readout, prim.region)]]
    # one-sided p in the discovery direction (carried units are positive), BH over carried units
    from scipy.stats import norm
    prim["p_one"] = norm.sf(prim.rho / prim.se)
    prim["q_rep"] = ST.bh(prim.p_one.to_numpy().tolist())
    prim["replicates"] = (prim.rho > 0) & (prim.q_rep < RU.Q) & (prim.lo > 0)

    def call(r):
        J = [b for b in str(r.J).split(",") if b and b != "nan"]
        lo = [r.get(f"delta_{b}_lo") for b in J]
        hi = [r.get(f"delta_{b}_hi") for b in J]
        de = [r.get(f"delta_{b}") for b in J]
        if not J or any(pd.isna(v) for v in lo):
            return "no baseline on these genes"
        if min(lo) > 0 and min(de) >= RU.T3_MIN:
            return "differentiates"
        if min(hi) < 0:
            return "baseline ahead"
        return "matches"

    prim["call"] = prim.apply(call, axis=1)
    prim.to_csv(OUT / "replication_primary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 100)
    print(prim[["cohort", "gold", "readout", "region", "n", "rho", "lo", "hi", "q_rep", "replicates", "n_common", "rho_common", "T3", "T3_lo", "T3_hi", "T3_best", "T4", "call"]].round(3))
    sens = Parallel(n_jobs=SC.N_JOBS, backend="loky")(delayed(sensitivities)(u) for u in car)
    pd.DataFrame([r for x in sens for r in x]).to_csv(OUT / "replication_sensitivity.csv", index=False)
    seeds = Parallel(n_jobs=SC.N_JOBS, backend="loky")(delayed(seed_range)(u) for u in car)
    pd.DataFrame(seeds).to_csv(OUT / "replication_seeds.csv", index=False)
    mt = Parallel(n_jobs=SC.N_JOBS, backend="loky")(delayed(matched)(u) for u in car)
    pd.DataFrame([r for x in mt for r in x]).to_csv(OUT / "replication_matched.csv", index=False)
    tm = Parallel(n_jobs=SC.N_JOBS, backend="loky")(delayed(terms)(u) for u in car)
    pd.DataFrame([r for x in tm for r in x]).to_csv(OUT / "replication_terms.csv", index=False)
    pd.concat([named(u) for u in car]).to_csv(OUT / "named_genes.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
