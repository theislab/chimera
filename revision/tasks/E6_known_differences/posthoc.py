"""E6 post hoc (written after reading auroc.tsv; labelled post hoc in the ledger).

Q1. Beyond the level gap: does S_g (and per-gene Δα, B1) still place list genes below matched genes once the no-model
    detection gap LVL is matched too? Strata = expr quintile x tau quintile x pi tertile x LVL tertile, primary cohort,
    supported universe, every truth set with n >= 30 (core where it exists), and divergent vs conserved the same way
    (expr x tau x LVL tertiles).
Q2. Head-to-head: paired bootstrap of AUROC(S_can) - AUROC(B1) on the same genes and strata (tau strata), primary cohort.
Output: results/revision/E6/posthoc_beyond_level.csv, posthoc_s_vs_b1.csv
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import known as K  # noqa: E402
import rules as RU  # noqa: E402


def lvl_t(g):
    return pd.qcut(g.LVL.rank(method="first"), 3, labels=False).to_numpy()


def paired_diff(gu, mu, st, a_col, b_col, rng, n_boot=RU.N_BOOT):
    """AUROC(a) - AUROC(b), same strata and same resampled genes."""
    blocks = []
    for s in np.unique(st):
        ix = np.where(st == s)[0]
        m = mu[ix]
        if m.sum() >= 1 and (~m).sum() >= RU.MIN_STRATUM_BG:
            blocks.append((gu[a_col].to_numpy(float)[ix], gu[b_col].to_numpy(float)[ix], m))
    w = np.array([b[2].sum() for b in blocks], float)

    def auc(x, m):
        r = rankdata(x)
        n1, n0 = m.sum(), (~m).sum()
        return (r[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    obs = sum(wb * (auc(a, m) - auc(b, m)) for (a, b, m), wb in zip(blocks, w)) / w.sum()
    boots = np.empty(n_boot)
    for i in range(n_boot):
        acc = 0.0
        for (a, b, m), wb in zip(blocks, w):
            i1, i0 = np.where(m)[0], np.where(~m)[0]
            pick = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
            mm = np.zeros(len(pick), bool)
            mm[: len(i1)] = True
            acc += wb * (auc(a[pick], mm) - auc(b[pick], mm))
        boots[i] = acc / w.sum()
    return obs, np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def one(a):
    rng = np.random.default_rng(RU.SEED + 7 + zlib.crc32(a["assign_id"].encode()) % 1000)
    out1, out2 = [], []
    c = a["primary"]
    lst = K.add_core(K.map_symbols(K.read_list(a), K._CTX["approved"], K._CTX["other"], K._CTX["spmaps"]))
    g, _ = K.add_lvl(K.load_cohort(c), a["other_species"].split(","))
    gu = g[(g.supp_both & g.B1.notna() & g.LVL.notna()).to_numpy()].reset_index(drop=True)
    if len(gu) == 0:
        return out1, out2
    sets = [s for s in lst.set.unique() if not K.is_control(s)]
    cores = {s.replace("__core", "") for s in sets if s.endswith("__core")}
    sets = [s for s in sets if s not in cores]
    ctrl = set(lst[lst.set.map(K.is_control)].hsapiens_gene_id.dropna())
    st_l = K.strata(gu, True) * 10 + lvl_t(gu)
    st_t = K.strata(gu, True)
    for s in sets:
        ids = set(lst[lst.set == s].hsapiens_gene_id.dropna())
        mu = gu.hsapiens_gene_id.isin(ids).to_numpy()
        if mu.sum() < RU.MIN_LIST_PRED:
            continue
        base = dict(assign_id=a["assign_id"], circular=a["circular"], cohort=c, set=s)
        for sc in ["S_can", "D_free", "D_can", "B1"]:
            r = K.strat_auroc(gu[sc].to_numpy(float), mu, st_l, rng)
            out1.append(dict(base, comparison="vs_matched_LVL", score=sc, **r))
        if ctrl:
            gb = gu[gu.hsapiens_gene_id.isin(ids | ctrl).to_numpy()].reset_index(drop=True)
            mb = gb.hsapiens_gene_id.isin(ids - ctrl).to_numpy()
            if mb.sum() >= RU.MIN_LIST and (~mb).sum() >= RU.MIN_LIST:
                stb = (pd.qcut(gb.expr.rank(method="first"), 3, labels=False).to_numpy() * 100
                       + pd.qcut(gb.tau.rank(method="first"), 3, labels=False).to_numpy() * 10 + lvl_t(gb))
                old = RU.MIN_STRATUM_BG
                RU.MIN_STRATUM_BG = 1
                for sc in ["S_can", "D_free", "D_can", "B1"]:
                    out1.append(dict(base, comparison="vs_conserved_LVL", score=sc,
                                     **K.strat_auroc(gb[sc].to_numpy(float), mb, stb, rng)))
                RU.MIN_STRATUM_BG = old
        for other in ["B1", "LVL"]:
            d, lo, hi = paired_diff(gu, mu, st_t, "S_can", other, rng)
            out2.append(dict(base, contrast=f"S_can - {other}", diff=d, lo=lo, hi=hi))
    return out1, out2


def main():
    from multiprocessing import Pool
    assign = pd.read_csv(Path(__file__).resolve().parent / "assign.tsv", sep="\t", comment="#", dtype=str).fillna("")
    out1, out2 = [], []
    with Pool(18, initializer=K._init) as pool:
        for r1, r2 in pool.imap_unordered(one, [a.to_dict() for _, a in assign.iterrows()]):
            out1 += r1
            out2 += r2
            print("done", r1[0]["assign_id"] if r1 else (r2[0]["assign_id"] if r2 else "-"), flush=True)
    pd.DataFrame(out1).to_csv(RU.OUT / "posthoc_beyond_level.csv", index=False)
    pd.DataFrame(out2).to_csv(RU.OUT / "posthoc_s_vs_b1.csv", index=False)


if __name__ == "__main__":
    main()
