"""E1 step 3: recurrence over the eight-tissue thread per read-out, the AGREE and STRICT layers, replication over 26 cohorts,
gate agreement on programmes (E-P1), tissue-matched programmes (E-P5), redundancy pruning for summaries.

Reads results/analysis/E1/{set_stats.parquet, cache/, sets/}. Writes results/analysis/E1/{recurrence.parquet, atlas.csv,
atlas_pruned.csv, gate_agreement.csv, tissue_programmes.csv, replication.csv}.

    PYTHONPATH=$PWD python -m analysis.tasks.E1_atlas.summarize
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.tasks.E1_atlas import rules as RU
from analysis.tasks.E1_atlas.atlas import CACHE, THREAD_READOUTS, tissue_programme_keys
from analysis.tasks.H1_harmonise.recurrence import bh

T = RU.TISSUES


def recurrence(st: pd.DataFrame, keys: list[str], readout: str) -> pd.DataFrame:
    prim = [RU.PRIMARY[t] for t in T]
    w = st[(st.readout == readout) & st.cohort.isin(prim)]
    Z = w.pivot(index="key", columns="cohort", values="z").reindex(index=keys, columns=prim)
    E = w.pivot(index="key", columns="cohort", values="effect").reindex(index=keys, columns=prim)
    Q = w.pivot(index="key", columns="cohort", values="q").reindex(index=keys, columns=prim)
    zo = Z.to_numpy(float)
    up = (np.nan_to_num(zo) >= RU.Z_CALL).sum(1)
    dn = (np.nan_to_num(zo) <= -RU.Z_CALL).sum(1)
    R = np.maximum(up, dn)
    S = np.stack([np.load(CACHE / f"{c}__{readout}.npz")["sign"] for c in prim], 2)  # draws x sets x tissues
    Rn = np.maximum((S == 1).sum(2), (S == -1).sum(2))
    p = (1 + (Rn >= R[None, :]).sum(0)) / (S.shape[0] + 1)
    lib = pd.Index(keys).str.split("|").str[0]
    q = pd.Series(p, index=keys).groupby(lib).transform(lambda s: pd.Series(bh(s.to_numpy()), index=s.index)).to_numpy()
    ntis = np.isfinite(zo).sum(1)
    out = pd.DataFrame(dict(key=keys, readout=readout, n_tissues=ntis, n_up=up, n_down=dn, R=R, p_rec=p, q_rec=q,
                            direction=np.where(up > dn, "above", np.where(dn > up, "below", "mixed")),
                            stouffer=np.nansum(zo, 1) / np.sqrt(np.maximum(ntis, 1)), mean_effect=np.nanmean(E.to_numpy(float), 1)))
    out["recurrent"] = (out.R >= RU.R_MIN) & (out.q_rec < 0.05)
    for j, t in enumerate(T):
        out[f"z_{t}"] = zo[:, j]
        out[f"e_{t}"] = E.to_numpy(float)[:, j]
        out[f"q_{t}"] = Q.to_numpy(float)[:, j]
    return out


def prune(df: pd.DataFrame, mem: pd.DataFrame, score: str) -> pd.DataFrame:
    """Keep a set unless a higher-ranked kept set shares Jaccard >= JACCARD_PRUNE (hallmark and class sets always kept)."""
    gs = mem.groupby("key").hsapiens_gene_id.apply(set).to_dict()
    kept, out = [], []
    for k in df.reindex(df[score].abs().sort_values(ascending=False).index).key:
        a = gs[k]
        red = None
        for k2 in kept:
            b = gs[k2]
            if len(a & b) / len(a | b) >= RU.JACCARD_PRUNE:
                red = k2
                break
        if red is None or k.split("|")[0] in ("hallmark", "class"):
            kept.append(k)
        out.append((k, red))
    return pd.DataFrame(out, columns=["key", "redundant_with"])


def tissue_programmes(st: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    prim = [RU.PRIMARY[t] for t in T]
    tp = tissue_programme_keys(keys)
    rows = []
    for r in ("D_can", "D_free", "S", "B1", "D_can_tau", "D_free_tau"):
        w = st[(st.readout == r) & st.cohort.isin(prim)]
        Z = w.pivot(index="key", columns="cohort", values="z").reindex(index=tp, columns=prim).to_numpy(float)
        E = w.pivot(index="key", columns="cohort", values="effect").reindex(index=tp, columns=prim).to_numpy(float)
        Q = w.pivot(index="key", columns="cohort", values="q").reindex(index=tp, columns=prim).to_numpy(float)
        ZN = np.stack([np.load(CACHE / f"{c}__{r}.npz")["zn_tp"] for c in prim], 2)  # draws x tp x tissues
        for own, ids in RU.TISSUE_PROGRAMMES.items():
            j = T.index(own)
            for x in ids:
                k = next((kk for kk in tp if kk.split("|")[1] == (x.split(":", 1)[1] if x.startswith("hallmark:") else x)), None)
                if k is None:
                    continue
                i = tp.index(k)
                others = np.delete(Z[i], j)
                cobs = Z[i, j] - np.nanmean(others)
                cn = ZN[:, i, j] - np.nanmean(np.delete(ZN[:, i, :], j, axis=1), axis=1)
                rank_own = int((np.nan_to_num(Z[i], nan=-99) > Z[i, j]).sum()) + 1
                rows.append(dict(readout=r, tissue=own, key=k, z_own=Z[i, j], effect_own=E[i, j], q_own=Q[i, j], mean_z_others=np.nanmean(others), contrast=cobs,
                                 p_contrast=(1 + (np.abs(cn) >= abs(cobs)).sum()) / (len(cn) + 1) if np.isfinite(cobs) else np.nan, rank_of_own_among_8=rank_own if np.isfinite(Z[i, j]) else np.nan,
                                 **{f"z_{t}": Z[i, jj] for jj, t in enumerate(T)}))
    out = pd.DataFrame(rows)
    out["q_contrast"] = out.groupby("readout").p_contrast.transform(lambda s: pd.Series(bh(s.to_numpy()), index=s.index))
    return out


def main() -> int:
    st = pd.read_parquet(RU.OUT / "set_stats.parquet")
    sets = pd.read_csv(RU.OUT / "sets" / "sets.csv").set_index("key")
    mem = pd.read_parquet(RU.OUT / "sets" / "membership.parquet")[["key", "hsapiens_gene_id"]]
    keys = sorted(st.key.unique())
    rec = pd.concat([recurrence(st, keys, r) for r in THREAD_READOUTS], ignore_index=True)
    rec.to_parquet(RU.OUT / "recurrence.parquet")
    piv = {r: rec[rec.readout == r].set_index("key") for r in THREAD_READOUTS}
    A = piv["D_can"][["n_tissues", "direction", "R", "q_rec", "recurrent", "stouffer", "mean_effect"] + [f"{p}_{t}" for t in T for p in ("z", "e", "q")]].add_prefix("can_")
    A = A.join(piv["D_free"][["direction", "R", "q_rec", "recurrent", "stouffer", "mean_effect"] + [f"{p}_{t}" for t in T for p in ("z", "e", "q")]].add_prefix("free_"))
    for r in ("S", "B1", "pi", "dNdS", "D_can_tau", "D_free_tau", "D_can_block", "D_free_block"):
        A = A.join(piv[r][["direction", "R", "recurrent", "stouffer"]].add_prefix(f"{r}_"))
    A["agree"] = A.can_recurrent & A.free_recurrent & (A.can_direction == A.free_direction) & (A.can_direction != "mixed")
    A["strict"] = A.agree & A.D_can_tau_recurrent & A.D_free_tau_recurrent & A.D_can_block_recurrent & A.D_free_block_recurrent & \
        (A.D_can_tau_direction == A.can_direction) & (A.D_free_tau_direction == A.can_direction) & (A.D_can_block_direction == A.can_direction) & (A.D_free_block_direction == A.can_direction)
    A["joint_stouffer"] = (A.can_stouffer + A.free_stouffer) / 2
    A = A.join(sets[["library", "set_id", "name", "n_vocab", "reactome_top"]])
    # replication over all 26 cohorts
    rep = []
    for r in ("D_can", "D_free"):
        w = st[st.readout == r]
        g = w.assign(up=w.z >= RU.Z_CALL, dn=w.z <= -RU.Z_CALL, tested=w.z.notna()).groupby("key")[["up", "dn", "tested"]].sum().add_prefix(f"{r}_all_")
        rep.append(g)
    A = A.join(pd.concat(rep, axis=1))
    pr = prune(A.reset_index().rename(columns={"index": "key"}).assign(s=lambda d: d.joint_stouffer.fillna(0))[["key", "s"]], mem, "s").set_index("key")
    A = A.join(pr)
    A.index.name = "key"
    A.to_csv(RU.OUT / "atlas.csv")
    A[A.redundant_with.isna()].to_csv(RU.OUT / "atlas_pruned.csv")
    # gate agreement on programmes, per cohort
    ga = []
    for c, w in st[st.readout.isin(["D_can", "D_free"])].groupby("cohort"):
        p = w.pivot(index="key", columns="readout", values="z").dropna()
        e = w.pivot(index="key", columns="readout", values="effect").dropna()
        both = ((w[w.readout == "D_can"].set_index("key").q < 0.05) & (w[w.readout == "D_free"].set_index("key").q < 0.05))
        sgn = np.sign(p.D_can) == np.sign(p.D_free)
        sig = both.reindex(p.index).fillna(False)
        ga.append(dict(cohort=c, thread=bool(w.thread.iloc[0]), n_sets=len(p), rho_z=spearmanr(p.D_can, p.D_free)[0], rho_effect=spearmanr(e.D_can, e.D_free)[0],
                       n_sig_can=int((w[w.readout == "D_can"].q < 0.05).sum()), n_sig_free=int((w[w.readout == "D_free"].q < 0.05).sum()),
                       n_sig_both=int(sig.sum()), sign_agree_sig_both=float(sgn[sig].mean()) if sig.any() else np.nan))
    pd.DataFrame(ga).to_csv(RU.OUT / "gate_agreement.csv", index=False)
    tissue_programmes(st, keys).to_csv(RU.OUT / "tissue_programmes.csv", index=False)
    # headline counts
    for r in THREAD_READOUTS:
        x = piv[r]
        print(r, "recurrent:", int(x.recurrent.sum()), "by library:", x[x.recurrent].index.str.split("|").str[0].value_counts().to_dict())
    print("AGREE", int(A.agree.sum()), "STRICT", int(A.strict.sum()), A[A.agree].library.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
