"""E1 step 5: recurrent named genes across the eight-tissue thread, on both gates (definitions in rules.py).

Writes results/revision/E1/genes/{calls.parquet, recurrent.csv, recurrence_null.csv, replication.csv}.

    PYTHONPATH=$PWD python -m revision.tasks.E1_atlas.genes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from revision.tasks.E1_atlas import rules as RU
from revision.tasks.E1_atlas.atlas import cohort_table
from revision.tasks.H1_harmonise import rules as H1R
from revision.tasks.H1_harmonise.recurrence import inv_norm_rank, strata
from revision.tasks.X3_cross_cohort import annotations as AN

OUTG = RU.OUT / "genes"


def calls(c: str) -> pd.DataFrame:
    g = cohort_table(c)
    u = g[g.universe].dropna(subset=["D_can", "D_free", "expr", "pi_raw"]).copy()
    u["int_can"] = inv_norm_rank(u.D_can.to_numpy(float))
    u["int_free"] = inv_norm_rank(u.D_free.to_numpy(float))
    u["stratum"] = strata(u, True, "pi_raw")
    ok = ~u.saturated_flag & ~u.saturated_flag_free
    cc = np.where(u.nameable & ok & (u.int_can.abs() >= RU.INT_CALL), np.sign(u.int_can), 0)
    cf = np.where(u.nameable_free & ok & (u.int_free.abs() >= RU.INT_CALL), np.sign(u.int_free), 0)
    u["call_can"], u["call_free"] = cc.astype(int), cf.astype(int)
    u["call"] = np.where((cc == cf) & (cc != 0), cc, 0).astype(int)
    det = {k.lower(): k for k in u.columns if k.startswith("detection_")}
    hum = det.get("detection_human")
    partner = [v for k, v in det.items() if k != "detection_human"]
    u["det_human"] = u[hum] if hum else np.nan
    u["det_partner_min"] = u[partner].min(1) if partner else np.nan
    u["det_partner_max"] = u[partner].max(1) if partner else np.nan
    u["cohort"] = c
    u["tissue"] = H1R.cohorts()[c]["tissue"]
    return u[["cohort", "tissue", "hsapiens_gene_id", "symbol", "pi_raw", "dNdS", "expr", "tau", "D_can", "D_free", "S", "B1", "int_can", "int_free",
              "call_can", "call_free", "call", "stratum", "det_human", "det_partner_min", "det_partner_max", "saturated_flag", "saturated_flag_free"]]


def recurrent_count(C: pd.DataFrame, rmin: int) -> tuple[int, int]:
    up = (C == 1).sum(1)
    dn = (C == -1).sum(1)
    return int(((up >= rmin) & (dn == 0)).sum()), int(((dn >= rmin) & (up == 0)).sum())


def main() -> int:
    OUTG.mkdir(parents=True, exist_ok=True)
    allc = [calls(c) for c in H1R.cohorts()]
    K = pd.concat(allc, ignore_index=True)
    K.to_parquet(OUTG / "calls.parquet")
    prim = [RU.PRIMARY[t] for t in RU.TISSUES]
    th = K[K.cohort.isin(prim)]
    C = th.pivot(index="hsapiens_gene_id", columns="tissue", values="call").reindex(columns=RU.TISSUES)
    Cc = th.pivot(index="hsapiens_gene_id", columns="tissue", values="call_can").reindex(columns=RU.TISSUES)
    Cf = th.pivot(index="hsapiens_gene_id", columns="tissue", values="call_free").reindex(columns=RU.TISSUES)
    obs = {"agree": recurrent_count(C.fillna(0), RU.GENE_R_MIN), "canonical": recurrent_count(Cc.fillna(0), RU.GENE_R_MIN), "per-gene": recurrent_count(Cf.fillna(0), RU.GENE_R_MIN)}
    # null: permute calls within pi x expression strata in each tissue independently
    rng = np.random.default_rng(RU.SEED)
    nulls = []
    for b in range(RU.N_GENE_NULL):
        cols = {}
        for t, c in zip(RU.TISSUES, prim):
            x = th[th.cohort == c]
            perm = x.groupby("stratum").call.transform(lambda s: rng.permutation(s.to_numpy()))
            cols[t] = pd.Series(perm.to_numpy(), index=x.hsapiens_gene_id.to_numpy())
        N = pd.DataFrame(cols).fillna(0)
        nulls.append(recurrent_count(N, RU.GENE_R_MIN))
    nulls = np.array(nulls)
    rn = pd.DataFrame(dict(direction=["above", "below"], observed=obs["agree"], null_mean=nulls.mean(0), null_q95=np.quantile(nulls, 0.95, axis=0),
                           p=[(1 + (nulls[:, i] >= obs["agree"][i]).sum()) / (len(nulls) + 1) for i in range(2)],
                           observed_canonical_only=obs["canonical"], observed_per_gene_only=obs["per-gene"]))
    rn["fold"] = rn.observed / rn.null_mean
    rn.to_csv(OUTG / "recurrence_null.csv", index=False)
    print(rn.round(3).to_string())
    # recurrent genes with annotation
    up, dn = (C == 1).sum(1), (C == -1).sum(1)
    rec = C.index[((up >= RU.GENE_R_MIN) & (dn == 0)) | ((dn >= RU.GENE_R_MIN) & (up == 0))]
    ann = th.groupby("hsapiens_gene_id").agg(symbol=("symbol", "first"), pi_raw=("pi_raw", "mean"), dNdS=("dNdS", "first"), tau=("tau", "mean"), expr=("expr", "mean"),
                                             int_can=("int_can", "mean"), int_free=("int_free", "mean"))
    det = th.assign(det_gap=np.log2((th.det_human + 0.01) / (th.det_partner_max + 0.01))).pivot(index="hsapiens_gene_id", columns="tissue", values="det_gap")
    allK = K.groupby("hsapiens_gene_id").agg(n_cohorts_tested=("call", "size"), n_cohorts_up=("call", lambda s: int((s == 1).sum())), n_cohorts_down=("call", lambda s: int((s == -1).sum())))
    pi_terc = pd.qcut(th.groupby("hsapiens_gene_id").pi_raw.mean(), 3, labels=["fast", "middle", "conserved"])
    R = pd.DataFrame(index=rec)
    R["direction"] = np.where(up.reindex(rec) > 0, "above", "below")
    R["n_tissues"] = np.maximum(up, dn).reindex(rec)
    R["tissues"] = [";".join(t for t in RU.TISSUES if C.loc[k, t] != 0 and np.isfinite(C.loc[k, t])) for k in rec]
    R["n_tissues_tested"] = C.notna().sum(1).reindex(rec)
    R = R.join(ann).join(allK).join(det.add_prefix("det_log2_human_vs_partner_"))
    R["pi_tertile"] = pi_terc.reindex(rec).astype(str)
    R["is_mt"] = R.symbol.astype(str).str.startswith("MT-")
    cl = AN.gene_classes(pd.Index(rec))
    R["classes"] = [";".join(k for k in H1R_classes() if bool(cl.loc[g, k])) for g in rec]
    up_kw = AN.uniprot_by_gene().reindex(pd.Index(rec))
    kt = AN.keyword_table()
    R["keywords"] = [";".join(sorted(kt.loc[k, "kw_name"] for k in kws if k in kt.index and kt.loc[k, "kw_category"] in ("Cellular component", "Molecular function", "Domain"))) if isinstance(kws, set) else "" for kws in up_kw.kw]
    R = R.sort_values(["direction", "n_tissues", "int_can"], ascending=[True, False, False])
    R.index.name = "hsapiens_gene_id"
    R.to_csv(OUTG / "recurrent.csv")
    print(R.groupby(["direction", "pi_tertile"]).size())
    return 0


def H1R_classes():
    return H1R.CLASSES


if __name__ == "__main__":
    raise SystemExit(main())
