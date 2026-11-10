"""EXPLORATORY, post hoc, written after F1 and F2 were seen. Enters no test and no BH family.

T4.3 showed pi x expression strata do not control cell-type dominance. Here F1 and the F2 node
permutation are repeated with a dominance flag added to the strata (liver: top quartile of X2
``hep_enrich``; testis: T2.5 commonest dominant type), r1 and r2, as in T4.3 dominance_exploratory.

    python analysis/tasks/T6_3_pathway/dominance_exploratory.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.T6_3_pathway.run import perm_stats, summarise
from analysis.tasks.T6_4_network import run as NET
from analysis.tasks.T6_common import functional as FN

RES = gene_ids.REPO_ROOT / "results/analysis"
OUT = RES / "T6_3/exploratory_dominance.csv"
N = 10000


def flag(tab: pd.DataFrame, cohort: str) -> np.ndarray:
    if cohort.startswith("liv"):
        g = pd.read_parquet(RES / f"X2/genes/{cohort}.parquet").set_index("var_name")["hep_enrich"]
        v = g.reindex(tab["var_name"]).to_numpy()
        return v >= np.nanquantile(g.to_numpy(), 0.75)
    g = pd.read_parquet(RES / f"T2_5/{cohort}__genes.parquet").set_index("var_name")["dominant_cell_type"]
    v = g.reindex(tab["var_name"])
    return (v == v.mode().iloc[0]).to_numpy()


def main() -> None:
    rows = []
    for ci, cohort in enumerate(FN.PRIMARY + FN.SECONDARY):
        tab, E = FN.gene_table(cohort)
        Rd = FN.residuals(tab, FN.designs(tab, E))
        f = flag(tab, cohort).astype(int)
        L = FN.libraries(tab)
        pooled = {f"{k}|{t}": v for k, d in L.items() for t, v in d.items()}
        mem, names = FN.membership(tab, pooled)
        lab = ST.strata(tab["prior_raw"].to_numpy(), tab["log_cp10k_spmean"].to_numpy()) * 2 + f
        cols = ["r1", "r2"]
        R = Rd[cols].to_numpy()
        Ms, Vs = perm_stats(R, mem, lab, N, 20261302 + ci)
        for q, c in enumerate(cols):
            r = summarise(R[:, q], mem, np.arange(len(names)), Ms[q], Vs[q])
            rows.append(dict(cohort=cohort, test="F1", graph="", readout=c, estimate=r["index"], null_mean=r["null_mean"],
                             null_q975=r["null_q975"], p_perm=r["p_index"], frac_flagged=float(f.mean())))
        for gname, edges in (("HuRI", FN.huri_edges()), ("STRING", FN.string_edges("9606", 400))):
            e = FN.graph_arrays(edges, tab["hsapiens_gene_id"])
            nodes = np.unique(e)
            pos = -np.ones(len(tab), dtype=np.int64)
            pos[nodes] = np.arange(len(nodes))
            u, v = pos[e[:, 0]], pos[e[:, 1]]
            n, m = len(nodes), len(e)
            A = sp.csr_matrix((np.ones(2 * m), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
            deg = np.asarray(A.sum(1)).ravel()
            Wr = (sp.diags(1 / deg) @ A).tocsr()
            X = R[nodes]
            X = X - X.mean(0, keepdims=True)
            tn = tab.iloc[nodes]
            ln = NET.node_strata(tn["prior_raw"].to_numpy(), tn["log_cp10k_spmean"].to_numpy(), deg) * 2 + f[nodes]
            I = NET.moran_all(X, Wr)
            P, _ = NET.perm_null(X, Wr, A.tocsr(), ln, N, 20261402 + ci)
            for q, c in enumerate(cols):
                rows.append(dict(cohort=cohort, test="F2", graph=gname, readout=c, estimate=float(I[q]),
                                 null_mean=float(P[:, q].mean()), null_q975=float(np.quantile(P[:, q], .975)),
                                 p_perm=float((1 + (P[:, q] >= I[q]).sum()) / (N + 1)), frac_flagged=float(f.mean())))
        print(cohort, "done", flush=True)
    df = pd.DataFrame(rows)
    K.write_csv(df, OUT)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
