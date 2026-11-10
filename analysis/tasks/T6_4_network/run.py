"""T6.4: F2 (Moran's I on PPI graphs) and F3 (interolog retention), 05 section 7.

F2. I = x'Wx / x'x on graph nodes, x centred there (``morans_I`` of notebooks/_esm_geometry_builder.py,
sparse and for graphs of unequal degree). Row-standardised weights are primary, binary weights a
sensitivity (I = n / (2E) * x'Ax / x'x). Graphs: HuRI, STRING direct ``experiments`` >= 400, each with
all edges and after removing edges above the 99th percentile of ESM cosine over all gene pairs.
Two nulls, both one-sided: node permutation within pi x mean x degree strata (10,000 draws) and
999 Maslov-Sneppen rewirings (degree preserved, 10 x E swap attempts, each from the original).
The test p is the larger of the two.

F3. R_g = human partners retained in mouse / testable human partners, liv_hmouse. Human partners
from STRING direct >= 150; mouse edges from Skinnider 2021 liver (testable: both orthologs
quantified in a liver chromatogram), with the STRING mouse direct >= 150 fallback beside. Partial
on log degrees and pair identity. Measures conservation, not coherence. Estimates, no p claim.

    python analysis/tasks/T6_4_network/run.py [--only F2|F3] [--procs 6]
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.common.outputs import git_sha
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.T6_1_functional.xlsx_min import read_sheet
from analysis.tasks.T6_common import functional as FN

OUT = gene_ids.REPO_ROOT / "results/analysis/T6_4"
N_PERM = 10000
N_RW = 999
MIN_DEG_STRATUM = 5
MIN_TESTABLE = 5


@numba.njit(cache=False)
def rewire_null(eu, ev, n, X, nswap_factor, n_rw, seed):
    """Moran's I under degree-preserving rewiring. eu, ev: edges on 0..n-1; X: centred, n x q.
    Returns I_row and I_bin, n_rw x q. Adjacency held as an n x n byte matrix."""
    np.random.seed(seed)
    E = len(eu)
    q = X.shape[1]
    deg = np.zeros(n, dtype=np.int64)
    for k in range(E):
        deg[eu[k]] += 1
        deg[ev[k]] += 1
    den = np.zeros(q)
    for j in range(q):
        den[j] = (X[:, j] ** 2).sum()
    A = np.zeros((n, n), dtype=np.uint8)
    out_r = np.empty((n_rw, q))
    out_b = np.empty((n_rw, q))
    accepted = 0
    for b in range(n_rw):
        A[:, :] = 0
        u = eu.copy()
        v = ev.copy()
        for k in range(E):
            A[u[k], v[k]] = 1
            A[v[k], u[k]] = 1
        for _ in range(nswap_factor * E):
            i = np.random.randint(E)
            j = np.random.randint(E)
            if i == j:
                continue
            a, c = u[i], v[i]
            d, e = u[j], v[j]
            if np.random.random() < 0.5:
                d, e = e, d
            # a-c, d-e  ->  a-e, d-c
            if a == e or d == c or a == d or c == e:
                continue
            if A[a, e] or A[d, c]:
                continue
            A[a, c] = 0
            A[c, a] = 0
            A[d, e] = 0
            A[e, d] = 0
            A[a, e] = 1
            A[e, a] = 1
            A[d, c] = 1
            A[c, d] = 1
            u[i], v[i] = a, e
            u[j], v[j] = d, c
            accepted += 1
        for j in range(q):
            sr = 0.0
            sb = 0.0
            for k in range(E):
                xx = X[u[k], j] * X[v[k], j]
                sr += xx * (1.0 / deg[u[k]] + 1.0 / deg[v[k]])
                sb += 2.0 * xx
            out_r[b, j] = sr / den[j]
            out_b[b, j] = (n / (2.0 * E)) * sb / den[j]
    return out_r, out_b, accepted


def node_strata(pi, expr, deg) -> np.ndarray:
    base = ST.strata(pi, expr)
    lab = np.zeros(len(pi), dtype=np.int64)
    nxt = 0
    for b in np.unique(base):
        idx = np.flatnonzero(base == b)
        r = pd.Series(deg[idx]).rank(method="average")
        try:
            q = pd.qcut(r, 4, labels=False, duplicates="drop").to_numpy()
        except ValueError:
            q = np.zeros(len(idx), dtype=int)
        groups, cur = [], []
        for k in np.unique(q):
            cur.append(k)
            if np.isin(q, cur).sum() >= MIN_DEG_STRATUM:
                groups.append(cur)
                cur = []
        if cur:
            if groups:
                groups[-1] += cur
            else:
                groups.append(cur)
        for g in groups:
            lab[idx[np.isin(q, g)]] = nxt
            nxt += 1
    return lab


def moran_all(X: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
    return ((X * (W @ X)).sum(0)) / (X ** 2).sum(0)


def perm_null(X, Wr, Wb, lab, n_perm, seed, chunk=250):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(lab == s) for s in np.unique(lab)]
    n, q = X.shape
    den = (X ** 2).sum(0)
    out_r = np.empty((n_perm, q))
    out_b = np.empty((n_perm, q))
    done = 0
    while done < n_perm:
        c = min(chunk, n_perm - done)
        P = np.empty((c, n), dtype=np.int64)
        for idx in groups:
            P[:, idx] = idx[np.argsort(rng.random((c, len(idx))), axis=1)]
        for j in range(q):
            Y = X[:, j][P].T  # n x c
            out_r[done:done + c, j] = (Y * (Wr @ Y)).sum(0) / den[j]
            out_b[done:done + c, j] = (Y * (Wb @ Y)).sum(0) / den[j]
        done += c
    return out_r, out_b


def f2_job(job) -> list[dict]:
    cohort, graph, edge_set, seed = job
    tab, E = FN.gene_table(cohort)
    Z = FN.designs(tab, E)
    extra = {"rI": FN.integrated_D(tab)} if cohort == "liv_hmouse" else None
    Rd = FN.residuals(tab, Z, extra)
    cols = list(Rd.columns)
    edges = FN.huri_edges() if graph == "HuRI" else FN.string_edges("9606", 400)
    e = FN.graph_arrays(edges, tab["hsapiens_gene_id"])
    n_all = len(e)
    thr = np.nan
    if edge_set == "no_top1pct_esm":
        Ec = E - E.mean(0, keepdims=True)
        Ec = Ec / np.linalg.norm(Ec, axis=1, keepdims=True)
        Cs = Ec @ Ec.T
        thr = float(np.quantile(Cs[np.triu_indices(Cs.shape[0], 1)], 0.99))
        ce = (Ec[e[:, 0]] * Ec[e[:, 1]]).sum(1)
        e = e[ce <= thr]
        del Cs
    nodes = np.unique(e)
    pos = -np.ones(len(tab), dtype=np.int64)
    pos[nodes] = np.arange(len(nodes))
    u, v = pos[e[:, 0]], pos[e[:, 1]]
    n, m = len(nodes), len(e)
    A = sp.csr_matrix((np.ones(2 * m), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    Wr = sp.diags(1 / deg) @ A
    Wb = A * (n / (2.0 * m))
    X = Rd[cols].to_numpy()[nodes]
    X = X - X.mean(0, keepdims=True)
    tn = tab.iloc[nodes]
    lab = node_strata(tn["prior_raw"].to_numpy(), tn["log_cp10k_spmean"].to_numpy(), deg)
    I_r, I_b = moran_all(X, Wr.tocsr()), moran_all(X, Wb.tocsr())
    Pr, Pb = perm_null(X, Wr.tocsr(), Wb.tocsr(), lab, N_PERM, seed)
    Rr, Rb, acc = rewire_null(u.astype(np.int64), v.astype(np.int64), n, X.astype(np.float64), 10, N_RW, seed)
    rows = []
    for j, c in enumerate(cols):
        for wname, I, P, R in (("row", I_r, Pr, Rr), ("binary", I_b, Pb, Rb)):
            pp = float((1 + (P[:, j] >= I[j]).sum()) / (N_PERM + 1))
            pm = float((1 + (R[:, j] >= I[j]).sum()) / (N_RW + 1))
            rows.append(dict(cohort=cohort, graph=graph, edge_set=edge_set, weights=wname, readout=c,
                             n_nodes=n, n_edges=m, n_edges_before=n_all, esm_cosine_thr=thr,
                             n_strata=int(lab.max() + 1), I=float(I[j]),
                             perm_mean=float(P[:, j].mean()), perm_q975=float(np.quantile(P[:, j], 0.975)),
                             p_perm=pp, ms_mean=float(R[:, j].mean()), ms_q975=float(np.quantile(R[:, j], 0.975)),
                             p_ms=pm, p_larger=max(pp, pm),
                             ms_accept_per_edge=acc / (N_RW * m)))
    print(cohort, graph, edge_set, "n", n, "E", m, {c: round(float(I_r[j]), 4) for j, c in enumerate(cols)}, flush=True)
    return rows


def run_f2(procs: int) -> pd.DataFrame:
    jobs = []
    k = 0
    for cohort in FN.PRIMARY + FN.SECONDARY:
        for g in ("HuRI", "STRING"):
            for es in ("all", "no_top1pct_esm"):
                jobs.append((cohort, g, es, 20261102 + k))
                k += 1
    with ProcessPoolExecutor(procs) as ex:
        res = list(ex.map(f2_job, jobs))
    df = pd.DataFrame([r for x in res for r in x])
    K.write_csv(df, OUT / "F2_moran.csv")
    return df


# ---------------------------------------------------------------- F3
def skinnider_liver(tab: pd.DataFrame) -> tuple[pd.DataFrame, set, dict]:
    """Liver edges and quantified proteins, mapped to human ENSG through the liver 1:1 table."""
    F = FN.FUNC / "skinnider2021"
    cache = OUT / "skinnider_liver_quantified.json"
    if cache.exists():
        q = json.loads(cache.read_text())
    else:
        rows = read_sheet(F / "Skinnider2021_mmc1.xlsx", "Table S1")
        hdr = rows[0]
        it, ip = hdr.index("Tissue"), hdr.index("Protein")
        ir = [i for i, c in enumerate(hdr) if c.startswith("Ratio")]
        nq: dict[str, int] = {}
        for r in rows[1:]:
            if r[it] != "Liver":
                continue
            k = sum(1 for i in ir if i < len(r) and r[i] not in ("", "NA", "NaN"))
            nq[r[ip]] = max(nq.get(r[ip], 0), k)
        q = {"n_fractions": nq}
        OUT.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(q))
    # protein labels carry a replicate suffix (Aspg.6, Aspg.7)
    nq2: dict[str, int] = {}
    for p, k in q["n_fractions"].items():
        b = re.sub(r"\.\d+$", "", p)
        nq2[b] = max(nq2.get(b, 0), k)
    quant = {p for p, k in nq2.items() if k >= 1}
    L = gene_ids.load_table("liv_hmouse")
    sym2h = L.dropna(subset=["mmusculus_gene_name"]).drop_duplicates("mmusculus_gene_name", keep=False)
    sym2h = dict(zip(sym2h.mmusculus_gene_name, sym2h.hsapiens_gene_id))
    sk = read_sheet(F / "Skinnider2021_mmc2.xlsx", "Table S3")
    sk = pd.DataFrame(sk[1:], columns=sk[0])
    sk = sk[sk.Tissue == "Liver"]
    a, b = sk["Protein A"].map(sym2h), sk["Protein B"].map(sym2h)
    edges = pd.DataFrame({"a": a, "b": b}).dropna()
    # a protein on a liver edge was quantified by construction
    q_h = {sym2h[p] for p in quant if p in sym2h} | set(edges.a) | set(edges.b)
    info = dict(n_rows_liver=len(sk), n_proteins_table=len(nq2), n_quantified=len(quant),
                n_quantified_mapped=len(q_h), n_edges_mapped=len(edges))
    return edges, q_h, info


def f3_one(tab, rd: dict, ids, H: np.ndarray, Mo: np.ndarray, testable_nodes: np.ndarray, source: str, seed: int) -> list[dict]:
    n = len(ids)
    mset = set(map(tuple, Mo))
    keep = testable_nodes[H[:, 0]] & testable_nodes[H[:, 1]]
    Ht = H[keep]
    ret = np.array([tuple(e) in mset for e in Ht], dtype=bool)
    nt = np.bincount(Ht.ravel(), minlength=n).astype(float)
    nr = (np.bincount(Ht[ret, 0], minlength=n) + np.bincount(Ht[ret, 1], minlength=n)).astype(float)
    dh = np.bincount(H.ravel(), minlength=n).astype(float)
    dm = np.bincount(Mo.ravel(), minlength=n).astype(float)
    ok = nt >= MIN_TESTABLE
    L = gene_ids.load_table("liv_hmouse")
    t2 = L.drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")["mmusculus_identity"]
    ident = pd.Series(ids.to_numpy()).map(t2).to_numpy(dtype=float)
    ok &= np.isfinite(ident)
    if source == "STRING_mouse":
        ok &= dm >= 5  # A7 fallback: both endpoints with at least 5 mouse direct partners
    Rg = nr[ok] / nt[ok]
    cov = pd.DataFrame(dict(dh=np.log1p(dh[ok]), dm=np.log1p(dm[ok]), ident=ident[ok]))
    Z = K.design(cov, columns=["dh", "dm", "ident"])
    rows = []
    lab = (pd.qcut(pd.Series(dh[ok]).rank(method="first"), 4, labels=False).to_numpy() * 4
           + pd.qcut(pd.Series(dm[ok]).rank(method="first"), 4, labels=False).to_numpy())
    eR = ST.resid(ST.midrank(Rg), Z)
    for c, v in rd.items():
        ev = ST.resid(ST.midrank(v[ok]), Z)
        rho = float(np.corrcoef(eR, ev)[0, 1])
        rng = np.random.default_rng(seed)
        null = ST.perm_null(ev, eR, lab, 10000, rng)
        idx = np.arange(len(eR))
        boots = []
        for _ in range(2000):
            i = rng.choice(idx, len(idx))
            boots.append(np.corrcoef(eR[i], ev[i])[0, 1])
        rows.append(dict(source=source, readout=c, n_genes=int(ok.sum()), n_testable_edges=int(len(Ht)),
                         n_retained_edges=int(ret.sum()), frac_retained=float(ret.mean()),
                         frac_genes_R0=float((Rg == 0).mean()), rho_partial=rho,
                         ci_lo=float(np.quantile(boots, 0.025)), ci_hi=float(np.quantile(boots, 0.975)),
                         null_q025=float(np.quantile(null, 0.025)), null_q975=float(np.quantile(null, 0.975))))
    return rows


def run_f3() -> pd.DataFrame:
    cohort = "liv_hmouse"
    tab, E = FN.gene_table(cohort)
    Z = FN.designs(tab, E)
    Rd = FN.residuals(tab, Z, {"rI": FN.integrated_D(tab)})
    rd = {c: Rd[c].to_numpy() for c in ("r1", "r2", "rI1", "rI2")}
    ids = tab["hsapiens_gene_id"]
    H = FN.graph_arrays(FN.string_edges("9606", 150), ids)
    sk_e, q_h, info = skinnider_liver(tab)
    print(info, flush=True)
    Ms = FN.graph_arrays(sk_e, ids)
    quant = ids.isin(q_h).to_numpy()
    rows = f3_one(tab, rd, ids, H, Ms, quant, "Skinnider_liver", 20261102)
    # fallback: STRING mouse direct >= 150, through the 1:1 mouse IDs
    L = gene_ids.load_table(cohort).dropna(subset=["mmusculus_gene_id"])
    L = L.drop_duplicates("mmusculus_gene_id", keep=False).set_index("mmusculus_gene_id")["hsapiens_gene_id"]
    sm = FN.string_edges("10090", 150)
    sm = pd.DataFrame({"a": sm.a.map(L), "b": sm.b.map(L)}).dropna()
    Mg = FN.graph_arrays(sm, ids)
    has_m = np.zeros(len(ids), bool)
    has_m[np.unique(Mg)] = True
    rows += f3_one(tab, rd, ids, H, Mg, has_m, "STRING_mouse", 20261103)
    df = pd.DataFrame(rows)
    K.write_csv(df, OUT / "F3_interolog.csv")
    (OUT / "F3_info.json").write_text(json.dumps(info, indent=1))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["F2", "F3"], default=None)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 500)
    if a.only != "F3":
        f = run_f2(a.procs)
        print(f[(f.weights == "row") & f.readout.isin(["r1", "r2", "S2"])][
            ["cohort", "graph", "edge_set", "readout", "n_nodes", "n_edges", "I", "perm_q975", "p_perm",
             "ms_q975", "p_ms", "p_larger", "ms_accept_per_edge"]].round(4).to_string(index=False), flush=True)
    if a.only != "F2":
        print(run_f3().round(4).to_string(index=False), flush=True)
    (OUT / "_provenance.json").write_text(json.dumps(dict(git_sha=git_sha(), n_perm=N_PERM, n_rewire=N_RW)))


if __name__ == "__main__":
    main()
