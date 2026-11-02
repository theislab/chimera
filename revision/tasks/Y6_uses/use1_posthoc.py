"""Y6 use 1, POST HOC, written after reading the card's tests, committed before running. See ledgers/Y6_uses.md, "Post hoc".

PH1 count-only analogue of S_row. The top flags of S_row are highly expressed genes absent from one species (ribosomal proteins in lemur and pig), and the margin of a
    near-absent gene pays in proportion to the expected count. So: gap_abs = largest other-species CP10k - CP10k of s, gap_log = the same on log1p. No model.
PH2 retrocopy label, independent of counts: T_retro = species gene has at most 1 coding exon in its best transcript while the human gene has at least 3 (Ensembl 110).
    A single-exon "ortholog" of a multi-exon gene is the usual shape of a processed pseudogene taken as the ortholog.
PH3 structural zeros. Exact-zero genes per row against T_newasm; S and D of the zero genes in testis macaque fits; gold reading with and without them.
PH4 candidate list: even and odd genes, S_row beyond the row's odd-tuned threshold in U_hi, with counts and Ensembl reasons.

    PYTHONPATH=$PWD python -m revision.tasks.Y6_uses.use1_posthoc
"""

from __future__ import annotations

import gzip
import re
import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from revision.common import gene_ids
from revision.tasks.X4_where_differs import common as CM
from revision.tasks.X4_where_differs import rules as XR
from revision.tasks.Y6_uses import rules as RU
from revision.tasks.Y6_uses.annot_genome import GTF
from revision.tasks.Y6_uses.use1 import OUT, ap_setup, block_weights, tune, wap, wprf

CM.N_JOBS = 8
H5 = {"liv_4sp": gene_ids.h5ad_path("liv_4sp"), "test_3sp": gene_ids.h5ad_path("test_3sp"), "bone_marrow": gene_ids.h5ad_path("bone_marrow"),
      "vz_full": gene_ids.h5ad_path("van_zyl"), "test_hmac": gene_ids.h5ad_path("test_hmac")}
SP_KEY = ["species", "Species"]
HUMAN_GTF = RU.ROOT / "data/revision/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz"


def cds_exons(path) -> pd.Series:
    n = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 3)
            if p[2] != "CDS":
                continue
            m = re.search(r'gene_id "([^"]+)"', line)
            t = re.search(r'transcript_id "([^"]+)"', line)
            k = (m.group(1), t.group(1))
            n[k] = n.get(k, 0) + 1
    s = pd.Series(n)
    s.index = pd.MultiIndex.from_tuples(s.index)
    return s.groupby(level=0).max()


def cp10k(fit: str) -> pd.DataFrame:
    import anndata as ad

    a = ad.read_h5ad(H5[fit])
    key = next(k for k in SP_KEY if k in a.obs)
    sp_ = a.obs[key].astype(str).to_numpy()
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    out = {}
    for s in np.unique(sp_):
        c = np.asarray(X[sp_ == s].sum(0)).ravel()
        out[s] = 1e4 * c / max(c.sum(), 1)
    return pd.DataFrame(out, index=[str(v) for v in a.var_names])


def boot_row(d, label, universe, scores):
    u = d[d[label].notna() & (d.det_other >= RU.DET_EXPR)]
    if universe == "hi":
        u = u[u.identity_s >= RU.ID_HIGH]
    odd, even = u[u.split == "odd"], u[u.split == "even"]
    if even[label].sum() < RU.MIN_POS or odd[label].sum() < RU.MIN_POS:
        return []
    y = even[label].to_numpy(float)
    base = y.mean()
    set_, flag, pt = {}, {}, {}
    for s in scores:
        sc = even[s].fillna(even[s].min() - 1).to_numpy(float)
        set_[s] = ap_setup(sc)
        v, _ = tune(odd[s].fillna(odd[s].min() - 1).to_numpy(float), odd[label].to_numpy(float))
        flag[s] = sc >= v
        pt[s] = wap(y, np.ones(len(y)), set_[s])
    rng = np.random.default_rng(RU.SEED + zlib.crc32(f"{d.fit.iloc[0]}|{d.species_label.iloc[0]}|{label}|{universe}".encode()) % 100000)
    B = {s: [] for s in scores}
    for w in block_weights(even.block.to_numpy(), rng, RU.DRAWS_PR):
        for s in scores:
            B[s].append(wap(y, w, set_[s]))
    B = {s: np.array(v, float) for s, v in B.items()}
    out = []
    for s in scores:
        lo, hi = np.nanpercentile(B[s], [2.5, 97.5])
        dl = B[s] - B["gap_abs"]
        p, r, f = wprf(y, np.ones(len(y)), flag[s])
        out.append(dict(fit=d.fit.iloc[0], species_label=d.species_label.iloc[0], label=label, universe=universe, score=s, n=len(even), n_pos=int(y.sum()), base_rate=base, ap=pt[s],
                        ap_lift=pt[s] / base, ap_lo=lo, ap_hi=hi, n_flag=int(flag[s].sum()), precision=p, recall=r, f1=f, d_gap_abs=pt[s] - pt["gap_abs"],
                        d_lo=np.nanpercentile(dl, 2.5), d_hi=np.nanpercentile(dl, 97.5)))
    return out


def main():
    d = pd.read_parquet(OUT / "scored_rows.parquet")
    hum = cds_exons(HUMAN_GTF)
    d["ex_human"] = d.hsapiens_gene_id.map(hum)
    exs = {sp_: cds_exons(GTF[sp_][110]) for sp_ in d.species.unique()}
    d["ex_s"] = [exs[s].get(g, np.nan) for s, g in zip(d.species, d.gene_id)]
    d["T_retro"] = ((d.ex_s <= 1) & (d.ex_human >= 3)).astype(float).where(d.ex_s.notna() & d.ex_human.notna())
    print(d.groupby(["fit", "species_label"]).T_retro.mean().round(4), flush=True)
    rows = []
    jobs = []
    for fit in RU.JOINT:
        cp = cp10k(fit)
        print(fit, cp.columns.tolist(), flush=True)
        for lab in RU.FITS[fit]["species"]:
            m = d[(d.fit == fit) & (d.species_label == lab)].copy()
            c = cp.reindex(m.var_name)
            others = [x for x in c.columns if x != lab]
            m["cp_s"], m["cp_other"] = c[lab].to_numpy(), c[others].max(axis=1).to_numpy()
            m["gap_abs"] = m.cp_other - m.cp_s
            m["gap_log"] = np.log1p(m.cp_other) - np.log1p(m.cp_s)
            rows.append(m)
            for label in ["T_lowconf", "T_later", "T_newasm", "T_retro", "T_goc"]:
                for uni in ("expr", "hi"):
                    jobs.append((m, label, uni, ["S_row", "gap_abs", "gap_log", "zero_gap", "zero_s", "id_neg"]))
    res = Parallel(n_jobs=12)(delayed(boot_row)(*j) for j in jobs)
    pd.DataFrame([r for x in res for r in x]).to_csv(OUT / "posthoc_ap.csv", index=False)
    D = pd.concat(rows, ignore_index=True)
    D.to_parquet(OUT / "posthoc_rows.parquet")
    # PH3
    z = []
    for (fit, lab), x in d.groupby(["fit", "species_label"]):
        zero = x.det_s == 0
        z.append(dict(fit=fit, species_label=lab, genes=len(x), exact_zero=int(zero.sum()), zero_in_expr=int((zero & (x.det_other >= RU.DET_EXPR)).sum()),
                      T_newasm_zero=x[zero].T_newasm.mean(), T_newasm_nonzero=x[~zero].T_newasm.mean(), newasm_total=int(x.T_newasm.sum()), newasm_and_zero=int((x.T_newasm.eq(1) & zero).sum()),
                      T_retro_zero=x[zero].T_retro.mean(), T_retro_nonzero=x[~zero].T_retro.mean()))
    pd.DataFrame(z).to_csv(OUT / "posthoc_zero.csv", index=False)
    # Gold reading with and without the macaque zeros (testis H+Macaque; G_H4, C_pat + H4_nsp).
    rr = []
    base3 = pd.read_parquet(RU.RES / "X4/genes/test_3sp.parquet")
    for fit, lab in (("test_hmac", "Macaque"), ("test_3sp", "Macaque")):
        if fit == "test_3sp":
            g = base3.copy()
        else:
            h = pd.read_parquet(RU.RES / "H1/genes/test_hmac.parquet")
            h = h.assign(var_name=h.index).rename(columns={"S_ema": "S"})[["hsapiens_gene_id", "var_name", "S", "D"]]
            keep = ["hsapiens_gene_id", "G_H4", "H4_nsp", "block"] + XR.C_PAT
            g = h.merge(base3[keep], on="hsapiens_gene_id", how="inner")
        x = d[(d.fit == fit) & (d.species_label == lab)].set_index("var_name")
        g["mac_zero"] = g.var_name.map(x.det_s.eq(0))
        unmask = np.abs(g.S.to_numpy()) >= 1e-6
        covs = XR.C_PAT + ["H4_nsp"]
        for tag, mask in (("all_unmasked", unmask), ("macaque_counted", unmask & g.mac_zero.eq(False).to_numpy()), ("macaque_zero", unmask & g.mac_zero.eq(True).to_numpy())):
            rr += CM.rho_unit(g, "G_H4", covs, ["S", "D"], rows=mask, draws=1000, label=dict(fit=fit, set=tag))
        s0 = g[unmask & g.mac_zero.eq(True)]
        s1 = g[unmask & g.mac_zero.eq(False)]
        print(fit, "share of unmasked with macaque zero", round(len(s0) / unmask.sum(), 3), "median S", round(s0.S.median(), 4), round(s1.S.median(), 4), "median D", round(s0.D.median(), 4), round(s1.D.median(), 4), flush=True)
    pd.DataFrame(rr).to_csv(OUT / "posthoc_macaque_zero_gold.csv", index=False)


if __name__ == "__main__":
    main()
