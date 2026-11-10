"""F4: conserved liver-TF binding B_g against r2, liv_hmouse, with test_hmouse as tissue
control on one gene universe (pre-specified analysis plan, section 7). Measures conservation, not coherence.

Bound in a species: max score (MACS2, -10 log10 Q; 0 = no peak) over the TF's liver-lineage
experiments >= 500, within +/-5 kb of the TSS. B_g = TFs bound in both / TFs bound in either; genes
bound in neither have no B_g. Statistic: Spearman(B_g, r2) in each tissue, difference liver - testis.
Null for each tissue's rho: permutation within pi x mean strata. Difference: paired bootstrap of
genes, 10,000 draws, on fixed ranks; ESM-module blocks beside.

    python analysis/tasks/T6_6_regulatory/run.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.common.outputs import git_sha
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.T6_6_regulatory.fix_panel import FUNC, OUT, mouse_name
from analysis.tasks.T6_common import functional as FN

THR = 500.0
N_BOOT = 10000
N_PERM = 10000
SEED = 20261102
T43_CACHE = gene_ids.REPO_ROOT / "results/analysis/T4_3/cache"


def bound(path, columns: str, sym_map: dict) -> set:
    cols = columns.split(";")
    t = pd.read_csv(path, sep="\t", usecols=["Target_genes"] + cols)
    t["max"] = t[cols].to_numpy(dtype=float).max(1)
    hit = t.loc[t["max"] >= THR, "Target_genes"]
    return {sym_map[s] for s in hit if s in sym_map}


def binding_table() -> pd.DataFrame:
    panel = json.loads((OUT / "panel.json").read_text())
    if panel["stop"]:
        raise SystemExit("under 3 TFs: F4 stops (card)")
    filt = pd.read_csv(OUT / "panel_filter.csv").set_index("tf")
    L = gene_ids.load_table("liv_hmouse")
    h = L.dropna(subset=["hsapiens_gene_name"]).drop_duplicates("hsapiens_gene_name", keep=False)
    hmap = dict(zip(h.hsapiens_gene_name, h.hsapiens_gene_id))
    m = L.dropna(subset=["mmusculus_gene_name"]).drop_duplicates("mmusculus_gene_name", keep=False)
    mmap = dict(zip(m.mmusculus_gene_name, m.hsapiens_gene_id))
    genes = sorted(L["hsapiens_gene_id"].dropna().unique())
    tab = pd.DataFrame(index=pd.Index(genes, name="hsapiens_gene_id"))
    for tf in panel["panel"]:
        tab[f"{tf}_h"] = tab.index.isin(bound(FUNC / "hg38" / f"{tf}.5.tsv", filt.loc[tf, "hg38_columns"], hmap))
        tab[f"{tf}_m"] = tab.index.isin(bound(FUNC / "mm10" / f"{mouse_name(tf)}.5.tsv", filt.loc[tf, "mm10_columns"], mmap))
    H = tab[[f"{t}_h" for t in panel["panel"]]].to_numpy()
    M = tab[[f"{t}_m" for t in panel["panel"]]].to_numpy()
    tab["n_both"] = (H & M).sum(1)
    tab["n_either"] = (H | M).sum(1)
    tab["B_g"] = np.where(tab.n_either > 0, tab.n_both / tab.n_either.replace(0, np.nan), np.nan)
    tab["n_human"] = H.sum(1)
    tab["n_mouse"] = M.sum(1)
    return tab


def boot_rho(rb: np.ndarray, ra: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Pearson of fixed ranks; weights: draws x genes."""
    W = weights
    sw = W.sum(1, keepdims=True)
    ma = (W * ra).sum(1, keepdims=True) / sw
    mb = (W * rb).sum(1, keepdims=True) / sw
    ca = (W * (ra - ma) * (rb - mb)).sum(1)
    va = (W * (ra - ma) ** 2).sum(1)
    vb = (W * (rb - mb) ** 2).sum(1)
    return ca / np.sqrt(va * vb)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    B = binding_table()
    B.to_csv(OUT / "binding_table.csv")
    liv, Eliv = FN.gene_table("liv_hmouse")
    tes, Etes = FN.gene_table("test_hmouse")
    Rl = FN.residuals(liv, FN.designs(liv, Eliv), {"rI": FN.integrated_D(liv)})
    Rt = FN.residuals(tes, FN.designs(tes, Etes))
    Rl.index = liv["hsapiens_gene_id"].to_numpy()
    Rt.index = tes["hsapiens_gene_id"].to_numpy()
    Bd = B.dropna(subset=["B_g"])
    U = Bd.index.intersection(Rl.index).intersection(Rt.index)
    info = dict(panel=[c[:-2] for c in B.columns if c.endswith("_h")], n_vocab=len(B), n_defined=len(Bd),
                n_liver=len(liv), n_testis=len(tes), n_universe=len(U),
                frac_B0=float((Bd.loc[U, "B_g"] == 0).mean()), frac_B1=float((Bd.loc[U, "B_g"] == 1).mean()),
                mean_B=float(Bd.loc[U, "B_g"].mean()),
                frac_vocab_no_binding=float(B["B_g"].isna().mean()))
    print(info, flush=True)
    ub = ST.midrank(Bd.loc[U, "B_g"].to_numpy())
    rows = []
    liv_i = liv.set_index("hsapiens_gene_id").loc[U]
    tes_i = tes.set_index("hsapiens_gene_id").loc[U]
    lab_l = ST.strata(liv_i["prior_raw"].to_numpy(), liv_i["log_cp10k_spmean"].to_numpy())
    lab_t = ST.strata(tes_i["prior_raw"].to_numpy(), tes_i["log_cp10k_spmean"].to_numpy())
    blocks = pd.read_parquet(T43_CACHE / "liv_hmouse__inputs.parquet").set_index("hsapiens_gene_id")["block"]
    blk = blocks.reindex(U).to_numpy()
    blk = pd.factorize(blk)[0]
    rng = np.random.default_rng(SEED)
    n = len(U)
    W = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(float)
    # block weights: draw modules with replacement
    WB = np.empty((N_BOOT, n))
    nb = blk.max() + 1
    for b in range(N_BOOT):
        WB[b] = rng.multinomial(nb, np.full(nb, 1 / nb))[blk]
    for ln, tn, tag in (("r2", "r2", "primary"), ("r1", "r1", "r1"), ("rI2", "r2", "integrated_seeds_liver")):
        rl = ST.midrank(Rl.loc[U, ln].to_numpy())
        rt = ST.midrank(Rt.loc[U, tn].to_numpy())
        rho_l = float(np.corrcoef(ub, rl)[0, 1])
        rho_t = float(np.corrcoef(ub, rt)[0, 1])
        d = rho_l - rho_t
        bl, bt = boot_rho(ub, rl, W), boot_rho(ub, rt, W)
        bd = bl - bt
        bdb = boot_rho(ub, rl, WB) - boot_rho(ub, rt, WB)
        nl = ST.perm_null(rl - rl.mean(), ub - ub.mean(), lab_l, N_PERM, np.random.default_rng(SEED + 1))
        nt = ST.perm_null(rt - rt.mean(), ub - ub.mean(), lab_t, N_PERM, np.random.default_rng(SEED + 2))
        rows.append(dict(version=tag, n=n, rho_liver=rho_l, rho_liver_lo=float(np.quantile(bl, .025)),
                         rho_liver_hi=float(np.quantile(bl, .975)), p_liver_perm=ST.p_one_sided(rho_l, nl),
                         null_liver_mean=float(nl.mean()), null_liver_q025=float(np.quantile(nl, .025)),
                         null_liver_q975=float(np.quantile(nl, .975)),
                         rho_testis=rho_t, rho_testis_lo=float(np.quantile(bt, .025)),
                         rho_testis_hi=float(np.quantile(bt, .975)), p_testis_perm=ST.p_one_sided(rho_t, nt),
                         diff=d, diff_lo=float(np.quantile(bd, .025)), diff_hi=float(np.quantile(bd, .975)),
                         p_diff=float((1 + (bd <= 0).sum()) / (N_BOOT + 1)),
                         diff_block_lo=float(np.quantile(bdb, .025)), diff_block_hi=float(np.quantile(bdb, .975)),
                         p_diff_block=float((1 + (bdb <= 0).sum()) / (N_BOOT + 1))))
    # B_g also residualized on the same design: B_g tracks expression, r2 does not (estimate only)
    Zl = FN.designs(liv.set_index("hsapiens_gene_id").loc[U].reset_index(), Eliv[liv["hsapiens_gene_id"].isin(U).to_numpy()][
        pd.Index(liv.loc[liv["hsapiens_gene_id"].isin(U), "hsapiens_gene_id"]).get_indexer(U)])
    ids_t = tes.loc[tes["hsapiens_gene_id"].isin(U), "hsapiens_gene_id"]
    Zt = FN.designs(tes.set_index("hsapiens_gene_id").loc[U].reset_index(),
                    Etes[tes["hsapiens_gene_id"].isin(U).to_numpy()][pd.Index(ids_t).get_indexer(U)])
    eb_l, eb_t = ST.resid(ub, Zl["Z2"]), ST.resid(ub, Zt["Z2"])
    rows.append(dict(version="B_g_also_residualised", n=n,
                     rho_liver=float(np.corrcoef(eb_l, ST.resid(ST.midrank(Rl.loc[U, "r2"].to_numpy()), Zl["Z2"]))[0, 1]),
                     rho_testis=float(np.corrcoef(eb_t, ST.resid(ST.midrank(Rt.loc[U, "r2"].to_numpy()), Zt["Z2"]))[0, 1])))
    df = pd.DataFrame(rows)
    K.write_csv(df, OUT / "F4_regulatory.csv")
    (OUT / "F4_info.json").write_text(json.dumps(info, indent=1))
    (OUT / "_provenance.json").write_text(json.dumps(dict(git_sha=git_sha(), n_boot=N_BOOT, n_perm=N_PERM, thr=THR)))
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print(df.T.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
