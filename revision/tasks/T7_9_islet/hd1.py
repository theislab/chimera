"""T7.9: H-D1, H-D1b, controls and sensitivities for the islet T2D case (plan section 8b, amendments).

Outcome per DE-able type t: Q = sign(l^h) * m. l^h is the unshrunk human log2FC (``log2fc_mle``,
``human_t2d_vs_nd_<t>``); m is the adaptive apeGLM mouse joint log2FC (``log2fc_shrunk``,
``mouse_joint_<t>``, ~ study + condition) as section 8b and its DE amendment say; SE(m) =
``lfcse_shrunk``. Genes: islet vocabulary (testis 1:1 table, the vocabulary of D) detected
(``det_frac``) in >= 5% of the type's cells in both species' DE units. Readout D from
``readout.py`` (seeds passing the health gate); C from the healthy training object.

rho_p, one-sided null and CI: the T7.3 implementation (``T7_3_mash.hd1.rho_row``): ranks of Q and
D residualised on C + splines of rank |l^h| and rank SE(m), Pearson; D residuals permuted within
pi_raw x expression strata; block bootstrap on the ESM modules; tertile test inside.

Pass rule (8b): (i) beta rho_p > 0 and q < 0.05 (BH over DE-able types); (ii) >= ceil(n/2) DE-able
types with rho_p > 0 and q < 0.05; (iii) over types with >= 150 human-DE genes (FDR < 0.1) the mean
top-minus-bottom D-tertile sign agreement >= 0.10, permutation p < 0.05 (one permutation of D within
strata of the whole vocabulary, applied to every type; P13 amendment (b)). Not testable: beta has
< 1,000 genes, or fewer than 3 types have >= 1,000 genes.

Beside D (Family 3): S_ema, D_raw, B1b. H-D1b: T7.3 ``nested`` (pi + expression, + B1b and B2b, + D).

Controls (8b): human condition labels permuted within sex x chemistry (``de/perm/human_<i>_<t>``,
l^h = MLE, m the true mouse m), 20 draws, every type. Mouse: the 11 within-study relabellings of the
joint fit (``de/perm/mouse_<i>_<t>``); these fits carry only the fixed-scale prior, so they are read
with m = ``log2fc_fixed`` and compared with the true-label fit read the same way. Draws whose mSTZ
labels are the true ones are the within-db/db relabellings of 8b (5 besides the true one).

Sensitivities: db/db-only and mSTZ-only m (per-model fits ``mouse_{dbdb,mstz}_<t>``, adaptive, and
fixed-scale beside), fixed-scale joint m, leave-one-out fits in ``de/loo`` (ND DE donors, mouse
libraries).

Output ``results/revision/T7_9/hd1/``, including ``hd1_per_type.csv`` (tidy, for the disease figure).

    PYTHONPATH=<chimera repo> python revision/tasks/T7_9_islet/hd1.py [--n-perm 10000] [--n-boot 10000]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.common import gene_ids
from revision.tasks.T4_3_scoreboard import stats as S
from revision.tasks.T7_3_mash import hd1 as H

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "revision" / "T7_9"
DE = RES / "de"
OUT = RES / "hd1"
GENES = Path(__import__("os").environ.get("T79_GENES", str(RES / "readout" / "islet_genes.parquet")))
DET = 0.05
FDR = 0.1
SEED = 20261102
N_CTRL = 20
C_COLS = H.C_COLS
NUIS = H.NUIS
PRIMARY_TYPE = "beta"
ADAPTIVE = ("log2fc_shrunk", "lfcse_shrunk")
FIXED = ("log2fc_fixed", "lfcse_fixed")


def read_de(path) -> pd.DataFrame:
    r = pd.read_csv(path, sep="\t")
    r = r.dropna(subset=["log2fc_mle", "log2fc_fixed", "gene_id"]).drop_duplicates("gene_id", keep=False)
    return r.set_index("gene_id")


def hfile(t: str) -> Path:
    return DE / f"human_t2d_vs_nd_{t}.tsv"


def mfile(t: str, contrast: str = "joint") -> Path:
    return DE / f"mouse_{contrast}_{t}.tsv"


def universe(tab: pd.DataFrame, hum: pd.DataFrame, mus: pd.DataFrame, mcols=ADAPTIVE) -> pd.DataFrame:
    d = tab.copy()
    d["l"] = hum["log2fc_mle"].reindex(d["hsapiens_gene_id"]).to_numpy()
    d["padj_h"] = hum["padj"].reindex(d["hsapiens_gene_id"]).to_numpy()
    d["det_h"] = hum["det_frac"].reindex(d["hsapiens_gene_id"]).to_numpy()
    d["m"] = mus[mcols[0]].reindex(d["mouse_id"]).to_numpy()
    d["se_m"] = mus[mcols[1]].reindex(d["mouse_id"]).to_numpy()
    d["padj_m"] = mus["padj"].reindex(d["mouse_id"]).to_numpy()
    d["det_m"] = mus["det_frac"].reindex(d["mouse_id"]).to_numpy()
    d = d[(d["det_h"] >= DET) & (d["det_m"] >= DET)]
    d = d.dropna(subset=["l", "m", "se_m", "D"] + C_COLS).copy()
    d["abs_l"] = d["l"].abs()
    d["Q"] = np.sign(d["l"]) * d["m"]
    d["agree"] = (np.sign(d["l"]) == np.sign(d["m"])).astype(float)
    return d.reset_index(drop=True)


def quick(u: pd.DataFrame) -> dict:
    Z = S.design(u, C_COLS + NUIS)
    Rm = np.column_stack([S.midrank(u["Q"]), S.midrank(u["D"])])
    de = (u["padj_h"] < FDR).to_numpy()
    t = float(H.tertile_stat(u["D"].to_numpy()[de], u["agree"].to_numpy()[de])[0]) if de.sum() >= 30 else np.nan
    return dict(n=len(u), rho_p_D=float(S.rho_p_matrix(Rm, Z)[1]), spearman_l_m=float(spearmanr(u["l"], u["m"])[0]),
                sign_agree=float(u["agree"].mean()), n_human_de=int(de.sum()), tert_diff=t)


_TAB = None


def _init(tab):
    global _TAB
    _TAB = tab


def _primary(args):
    t, n_perm, n_boot, seed = args
    rng = np.random.default_rng(seed)
    u = universe(_TAB, read_de(hfile(t)), read_de(mfile(t)))
    rows = []
    scores = ["D", "S_ema", "D_raw", "B1b"] + [c for c in ("D_unflagged", "D_all_nonnan") if c in _TAB]
    for score in scores:
        r = H.rho_row(u, score, n_perm if score == "D" else min(n_perm, 2000),
                      n_boot if score == "D" else min(n_boot, 2000), rng, with_tertile=(score == "D"))
        rows.append(dict(cell_type=t, **r))
    for r in H.pi_row(u):
        rows.append(dict(cell_type=t, **r))
    nest = H.nested(u, min(n_perm, 2000), min(n_boot, 2000), rng, variant=t)
    an = H.anchor(t, u)
    genes = u[["var_name", "hsapiens_gene_id", "symbol", "l", "m", "se_m", "padj_h", "padj_m", "Q", "D", "agree"]].assign(cell_type=t)
    return t, rows, nest, an, genes


def _quick_pair(args):
    t, kind, hf, mf, mcols = args
    u = universe(_TAB, read_de(hf), read_de(mf), mcols)
    return dict(cell_type=t, kind=kind, human=hf.name, mouse=mf.name, m_column=mcols[0], **quick(u))


def joint_tertile(tab: pd.DataFrame, types: list, n_perm: int, rng) -> dict:
    """Mean over types with >= 150 human-DE genes of the tertile statistic; D permuted within
    pi_raw x expression strata of the whole vocabulary, the same draw for every type."""
    ok = tab.dropna(subset=["D"] + C_COLS).reset_index(drop=True)
    lab = S.strata(ok["prior_raw"].to_numpy(), ok["log_cp10k_spmean"].to_numpy())
    pos = pd.Series(np.arange(len(ok)), index=ok["var_name"])
    parts = []
    for t in types:
        u = universe(ok, read_de(hfile(t)), read_de(mfile(t)))
        de = (u["padj_h"] < FDR).to_numpy()
        if de.sum() >= 150:
            parts.append((t, pos.loc[u.loc[de, "var_name"]].to_numpy(), u.loc[de, "agree"].to_numpy()))
    if not parts:
        return dict(n_types=0)
    D = ok["D"].to_numpy()
    obs = float(np.mean([H.tertile_stat(D[i], ag)[0] for _, i, ag in parts]))
    cnt, tot = 0, 0
    for P in H.perm_indices(lab, n_perm, rng, chunk=200):
        Dp = D[P]
        m = np.mean([H.tertile_stat(Dp[:, i], ag) for _, i, ag in parts], axis=0)
        cnt += int((m >= obs).sum())
        tot += len(m)
    return dict(n_types=len(parts), types=[p[0] for p in parts], mean_tert_diff=obs, p=(1 + cnt) / (1 + tot), n_perm=tot)


def mouse_draw_kind() -> dict:
    """Mouse permuted draw -> 'within_dbdb' if its mSTZ labels are the true ones, else 'both_studies'."""
    lab = pd.read_csv(DE / "perm" / "labels_mouse.tsv", sep="\t", index_col="draw")
    true = {"mSTZ__STZ__G2": "disease", "mSTZ__control__G1": "control"}
    return {int(i): ("within_dbdb" if all(r[k] == v for k, v in true.items()) else "both_studies")
            for i, r in lab.iterrows()}


def sensitivity_jobs(types: list) -> list:
    jobs = []
    kinds = mouse_draw_kind()
    for t in types:
        jobs.append((t, "primary|joint_fixed", hfile(t), mfile(t), FIXED))
        for c in ("dbdb", "mstz"):
            if mfile(t, c).exists():
                jobs.append((t, f"primary|{c}", hfile(t), mfile(t, c), ADAPTIVE))
                jobs.append((t, f"primary|{c}_fixed", hfile(t), mfile(t, c), FIXED))
        for i in range(N_CTRL):
            f = DE / "perm" / f"human_{i:02d}_{t}.tsv"
            if f.exists():
                jobs.append((t, "control_human_permuted", f, mfile(t), ADAPTIVE))
        for i, k in kinds.items():
            f = DE / "perm" / f"mouse_{i:02d}_{t}.tsv"
            if f.exists():
                jobs.append((t, f"control_mouse_permuted_{k}", hfile(t), f, FIXED))
        loo = DE / "loo" / f"loo_{t}.tsv"
        if loo.exists():
            for _, r in pd.read_csv(loo, sep="\t").iterrows():
                if not isinstance(r["file"], str) or not r["file"]:
                    continue
                f = DE / "loo" / r["file"]
                if r["species"] == "human":
                    jobs.append((t, f"loo_human|{r['left_out']}", f, mfile(t), ADAPTIVE))
                else:
                    jobs.append((t, f"loo_mouse|{r['left_out']}", hfile(t), f, ADAPTIVE))
    return jobs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--plumbing-test", default=None,
                   help="test only: output dir; D, S_ema, D_raw, B1b, B2b replaced by random noise")
    a = p.parse_args()
    global OUT
    tab = pd.read_parquet(GENES)
    meta = json.loads((GENES.parent / "readout_meta.json").read_text())
    if a.plumbing_test:
        OUT = Path(a.plumbing_test)
        rng0 = np.random.default_rng(0)
        for c in ("D", "S_ema", "D_raw", "D_unflagged", "D_all_nonnan", "B1b", "B2b"):
            if c in tab:
                tab[c] = rng0.normal(size=len(tab))
    elif len(meta["seeds"]) < 2:
        raise AssertionError(f"D needs at least 2 health-gate seeds, got {meta['seeds']}")
    OUT.mkdir(parents=True, exist_ok=True)
    types = json.loads((DE / "de_meta.json").read_text())["de_types"]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init, initargs=(tab,)) as ex:
        res = list(ex.map(_primary, [(t, a.n_perm, a.n_boot, SEED + i) for i, t in enumerate(types)]))
        sens = list(ex.map(_quick_pair, sensitivity_jobs(types)))
    rows = [r for _, rr, _, _, _ in res for r in rr]
    h1 = pd.DataFrame(rows)
    dsel = h1[h1["score"] == "D"].copy()
    dsel["q_bh"] = S.bh(dsel["p_one_sided"].tolist())
    h1 = h1.merge(dsel[["cell_type", "score", "q_bh"]], on=["cell_type", "score"], how="left")
    h1.to_csv(OUT / "h1.tsv", sep="\t", index=False)
    pd.concat([n for _, _, n, _, _ in res]).to_csv(OUT / "h1b.tsv", sep="\t", index=False)
    pd.DataFrame([an for _, _, _, an, _ in res]).to_csv(OUT / "h1_anchor.tsv", sep="\t", index=False)
    pd.concat([g for *_, g in res]).to_csv(OUT / "h1_genes.tsv", sep="\t", index=False)
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "h1_sensitivity_controls.tsv", sep="\t", index=False)

    jt = joint_tertile(tab, types, min(a.n_perm, 10000), np.random.default_rng(SEED + 99))
    n_ok = dsel.set_index("cell_type")["n"]
    sig = (dsel["rho_p"] > 0) & (dsel["q_bh"] < 0.05)
    testable = bool(n_ok.get(PRIMARY_TYPE, 0) >= 1000 and (n_ok >= 1000).sum() >= 3)
    pt = dsel[dsel.cell_type == PRIMARY_TYPE].iloc[0]
    rule = dict(
        testable=testable, n_types=len(types), n_types_ge_1000_genes=int((n_ok >= 1000).sum()),
        seeds=meta["seeds"], plumbing_test=bool(a.plumbing_test),
        i_beta=bool(pt["rho_p"] > 0 and pt["q_bh"] < 0.05), beta_rho_p=float(pt["rho_p"]), beta_q=float(pt["q_bh"]),
        ii_n_sig=int(sig.sum()), ii_needed=int(math.ceil(len(types) / 2)),
        ii=bool(sig.sum() >= math.ceil(len(types) / 2)),
        iii=jt, iii_pass=bool(jt.get("n_types", 0) > 0 and jt["mean_tert_diff"] >= 0.10 and jt["p"] < 0.05),
    )
    rule["pass"] = bool(testable and rule["i_beta"] and rule["ii"] and rule["iii_pass"])
    ctrl = sens[sens["kind"].str.startswith("control")].groupby(["cell_type", "kind"])["rho_p_D"].agg(
        ["count", "mean", "std", "min", "max"]).reset_index()
    ctrl.to_csv(OUT / "h1_controls_summary.tsv", sep="\t", index=False)
    (OUT / "h1_rule.json").write_text(json.dumps(rule, indent=2, default=float))

    # tidy per-type table for the disease figure: label null = range of the human-permuted controls
    hc = ctrl[ctrl["kind"] == "control_human_permuted"].set_index("cell_type")
    tidy = pd.DataFrame(dict(
        tissue="islet", cell_type=dsel["cell_type"].to_numpy(), rho_p=dsel["rho_p"].to_numpy(),
        ci_lo=dsel["ci_lo"].to_numpy(), ci_hi=dsel["ci_hi"].to_numpy(), q=dsel["q_bh"].to_numpy(),
        label_null_lo=hc["min"].reindex(dsel["cell_type"]).to_numpy(),
        label_null_hi=hc["max"].reindex(dsel["cell_type"]).to_numpy(), n_genes=dsel["n"].to_numpy()))
    tidy.to_csv(OUT / "hd1_per_type.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(dsel[["cell_type", "n", "rho_raw", "rho_pi", "rho_p", "ci_lo", "ci_hi", "p_one_sided", "q_bh",
                    "n_human_de", "tert_diff", "tert_p"]].to_string(index=False))
        print(json.dumps(rule, indent=2, default=float))
        print(ctrl.to_string(index=False))
    (OUT / "DONE").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
