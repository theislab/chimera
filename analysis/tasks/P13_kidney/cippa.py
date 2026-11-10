"""External anchor E-K3 of the kidney case: Cippa et al. 2018 human-specific vs shared injury genes
(pre-specified analysis plan, section 8c).

Family 3. Source: ``data/analysis/disease_kidney/raw/cippa2018/jci.insight.123151.sdt1-11.xlsx``
(Table S3 primary: acute, human post- vs pre-reperfusion and mouse 4 h IRI; Table S9 secondary: late).
Human symbols map to the kidney table by unique ``hsapiens_gene_name``; ambiguous or absent symbols
are dropped and counted. "Mouse specific" lists are reported only as counts.

Mechanics (fixed before reading D against the lists; amendment log of the plan, row of 2026-11-02):
* primary: stratified AUROC of D, Shared (set) vs Human specific (comparison), within strata of
  rank pi_raw tertile x rank expression tertile computed over the whole vocabulary (the lists are
  too small for the decile cells and 5:1 background rule of plan, section 4); strata need >= 1 gene of each
  list; one-sided permutation p (list labels permuted within strata, 10,000 draws); raw AUROC beside;
* each list against matched vocabulary background with the rule of plan, section 4 (pi_raw x expression
  decile cells, >= 5 background genes per list gene), and the difference Shared minus Human specific;
* the same for B1b_raw (pseudobulk correlation) and D_raw, beside D.

    PYTHONPATH=<chimera repo> python -I analysis/tasks/P13_kidney/cippa.py [--n-perm 10000]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from analysis.common import gene_ids
from analysis.tasks.P13_kidney.xlsx_min import read_xlsx
from analysis.tasks.T4_3_scoreboard import stats as S

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis" / "P13"
OUT = __import__("pathlib").Path(__import__("os").environ.get("P13_CIPPA_OUT", str(RES / "cippa")))
WB = ROOT / "data" / "analysis" / "disease_kidney" / "raw" / "cippa2018" / "jci.insight.123151.sdt1-11.xlsx"
GENES = __import__("pathlib").Path(__import__("os").environ.get("P13_GENES", str(RES / "readout" / "kidney_genes.parquet")))
SEED = 20261103
SCORES = ("D", "D_raw", "B1b_raw")


def lists() -> dict:
    wb = read_xlsx(str(WB))
    out = {}
    for sheet, tag in (("TableS3", "acute"), ("TableS9", "late")):
        rows = wb[sheet]
        hdr = rows[1]
        for j, h in enumerate(hdr):
            out[(tag, h.strip())] = sorted({r[j].strip() for r in rows[2:] if len(r) > j and r[j].strip()})
    return out


def tertiles(x: np.ndarray) -> np.ndarray:
    r = S.midrank(x)
    return np.minimum((3 * (r - 0.5) / len(r)).astype(int), 2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=10000)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tab = pd.read_parquet(GENES).dropna(subset=["prior_raw", "log_cp10k_spmean", "D"]).reset_index(drop=True)
    sym = tab["symbol"].astype(str)
    uniq = sym[~sym.duplicated(keep=False)]
    pos = pd.Series(uniq.index, index=uniq.to_numpy())
    L = lists()
    mapping = []
    for (tag, name), genes in L.items():
        hit = [g for g in genes if g in pos.index]
        mapping.append(dict(table=tag, list=name, n_list=len(genes), n_in_vocabulary=len(hit)))
    pd.DataFrame(mapping).to_csv(OUT / "list_mapping.tsv", sep="\t", index=False)
    coarse = tertiles(tab["prior_raw"].to_numpy()) * 3 + tertiles(tab["log_cp10k_spmean"].to_numpy())
    fine = S.cells(tab["prior_raw"].to_numpy(), tab["log_cp10k_spmean"].to_numpy())
    rng = np.random.default_rng(SEED)
    rows = []
    for tag in ("acute", "late"):
        sh = set(pos.reindex([g for g in L[(tag, "Shared")] if g in pos.index]).to_numpy())
        hs = set(pos.reindex([g for g in L[(tag, "Human specific")] if g in pos.index]).to_numpy())
        both = sh & hs
        sh, hs = sh - both, hs - both
        idx = np.array(sorted(sh | hs), dtype=int)
        inset = np.isin(idx, list(sh))
        lab = coarse[idx]
        keep = np.zeros(len(idx), bool)
        for s in np.unique(lab):
            m = lab == s
            if inset[m].any() and (~inset[m]).any():
                keep |= m
        for score in SCORES:
            v = tab[score].to_numpy(dtype=float)
            ok = keep & np.isfinite(v[idx])
            x, i_, l_ = v[idx][ok], inset[ok], lab[ok]
            auc = S.strat_auroc(x, i_, l_)
            null = S.auroc_perm(x, i_, l_, a.n_perm, rng)
            raw = S.strat_auroc(x, i_, np.zeros(len(x), int))
            rec = dict(table=tag, score=score, n_shared=int(i_.sum()), n_human_specific=int((~i_).sum()),
                       n_overlap_dropped=len(both), n_strata=int(len(np.unique(l_))), auroc_stratified=auc,
                       p_one_sided=float((1 + (null >= auc).sum()) / (1 + len(null))), null_mean=float(null.mean()),
                       auroc_raw=raw)
            # each list vs matched vocabulary background (rule of pre-specified analysis plan, section 4)
            vv = np.isfinite(v)
            for nm, st in (("shared", sh), ("human_specific", hs)):
                ins = np.zeros(len(tab), bool)
                ins[list(st)] = True
                kp = S.kept_strata(ins, fine) & vv
                rec[f"auroc_vs_background_{nm}"] = S.strat_auroc(v[kp], ins[kp], fine[kp])
                rec[f"n_kept_{nm}"] = int(ins[kp].sum())
            rec["background_difference"] = rec["auroc_vs_background_shared"] - rec["auroc_vs_background_human_specific"]
            rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ek3.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(mapping).to_string(index=False))
        print(res.to_string(index=False))
    (OUT / "DONE").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
