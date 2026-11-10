"""Cross-check: per-gene COE published on the Shay et al. 2013 web portal.

Shay Dataset S3 could not be fetched (PMC serves a proof-of-work page to scripts, PNAS and
Europe PMC answer 403; Europe PMC lists the article as not open access). The portal named in
the paper (http://rstats.immgen.org/comparative/comparative_search.php) returns, per human gene,
the probeset and the human-mouse Pearson correlation ("pearson") it shows. This script queries
the ten genes whose divergent lineage Shay names plus a seeded random sample of the recomputed
COE set, one request per second, and writes the portal values beside the recomputed COE.

    python -I analysis/tasks/T12_1_bone_marrow/fetch_portal_coe.py [--n 150] [--seed 20261101]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BM = REPO / "data" / "analysis" / "bone_marrow"
URL = "http://rstats.immgen.org/comparative/comparative_search.php"
NAMED = ["DPP4", "NT5E", "CD27", "TFEC", "LMO2", "IL15", "MYBL1", "LDB2", "NFIX", "PRDM1"]
PAT = re.compile(r"human_symbol=([^&\"]*)&probeset_id=([^&\"]*)&mouse_symbol=([^&\"]*)&pearson=([-0-9.eE]+)")


def query(symbol: str) -> dict:
    data = urllib.parse.urlencode({"sTermHuman": symbol, "submit_term_human": "Search"}).encode()
    req = urllib.request.Request(URL, data=data, headers={"User-Agent": "chimera-crosscheck"})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf8", errors="replace")
    m = PAT.search(html)
    if not m:
        return {"symbol": symbol, "portal_found": False}
    return {"symbol": symbol, "portal_found": True, "portal_human_symbol": m.group(1),
            "portal_probeset": m.group(2), "portal_mouse_symbol": m.group(3), "portal_pearson": float(m.group(4))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    coe = pd.read_csv(BM / "shay_coe.tsv", sep="\t")
    coe = coe.dropna(subset=["hsapiens_gene_name"]).drop_duplicates("hsapiens_gene_name")
    pool = coe[~coe["hsapiens_gene_name"].isin(NAMED)].sort_values("hsapiens_gene_id")
    rng = np.random.default_rng(args.seed)
    sample = pool.iloc[np.sort(rng.choice(len(pool), size=min(args.n, len(pool)), replace=False))]
    genes = NAMED + sample["hsapiens_gene_name"].tolist()
    rows = []
    for g in genes:
        try:
            r = query(g)
        except Exception as e:  # noqa: BLE001
            r = {"symbol": g, "portal_found": False, "error": type(e).__name__}
        r["group"] = "named" if g in NAMED else "random"
        rows.append(r)
        time.sleep(1.0)
    out = pd.DataFrame(rows).merge(coe[["hsapiens_gene_name", "coe", "divergent", "human_probeset"]],
                                   left_on="symbol", right_on="hsapiens_gene_name", how="left")
    (BM / "shay").mkdir(exist_ok=True)
    out.to_csv(BM / "shay" / "portal_coe_crosscheck.tsv", sep="\t", index=False)
    ok = out[out["portal_found"] & out["coe"].notna()]
    same = ok[ok["portal_probeset"] == ok["human_probeset"]]
    from scipy.stats import pearsonr, spearmanr

    print(f"queried {len(out)}, found {int(out['portal_found'].sum())}, with recomputed COE {len(ok)}")
    if len(ok) > 3:
        print("pearson(recomputed, portal) all:", round(pearsonr(ok["coe"], ok["portal_pearson"])[0], 3),
              "spearman:", round(spearmanr(ok["coe"], ok["portal_pearson"])[0], 3),
              "median |diff|:", round(float((ok["coe"] - ok["portal_pearson"]).abs().median()), 3))
    if len(same) > 3:
        print(f"same human probeset n={len(same)}: pearson", round(pearsonr(same["coe"], same["portal_pearson"])[0], 3),
              "median |diff|:", round(float((same["coe"] - same["portal_pearson"]).abs().median()), 3))
    print(out[out["group"] == "named"][["symbol", "portal_pearson", "coe", "divergent", "portal_probeset", "human_probeset"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
