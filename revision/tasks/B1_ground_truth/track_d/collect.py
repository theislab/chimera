"""B1 Track D tables: results/revision/B1/track_d/{runtime.csv, returns.csv} and tidy rows in B1_trackD_tidy.csv."""

from __future__ import annotations

import json
import re

import pandas as pd

from revision.tasks.B1_ground_truth import rules as RU
from revision.tasks.B1_ground_truth.track_d.make import HERE

OUT = RU.RES / "track_d"
RETURNS = [
    dict(method="Chimera", one_fit_returns="embedding z; species-blind shared decoder; alpha, Delta-alpha, S_g per gene; S_gk per cell type and species by a forward pass; decoders for imputation",
         labels_needed="no for liver recipes; cell-type conditional MMD on testis and kidney", gene_scalar="native", separate_runs_for_same_outputs=0),
    dict(method="Direct (B1, B1b, B1c, B2b, Sdir, B3a)", one_fit_returns="gene scalars only; no embedding, no decoder", labels_needed="yes, matched across species", gene_scalar="native",
         separate_runs_for_same_outputs="needs an integration tool for cells (B1z needs Chimera's or another latent)"),
    dict(method="SATURN", one_fit_returns="cell embedding; macrogene weights; gene scalars S_deploy, S_cos are our derivations", labels_needed="yes, within species", gene_scalar="derived", separate_runs_for_same_outputs="gene-level reading needs a further analysis"),
    dict(method="CAMEX", one_fit_returns="cell embedding; gene node embedding for HVG and DEG; scalar C_cos is our derivation", labels_needed="human labels, unused in integration", gene_scalar="derived, partial gene coverage", separate_runs_for_same_outputs="gene-level reading needs a further analysis"),
]


def parse_time(p):
    t = p.read_text()
    sec = float(re.search(r"wall_s: (\S+)", t).group(1))
    rss = int(re.search(r"peak_rss_kb: (\d+)", t).group(1)) / 1024 ** 2
    code = int(re.search(r"exit: (-?\d+)", t).group(1))
    return sec, rss, code


def main():
    j = pd.read_csv(HERE / "jobs.tsv", sep="\t", keep_default_na=False)
    rows = []
    for _, r in j.iterrows():
        f = OUT / f"{r['job']}.time.txt"
        if not f.exists():
            continue
        sec, rss, code = parse_time(f)
        d = r.to_dict()
        d.update(wall_s=sec, peak_rss_gb=rss, exit=code)
        fd = OUT / f"{r['job']}.direct.json"
        if fd.exists():
            d["direct_steps"] = json.dumps(json.loads(fd.read_text()))
        rows.append(d)
    t = pd.DataFrame(rows).drop(columns=["recipe"])
    t.to_csv(OUT / "runtime.csv", index=False)
    pd.DataFrame(RETURNS).to_csv(OUT / "returns.csv", index=False)
    tidy = []
    for _, r in t[t.exit == 0].groupby(["method", "cohort", "object", "n_species"]).agg(wall_s=("wall_s", "mean"), rss=("peak_rss_gb", "max"), n=("wall_s", "size")).reset_index().iterrows():
        s = f"cohort={r['cohort']};object={r['object']};n_species={r['n_species']};cpus=16"
        tidy.append(dict(track="D", task="runtime", setting=s, method=r["method"], metric="wall_h", value=r["wall_s"] / 3600, ci_lo=None, ci_hi=None, n=r["n"], rank_direction="lower"))
        tidy.append(dict(track="D", task="memory", setting=s, method=r["method"], metric="peak_rss_gb", value=r["rss"], ci_lo=None, ci_hi=None, n=r["n"], rank_direction="lower"))
    pd.DataFrame(tidy).to_csv(RU.RES / "B1_trackD_tidy.csv", index=False)
    print(t.to_string())


if __name__ == "__main__":
    main()
