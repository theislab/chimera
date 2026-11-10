"""Rate and effective axes of factorial runs: python -m analysis.tasks.Z2_embedding.snr_runs  Output: results/analysis/Z2/snr_runs.csv. Capacity and effective axes as snr.py. Skips finished runs."""
import pandas as pd
from analysis.tasks.Z2_embedding import rules as RU
from analysis.tasks.Z2_embedding.train_patched import apply_patch
import sys
sys.argv = [sys.argv[0], "runs"]
from analysis.tasks.Z2_embedding import snr as S

f = RU.OUT / "snr_runs.csv"
old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["run"])
fac = pd.read_csv(RU.REPO / "analysis/tasks/Z2_embedding/factorial.tsv", sep="\t")
rows = old.to_dict("records")
patched = None
for r in fac.itertuples():
    if r.input != "raw" or r.run in set(old.run) or not (RU.RUNS / r.run / "summary.json").exists() :
        continue
    from analysis.tasks.Z2_embedding.train_patched import apply_patch as _ap
    post = S.chimera_post(r.cohort, r.run)
    if post is None:
        continue
    a, p = S.stats(r.cohort, r.run, "Chimera", r.seed, *post)
    rows.append(dict(run=r.run, cohort=r.cohort, level=r.level, seed=r.seed, rate=a["rate"], capacity=a["capacity"], axes=a["axes"], sp_auc_biglabel=a["sp_auc_biglabel"], sp_auc_all=a["sp_auc_all"], pc1=float(p.share.iloc[0]), pc2=float(p.share.iloc[1]), pc3=float(p.share.iloc[2])))
    pd.DataFrame(rows).to_csv(f, index=False)
    print(r.run, flush=True)
print(pd.DataFrame(rows).groupby(["cohort", "level"])[["rate", "capacity", "axes", "pc1", "pc2", "pc3", "sp_auc_biglabel"]].mean().round(3).to_string())
