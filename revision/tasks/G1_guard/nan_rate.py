"""G1: per setting, guard-on fits finished / non-finite / tried against the unguarded rows of the owning task (health records of B2, D1, T12.2b), plus where each non-finite guard fit died.
Unguarded counts use every rep the owner tried for that (cohort, setting, form), not only reps 0 and 1; rows still running are not counted.    python -m revision.tasks.G1_guard.nan_rate
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from revision.tasks.G1_guard.step3_prepare import REPO, RES

B2H, D1H = REPO / "results/revision/B2/health", REPO / "results/revision/D1/health"


def unguarded(kind, cohort, setting, form):
    if kind == "b2":
        pat = f"b2-{cohort}-{setting}{'_law' if form == 'law' else ''}-r[0-9].json"
        files = list(B2H.glob(pat))
        if form == "unchanged":  # D1 runs recipe-unchanged depth rows where B2 has none (r1 on testis)
            files += [f for f in D1H.glob(f"d1-{cohort}-{setting}-r[0-9].json") if not (B2H / f.name.replace("d1-", "b2-")).exists()]
    else:
        files = list(D1H.glob(f"d1-{cohort}-{setting}{'law' if form == 'law' else ''}-r[0-9].json"))
        if form == "unchanged":
            files += [f for f in B2H.glob(f"b2-{cohort}-{setting}-r[0-9].json") if not (D1H / f.name.replace("b2-", "d1-")).exists()]
    h = [json.loads(f.read_text()) for f in files]
    return len(h), sum(not x["discard"] for x in h), sum(x.get("finite") is False for x in h), ",".join(sorted(f.stem for f in files))


def main():
    g = pd.read_csv(REPO / "revision/tasks/G1_guard/step3_grid.tsv", sep="\t", keep_default_na=False, dtype=str)
    rows, died = [], []
    for (kind, cohort, setting, form), t in g.groupby(["kind", "cohort", "setting", "form"], sort=False):
        hs = [json.loads((RES / "health" / f"{r}.json").read_text()) for r in t.run if (RES / "health" / f"{r}.json").exists()]
        r = dict(kind=kind, cohort=cohort, setting=setting, form=form, guard_planned=len(t), guard_ended=len(hs),
                 guard_finished=sum(not h["discard"] for h in hs), guard_nonfinite=sum(h.get("finite") is False for h in hs),
                 guard_finished_tail2=sum((not h["discard"]) and max(h.get("tail_pos") or 0, h.get("tail_neg") or 0) >= 0.02 for h in hs),
                 guard_zlv_train_max=max([h.get("z_logvar_max_train", np.nan) for h in hs], default=np.nan))
        if kind in ("b2", "d1"):
            n, f, nf, names = unguarded(kind, cohort, setting, form)
            r.update(unguarded_tried=n, unguarded_finished=f, unguarded_nonfinite=nf, unguarded_runs=names)
        elif kind == "bm":
            r.update(unguarded_tried=2, unguarded_finished=0, unguarded_nonfinite=2, unguarded_runs="rev-bm2-s44,rev-bm2-s45 (P12: NaN)")
        for h in hs:
            if h.get("finite") is False:
                died.append({k: h.get(k) for k in ("run", "first_nonfinite_step", "first_nonfinite_epoch", "first_nonfinite_terms", "before_first_nonfinite", "z_logvar_max_train")})
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(RES / "nan_rate.csv", index=False)
    pd.DataFrame(died).to_csv(RES / "nan_guard_died.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(d.drop(columns=["unguarded_runs"]).to_string())
    print(pd.DataFrame(died).to_string())


if __name__ == "__main__":
    main()
