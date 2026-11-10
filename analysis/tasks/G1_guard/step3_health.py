"""Step 3: health of one guard-arm fit, by the owning task's definitions (B2/D1: alpha, z, S_g finite and 10 latent dims with Var(z_mean) >= 0.01; gate tails beyond 0.05 of the raw prior;
bone marrow also the T12_2b_recipe G-V gate). Plus where the run first went non-finite, from the probe. Reads no gold, no readout beyond these.
    python -m analysis.tasks.G1_guard.step3_health --index I
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from analysis.tasks.G1_guard.step3_prepare import REPO, RES, row

RUNS = REPO / "training" / "runs"


def probe_summary(d) -> dict:
    p = d / "nan_probe.csv"
    if not p.exists():
        return dict(probe="missing")
    t = pd.read_csv(p)
    out = dict(probe_steps=int(len(t)), z_logvar_max_train=float(t["z_logvar_max"].max()), z_logvar_min_train=float(t["z_logvar_min"].min()),
               steps_z_logvar_above_10=int((t["z_logvar_max"] > 10).sum()), z_absmax_train=float(t["z_absmax"].replace(np.inf, np.nan).max()))
    bad = t[t["nonfinite"].fillna("").astype(str) != ""]
    if len(bad):
        b = bad.iloc[0]
        out.update(first_nonfinite_step=int(b["step"]), first_nonfinite_epoch=int(b["epoch"]), first_nonfinite_terms=str(b["nonfinite"]))
        prev = t[t["step"] < b["step"]].tail(1)
        if len(prev):
            out["before_first_nonfinite"] = {k: float(prev.iloc[0][k]) for k in ("z_logvar_max", "z_absmax", "z_mean_absmax", "log_rate_full_max", "mu_full_max", "loss") if k in prev}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    r = row(a.index)
    d = RUNS / r["run"]
    h = dict(run=r["run"], kind=r["kind"], cohort=r["cohort"], setting=r["setting"], rep=r["rep"], seed=r["seed"], form=r["form"], base=r["base"], arm="guard")
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    h["clamp_z_logvar"] = cfg.get("clamp_z_logvar")
    if (d / "g1_mode.json").exists():
        h["mode"] = json.loads((d / "g1_mode.json").read_text())["mode"]
    if (d / "final_state.npz").exists():
        with np.load(d / "final_state.npz", allow_pickle=False) as f:
            al, pr, z, sg = (np.asarray(f[k], dtype=float) for k in ("alpha", "prior", "z_mean", "S_g_ema"))
        raw = np.load(d / "conservation_prior_raw.npy") if (d / "conservation_prior_raw.npy").exists() else pr
        fin = bool(np.isfinite(al).all() and np.isfinite(z).all() and np.isfinite(sg).all())
        vz = z.var(0)
        act = int((vz >= 0.01).sum()) if fin else 0
        dr = al - raw
        h.update(finished=True, finite=fin, active_dims=act, discard=not (fin and act >= 10), tail_pos=float((dr > 0.05).mean()) if fin else 0.0,
                 tail_neg=float((dr < -0.05).mean()) if fin else 0.0, sat_hi=float((al > 0.95).mean()) if fin else None, sat_lo=float((al < 0.05).mean()) if fin else None)
        if r["kind"] == "bm":
            from analysis.tasks.T12_2b_recipe.select_recipe import gate

            g = gate(d)
            h["GV"] = {k: (v if isinstance(v, (bool, int, float, str)) else str(v)) for k, v in g.items()}
    else:
        h.update(finished=False, finite=False, discard=True)
    h.update(probe_summary(d))
    (RES / "health").mkdir(parents=True, exist_ok=True)
    (RES / "health" / f"{r['run']}.json").write_text(json.dumps(h, indent=1, default=str))
    print(h, flush=True)


if __name__ == "__main__":
    main()
