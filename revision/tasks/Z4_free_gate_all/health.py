"""Z4, gold-free: health of every free-gate candidate (Z4 fits and Z3's fits of liver H+Mouse, testis H+Mouse, liver 4 species),
integration against the cohort's kept canonical seeds, recipe check against the canonical config, and the kept seeds per cohort.
Reads final_state.npz, config.json and summary.json only; no gold, set or reference is opened.

    PYTHONPATH=$PWD python -m revision.tasks.Z4_free_gate_all.health [--jobs 8]
Output: results/revision/Z4/health.csv, kept.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.tasks.H1_harmonise import rules as H1R
from revision.tasks.Z4_free_gate_all import rules as RU

CACHE = RU.OUT / "health_cache"
IGNORE = {"run_name", "seed", "started_at", "ablation_config", "use_per_gene_bias", "clamp_z_logvar", "batch_key"}


def manifests() -> pd.DataFrame:
    fs = [RU.TASK / "manifest.tsv"] + sorted(RU.TASK.glob("manifest_*.tsv"))
    return pd.concat([pd.read_csv(f, sep="\t").assign(manifest=f.name) for f in fs if f.exists()], ignore_index=True)


def candidates() -> list[dict]:
    rows = []
    for c, d in RU.Z3_FREE.items():
        rows += [dict(cohort=c, seed=s, run=r, source="Z3") for s, r in sorted(d.items())]
    m = manifests()
    rows += [dict(cohort=r.cohort, seed=int(r.seed), run=r.run_name, source=f"Z4 {r.manifest}", recipe_from=r.recipe_from) for r in m.itertuples()]
    return rows


def recipe_check(run: str, ref: str) -> str:
    """Keys whose value differs between the free run's config and the canonical run it copies (besides the gate, seed, name, time)."""
    a = json.loads((RU.RUNS / run / "config.json").read_text())
    p = RU.RUNS / ref / "config.json"
    if not p.exists():
        return "no reference config"
    b = json.loads(p.read_text())
    diff = []
    for k in sorted(set(a) | set(b)):
        if k in IGNORE:
            continue
        va, vb = a.get(k), b.get(k)
        if k == "plan_kwargs":
            va, vb = json.dumps(va, sort_keys=True), json.dumps(vb, sort_keys=True)
        if va != vb and not (vb is None and k not in b):  # keys newer than the canonical run's code are not a recipe change
            diff.append(f"{k}: {a.get(k)!r} vs {b.get(k)!r}")
    return "; ".join(diff)


def gate(run: str) -> dict:
    d = RU.RUNS / run
    r = dict(finished=(d / "summary.json").exists() and (d / "final_state.npz").exists())
    if not r["finished"]:
        return dict(r, healthy_gate=False)
    with np.load(d / "final_state.npz", allow_pickle=False) as z:
        st = {k: np.asarray(z[k]) for k in z.files}
    cfg = json.loads((d / "config.json").read_text())
    summ = json.loads((d / "summary.json").read_text())
    a, p, zm, S = (np.asarray(st[k], float) for k in ("alpha", "prior", "z_mean", "S_g_ema"))
    f = d / "conservation_prior_raw.npy"
    praw = np.load(f).astype(float) if f.exists() else p
    r["config"] = cfg.get("ablation_config")
    r["no_nan"] = bool(np.isfinite(a).all() and np.isfinite(p).all() and np.isfinite(zm).all() and np.isfinite(S).all()
                       and "E2.1" not in summ.get("failed_expectations", []))
    if not r["no_nan"]:
        return dict(r, healthy_gate=False)
    r["sat_high"], r["sat_low"] = float((a > 0.95).mean()), float((a < 0.05).mean())
    r["n_active_z"] = int((zm.var(0) >= RU.ZVAR_MIN).sum())
    D, Dr = a - p, a - praw
    r["tail_up_raw"], r["tail_dn_raw"] = float((Dr > RU.TAIL_DELTA).mean()), float((Dr < -RU.TAIL_DELTA).mean())
    r["tail_up_anch"], r["tail_dn_anch"] = float((D > RU.TAIL_DELTA).mean()), float((D < -RU.TAIL_DELTA).mean())
    Dc = D - np.median(D)
    r["tail_up_centred"], r["tail_dn_centred"] = float((Dc > RU.TAIL_DELTA).mean()), float((Dc < -RU.TAIL_DELTA).mean())
    sup = np.abs(S) >= RU.EPS
    r["frac_count_supported"] = float(sup.mean())
    r["rho_D_S_supported"] = float(spearmanr(D[sup], S[sup])[0])
    r["rho_alpha_prior"] = float(spearmanr(a, p)[0])
    r["sd_D"], r["sd_D_supported"] = float(np.std(D)), float(np.std(D[sup]))
    r["masked_at_prior"] = float((np.abs(D[~sup]) < 0.005).mean()) if (~sup).any() else np.nan
    r["adversary"] = float(st["adversarial_accuracy_final"]) if "adversarial_accuracy_final" in st else np.nan
    r["n_failed_expectations"] = len(summ.get("failed_expectations", []))
    r["saturation_flag"] = bool(max(r["sat_high"], r["sat_low"]) > RU.SAT_FLAG)
    r["healthy_gate"] = bool(r["n_active_z"] == RU.N_ACTIVE and zm.shape[1] == RU.N_ACTIVE and r["sat_high"] <= RU.SAT_MAX
                             and r["sat_low"] <= RU.SAT_MAX and r["rho_D_S_supported"] >= RU.RHO_MIN and r["config"] == "M")
    return r


def integ(args) -> dict:
    c, run = args
    f = CACHE / f"{run}.json"
    if f.exists():
        return json.loads(f.read_text())
    sys.path.insert(0, str(RU.REPO / "training"))
    from compare_vs_scvi import compute_integration_metrics  # noqa: PLC0415

    d = H1R.cohorts()[c]
    a = ad.read_h5ad(d["h5ad"], backed="r")
    sp_, lab = a.obs["species"].astype(str).to_numpy(), a.obs[d["ct"]].astype(str).to_numpy()
    a.file.close()
    with np.load(RU.RUNS / run / "final_state.npz", allow_pickle=False) as z:
        zm = np.asarray(z["z_mean"])
    if not np.isfinite(zm).all():
        return {"int_note": "non-finite z"}
    m = compute_integration_metrics(zm, sp_, lab, k=15, seed=42)["global"]
    out = {f"int_{k}": float(m[k]) for k in ("knn_mixing_entropy", "knn_species_purity", "label_transfer_acc")}
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    RU.OUT.mkdir(parents=True, exist_ok=True)
    runs_h1 = pd.read_csv(H1R.OUT / "runs.tsv", sep="\t")
    kept_can = runs_h1[runs_h1.kept.astype(bool)]
    ic = pd.read_csv(H1R.OUT / "cache" / "integration.csv")
    ic = ic[ic.run.isin(kept_can.run)]
    cand = candidates()
    rows = []
    for x in cand:
        g = gate(x["run"])
        rows.append({**x, **g})
    H = pd.DataFrame(rows)
    todo = [(r.cohort, r.run) for r in H.itertuples() if r.finished and r.no_nan]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(integ, todo))
    I = pd.DataFrame([dict(run=r, **v) for (c, r), v in zip(todo, res)])
    H = H.merge(I, on="run", how="left") if len(I) else H
    ref = ic.groupby("cohort").agg(can_entropy_median=("knn_mixing_entropy", "median"), can_entropy_min=("knn_mixing_entropy", "min"),
                                    can_adv_max=("adv_acc", "max"), can_label_transfer_median=("label_transfer_acc", "median"))
    H = H.merge(ref, left_on="cohort", right_index=True, how="left")
    ent = H.get("int_knn_mixing_entropy", pd.Series(np.nan, index=H.index))
    H["mixing_ok"] = (ent >= RU.MIX_FRAC * H.can_entropy_median) & (H.adversary.fillna(1) <= H.can_adv_max + RU.ADV_MARGIN)
    H["healthy"] = H.healthy_gate.fillna(False).astype(bool) & H.mixing_ok.fillna(False).astype(bool)
    # recipe check against the canonical run the recipe was copied from (Z3 fits: Z3 checked theirs)
    first_kept = kept_can.sort_values("seed").groupby("cohort").run.first()
    chk = []
    for r in H.itertuples():
        if not r.finished or r.source == "Z3":
            chk.append("")
            continue
        refrun = r.recipe_from if isinstance(r.recipe_from, str) and (RU.RUNS / r.recipe_from).exists() else first_kept[r.cohort]
        chk.append(recipe_check(r.run, refrun))
    H["recipe_diff"] = chk
    H.to_csv(RU.OUT / "health.csv", index=False)
    kept = {}
    for c, g in H.groupby("cohort", sort=False):
        g = g.sort_values("seed")
        if c in RU.Z3_FREE:  # Z3's line: first 5 passing Z3's gate rule (mixing reported, not applied, as Z3 fixed it)
            ok = g[g.healthy_gate.fillna(False).astype(bool)].head(RU.Z3_MAX_SEEDS)
        else:
            ok = g[g.healthy]
        kept[c] = {int(s): r for s, r in zip(ok.seed, ok.run)}
    (RU.OUT / "kept.json").write_text(json.dumps(kept, indent=1))
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    cols = ["cohort", "seed", "finished", "no_nan", "healthy_gate", "mixing_ok", "healthy", "sat_high", "sat_low", "n_active_z", "rho_D_S_supported",
            "tail_up_raw", "tail_dn_raw", "tail_up_centred", "tail_dn_centred", "masked_at_prior", "adversary", "can_adv_max", "int_knn_mixing_entropy",
            "can_entropy_median", "int_label_transfer_acc", "recipe_diff"]
    print(H[[k for k in cols if k in H]].round(3).to_string())
    print({c: list(v) for c, v in kept.items()})


if __name__ == "__main__":
    main()
