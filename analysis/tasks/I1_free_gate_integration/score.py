"""Per-run scoring: one fitted latent, every integration and embedding metric the paper uses, with the paper's own functions.

    python -m analysis.tasks.I1_free_gate_integration.score unit <cohort> <run> [--stages fig3,h1,z2,enc,spec]
    python -m analysis.tasks.I1_free_gate_integration.score list            units of rules.lines() not yet scored (cohort run per line)
    python -m analysis.tasks.I1_free_gate_integration.score check           harness check on canonical runs (no free-gate value)

Outputs results/analysis/I1/units/<cohort>__<run>__<stage>.{json,parquet}; a finished stage is skipped.
"""
from __future__ import annotations

import json
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

from analysis.tasks.I1_free_gate_integration import rules as RU

sys.path.insert(0, str(RU.REPO / "training"))
sys.path.insert(0, str(RU.REPO / "notebooks"))
sys.path.insert(0, str(RU.REPO / "analysis/tasks/T5_5_benchmark"))
from compare_vs_scvi import compute_integration_metrics  # noqa: E402
import score_external as SE  # noqa: E402

UNITS = RU.OUT / "units"
STAGES = ("fig3", "h1", "z2", "enc", "spec")


def load_z(run: str) -> tuple[np.ndarray, float]:
    with np.load(RU.RUNS / run / "final_state.npz", allow_pickle=True) as f:
        z = np.asarray(f["z_mean"])
        adv = float(f["adversarial_accuracy_final"]) if "adversarial_accuracy_final" in f.files else np.nan
    return z, adv


def _obs(h5ad, keys):
    a = ad.read_h5ad(h5ad, backed="r")
    o = {k: a.obs[k].astype(str).to_numpy() for k in keys}
    a.file.close()
    return o


def fig3_metrics(c: str, z: np.ndarray) -> dict:
    stem = SE.DATASETS[RU.FIG3[c]][0]
    o = _obs(SE.DATA / f"{stem}.h5ad", ["species", "cell_type"])
    if z.shape[0] != len(o["species"]):
        raise SystemExit(f"{c}: latent {z.shape} vs {len(o['species'])} cells")
    g = SE.scorecard(z, pd.DataFrame(o))["global"]
    return {k: float(g[k]) for k in RU.FIG3_METRICS} | {"_n": int(z.shape[0])}


def h1_metrics(c: str, z: np.ndarray, adv: float) -> dict:
    from analysis.tasks.H1_harmonise import rules as H1R
    d = H1R.cohorts()[c]
    o = _obs(d["h5ad"], ["species", d["ct"]])
    if z.shape[0] != len(o["species"]):
        raise SystemExit(f"{c}: latent {z.shape} vs {len(o['species'])} cells")
    m = compute_integration_metrics(z, o["species"], o[d["ct"]], k=15, seed=42)["global"]
    return {k: float(m[k]) for k in RU.FIG3_METRICS} | {"adv_acc": adv, "label_key": d["ct"], "_n": int(z.shape[0])}


def z2_rows(c: str, z: np.ndarray) -> pd.DataFrame | None:
    from analysis.tasks.Z2_embedding import common as K, rules as Z2R
    if c not in Z2R.H5AD or not (K.CACHE / f"{c}_targets.npz").exists():
        return None
    return K.all_rows(c, z, with_global=False)


def enc_rows(c: str, run: str) -> pd.DataFrame | None:
    if c not in ("liv_hmouse", "test_hmouse"):
        return None
    from analysis.tasks.Z2_embedding import encode as E
    enc, z, a = E.chimera_encoder(run, c)
    rows = E.liver_rows(enc, a.var_names) if c == "liv_hmouse" else E.testis_rows(enc, z, a)
    return pd.DataFrame(rows)


def spectrum(z: np.ndarray) -> dict:
    sv = np.linalg.svd(z - z.mean(0), compute_uv=False) ** 2
    p = sv / sv.sum()
    return dict(pc1_share=float(p[0]), pc2_share=float(p[1]), participation_ratio=float(1 / np.sum(p ** 2)))


def path(c: str, run: str, stage: str):
    return UNITS / f"{c}__{run}__{stage}.{'parquet' if stage in ('z2', 'enc') else 'json'}"


def unit(c: str, run: str, stages=STAGES) -> None:
    UNITS.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stages if not path(c, run, s).exists()]
    if not todo:
        return
    z, adv = load_z(run)
    if not np.isfinite(z).all():
        print(c, run, "non-finite latent; skipped", flush=True)
        return
    for s in todo:
        t0 = time.time()
        if s == "fig3":
            if c not in RU.FIG3:
                continue
            path(c, run, s).write_text(json.dumps(fig3_metrics(c, z), indent=1))
        elif s == "h1":
            path(c, run, s).write_text(json.dumps(h1_metrics(c, z, adv), indent=1))
        elif s == "spec":
            path(c, run, s).write_text(json.dumps(spectrum(z), indent=1))
        elif s == "z2":
            d = z2_rows(c, z)
            if d is None:
                continue
            d.to_parquet(path(c, run, s))
        elif s == "enc":
            d = enc_rows(c, run)
            if d is None:
                continue
            d.to_parquet(path(c, run, s))
        print(c, run, s, f"{time.time() - t0:.0f}s", flush=True)


def check() -> dict:
    """Canonical runs only: the Chimera s42 rows of the integration benchmark (fig:integration), H1_harmonise integration cache, Z2_embedding measure and encoded tables must be reproduced."""
    out = {}
    from analysis.tasks.Z2_embedding import rules as Z2R
    cache = SE.REPO / "notebooks" / "cache" / "baselines_master"
    reg = {"liv_hmouse": "xds-hmouse-iter90-150ep-s42", "liv_4sp": "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42",
           "test_hmouse": "testis-hmouse-iter111-pclamp-coc-200ep-s42"}
    for c, run in reg.items():
        z, adv = load_z(run)
        new = fig3_metrics(c, z)
        old = json.loads((cache / f"{RU.FIG3[c]}__Chimera.json").read_text())["global"]
        out[f"fig3|{c}|{run}"] = max(abs(new[k] - old[k]) for k in RU.FIG3_METRICS)
    ic = pd.read_csv(RU.RES / "H1" / "cache" / "integration.csv").set_index("run")
    for c, run in (("liv_hmouse", "rev-liv_hmouse-s47"), ("test_hmouse", "rev-test_hmouse-s43"), ("test_3sp", ic[ic.cohort == "test_3sp"].index[0])):
        z, adv = load_z(run)
        new = h1_metrics(c, z, adv)
        out[f"h1|{c}|{run}"] = max(abs(new[k] - ic.loc[run, k]) for k in ("knn_mixing_entropy", "knn_species_purity", "label_transfer_acc"))
    for c, s in (("liv_hmouse", 47), ("test_hmouse", 43)):
        f = Z2R.OUT / "measure" / f"{c}__chimera_s{s}.parquet"
        from analysis.tasks.Z2_embedding import common as K
        run = K.chimera_runs(c)[s]
        old = pd.read_parquet(f)
        old = old[old.task != "global"].reset_index(drop=True)
        z, _ = load_z(run)
        new = z2_rows(c, z)
        key = ["task", "kind", "metric", "a", "b"]
        m = old.merge(new, on=key, suffixes=("_old", "_new"))
        out[f"z2|{c}|{run}|rows {len(old)}/{len(new)}/{len(m)}"] = float((m.value_old - m.value_new).abs().max())
    e = Z2R.OUT / "encoded" / "liv_hmouse__rev-liv_hmouse-s47.parquet"
    if e.exists():
        old = pd.read_parquet(e)
        new = enc_rows("liv_hmouse", "rev-liv_hmouse-s47")
        m = old.merge(new, on=["set", "metric", "depth", "marker"], suffixes=("_old", "_new"))
        out[f"enc|liv_hmouse|rev-liv_hmouse-s47|rows {len(old)}/{len(m)}"] = float((m.value_old - m.value_new).abs().max())
    RU.OUT.mkdir(parents=True, exist_ok=True)
    (RU.OUT / "harness_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    return out


def todo_units() -> list[tuple[str, str]]:
    L = RU.lines().drop_duplicates(["cohort", "run"])
    out = []
    for r in L.itertuples():
        if not (RU.RUNS / r.run / "final_state.npz").exists():
            continue
        need = [s for s in STAGES if not path(r.cohort, r.run, s).exists()]
        if r.cohort not in RU.FIG3:
            need = [s for s in need if s != "fig3"]
        if r.cohort not in ("liv_hmouse", "test_hmouse"):
            need = [s for s in need if s != "enc"]
        from analysis.tasks.Z2_embedding import rules as Z2R
        if r.cohort not in Z2R.H5AD:
            need = [s for s in need if s != "z2"]
        if need:
            out.append((r.cohort, r.run))
    return out


def main():
    k = sys.argv[1]
    if k == "unit":
        st = STAGES
        if "--stages" in sys.argv:
            st = tuple(sys.argv[sys.argv.index("--stages") + 1].split(","))
        unit(sys.argv[2], sys.argv[3], st)
    elif k == "list":
        for c, r in todo_units():
            print(c, r)
    elif k == "check":
        check()


if __name__ == "__main__":
    main()
