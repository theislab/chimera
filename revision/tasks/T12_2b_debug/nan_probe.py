"""T12.2b hypothesis 3: locate the first non-finite tensor in a bone-marrow training run.

Wraps ``ChimeraModule.forward`` (no change to the package) and records, for every training
step, the maxima of the tensors that can overflow: library size, counts, z_mean, z_logvar,
the sampled z, the log-rate pieces (shared, batch offset, residual), mu_full, theta, and
every loss term. Rows go to ``<run_dir>/nan_probe.csv`` every 50 steps and at the first
step where any recorded value is non-finite; at that step the per-cell rows of the batch
(library size, max count, max |z|, max z_logvar, max log-rate) go to
``<run_dir>/nan_probe_first_batch.csv``. Training then continues as usual.

Health only: nothing here reads alpha, S_g or any readout beyond their finiteness.

    python revision/tasks/T12_2b_debug/nan_probe.py <run_experiment.py arguments>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from chimera import _module  # noqa: E402

STATE = {"step": 0, "rows": [], "dumped": False, "run_dir": None, "header": None}
TERMS = ["L_recon", "L_kl", "L_shared_aux", "L_residual_sparsity", "L_alpha_anchor",
         "L_fit_reward", "L_mmd", "L_KL_l"]


def _mx(t: torch.Tensor) -> float:
    t = t.detach()
    if not torch.isfinite(t).all():
        return float("inf")
    return float(t.abs().max())


def _flush() -> None:
    if STATE["run_dir"] is None or not STATE["rows"]:
        return
    path = STATE["run_dir"] / "nan_probe.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATE["header"])
        if new:
            w.writeheader()
        w.writerows(STATE["rows"])
    STATE["rows"] = []


_orig = _module.ChimeraModule.forward


def forward(self, tensors, compute_loss=True, **kw):
    loss, out = _orig(self, tensors, compute_loss, **kw)
    if not self.training:
        return loss, out
    STATE["step"] += 1
    x = tensors["X"]
    lib = x.sum(-1)
    log_rate = out["log_rate_shared_full"] + out["log_rate_batch"] + (1.0 - out["alpha"].detach()) * out["log_rate_residual"]
    row = {
        "step": STATE["step"], "epoch": int(getattr(self, "current_epoch", -1)),
        "kl_weight": float(getattr(self, "kl_weight", float("nan"))),
        "loss": float(loss.detach()) if torch.isfinite(loss).all() else float("inf"),
        "lib_max": float(lib.max()), "x_max": float(x.max()),
        "z_mean_absmax": _mx(out["z_mean"]), "z_logvar_max": float(out["z_logvar"].detach().max()),
        "z_logvar_min": float(out["z_logvar"].detach().min()), "z_absmax": _mx(out["z"]),
        "log_rate_shared_max": float(out["log_rate_shared_full"].detach().max()),
        "log_rate_batch_absmax": _mx(out["log_rate_batch"]),
        "log_rate_residual_absmax": _mx(out["log_rate_residual"]),
        "log_rate_full_max": float(log_rate.detach().max()),
        "mu_full_max": _mx(out["mu_full"]), "theta_max": _mx(out["theta"]),
        "theta_min": float(out["theta"].detach().min()), "alpha_finite": bool(torch.isfinite(out["alpha"]).all()),
    }
    for k in TERMS:
        v = out.get(k)
        row[k] = (float(v.detach()) if torch.isfinite(v).all() else float("inf")) if isinstance(v, torch.Tensor) else None
    nonfinite = [k for k, v in row.items() if isinstance(v, float) and not torch.isfinite(torch.tensor(v))]
    row["nonfinite"] = "|".join(nonfinite)
    if STATE["header"] is None:
        STATE["header"] = list(row)
    STATE["rows"].append(row)
    if nonfinite and not STATE["dumped"] and STATE["run_dir"] is not None:
        STATE["dumped"] = True
        zl = out["z_logvar"].detach()
        cells = {
            "lib": lib.tolist(), "x_max": x.max(-1).values.tolist(),
            "z_mean_absmax": out["z_mean"].detach().abs().max(-1).values.tolist(),
            "z_logvar_max": zl.max(-1).values.tolist(), "z_absmax": out["z"].detach().abs().max(-1).values.tolist(),
            "log_rate_full_max": log_rate.detach().max(-1).values.tolist(),
            "species": tensors.get("labels", tensors.get("species")).squeeze(-1).tolist(),
        }
        with open(STATE["run_dir"] / "nan_probe_first_batch.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(cells))
            w.writerows(zip(*cells.values()))
        _flush()
    elif STATE["step"] % 50 == 0:
        _flush()
    return loss, out


_module.ChimeraModule.forward = forward


def main() -> int:
    argv = sys.argv[1:]
    name = argv[argv.index("--run-name") + 1]
    STATE["run_dir"] = REPO / "training" / "runs" / name
    from training import run_experiment

    sys.argv = ["run_experiment.py"] + argv
    try:
        return run_experiment.main()
    finally:
        _flush()


if __name__ == "__main__":
    sys.exit(main())
