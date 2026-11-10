"""An upper-only clamp of the sampled log-variance, as an experiment, without a package change (the package option is symmetric, the KL's range [-10, 10]).

A forward hook on z_logvar_linear clamps its output from above at Z_LOGVAR_BOUND. With the VAMP prior (every recipe) this equals an upper-only clamp in the sample: the VAMP KL clamps the
same tensor to [-10, 10] already, so its value and gradient do not change; pseudo-input log-variances are clamped there too. What differs from a sampling-only clamp: the returned z_logvar and
the metric z/mean_learned_var are capped at 10 in these runs. Checked against the package option by check_upper_equiv.py. Training goes through nan_probe.py as in step 3.
    python analysis/tasks/G1_guard/upper_only.py <run_experiment.py arguments>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "tasks" / "T12_2b_debug"))

import nan_probe  # noqa: E402  (wraps ChimeraModule.forward)
from chimera import _module  # noqa: E402

BOUND = _module.Z_LOGVAR_BOUND


def install(lower: float | None = None, upper: float | None = BOUND) -> None:
    orig = _module.ChimeraModule.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        if self.use_vamp_prior is not True:
            raise RuntimeError("upper_only.py assumes the VAMP prior")
        self.z_logvar_linear.register_forward_hook(lambda m, i, o: o.clamp(min=lower, max=upper))

    _module.ChimeraModule.__init__ = init


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--clamp-z-logvar" in argv:
        raise SystemExit("upper_only.py runs with the package option off")
    install()
    run = REPO / "training" / "runs" / argv[argv.index("--run-name") + 1]
    run.mkdir(parents=True, exist_ok=True)
    (run / "g1_mode.json").write_text(json.dumps(dict(mode="upper_only", how="forward hook on z_logvar_linear, clamp(max=10)", package_option=False), indent=1))
    sys.argv = ["nan_probe.py"] + argv
    sys.exit(nan_probe.main())
