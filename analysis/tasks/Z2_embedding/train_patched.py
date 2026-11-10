"""Chimera run with the encoder reading log counts instead of raw counts. Mechanism test, not a setting: nothing in chimera/ changes; the method is overridden on the class in this process.
python -m analysis.tasks.Z2_embedding.train_patched <raw|log1p|logcp> <run_experiment args>. The mode is written to <run>/z2_input_mode.txt so an analysis can load the model the same way."""
from __future__ import annotations

import sys


def apply_patch(mode: str) -> None:
    """raw: untouched. log1p: log(1 + counts). logcp: log(1 + counts per 10k). Applies to cells and to VAMP pseudo-inputs, which share _encoder_input."""
    if mode == "raw":
        return
    import torch
    from chimera._module import ChimeraModule

    orig = ChimeraModule._encoder_input

    def patched(self, x, species_oh, batch_oh):
        if mode == "log1p":
            x = torch.log1p(x)
        elif mode == "logcp":
            x = torch.log1p(x / x.sum(-1, keepdim=True).clamp(min=1.0) * 1e4)
        else:
            raise ValueError(mode)
        return orig(self, x, species_oh, batch_oh)

    ChimeraModule._encoder_input = patched


def main() -> int:
    mode = sys.argv[1]
    apply_patch(mode)
    sys.argv = ["run_experiment.py"] + sys.argv[2:]
    from training import run_experiment as RE

    rc = RE.main()
    name = sys.argv[sys.argv.index("--run-name") + 1]
    (RE.RUNS_ROOT / name / "z2_input_mode.txt").write_text(mode + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
