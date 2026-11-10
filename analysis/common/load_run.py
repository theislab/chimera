"""Load the saved model of any archived run, including per-gene (free) gate fits.

``ChimeraModel.load`` rebuilds the module from its saved init parameters, which hold the canonical
protein-feature gate only; the per-gene correction of a ``--config M`` fit is added after construction
(``ChimeraModel.from_config``), so its saved state carries a ``per_gene_correction`` the rebuilt module does
not have. This helper loads such a model without changing the package: it loads the state without that
entry, then attaches it as the parameter ``from_config`` adds, with the same flag.

    from analysis.common.load_run import load_run
    model = load_run("training/runs/z3-test_hmouse-m-s42", adata)   # adata: the run's training object
    alpha = model.get_alpha()

Canonical-gate runs load exactly as with ``ChimeraModel.load``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

KEY = "per_gene_correction"


def load_run(run_dir: str | Path, adata, accelerator: str = "cpu", **kwargs):
    from chimera import ChimeraModel

    model_dir = Path(run_dir) / "model"
    saved = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    if KEY not in saved["model_state_dict"]:
        return ChimeraModel.load(str(model_dir), adata=adata, accelerator=accelerator, **kwargs)
    correction = saved["model_state_dict"].pop(KEY)
    with tempfile.TemporaryDirectory() as tmp:
        for f in model_dir.iterdir():
            if f.name != "model.pt" and f.is_file():
                shutil.copy2(f, Path(tmp) / f.name)
        torch.save(saved, Path(tmp) / "model.pt")
        model = ChimeraModel.load(tmp, adata=adata, accelerator=accelerator, **kwargs)
    module = model.module
    module.per_gene_correction = torch.nn.Parameter(correction.to(module.conservation_prior.device))
    for p in module.gin.parameters():
        p.requires_grad_(False)
    module._use_per_gene_alpha = True
    return model
