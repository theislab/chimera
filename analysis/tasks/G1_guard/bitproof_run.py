"""Step 2: one short fit from a given source tree; records the hash of every RNG state at the end, so RNG consumption can be compared and not only parameters.

scvi.settings.seed is None in chimera_env, so scvi's train/validation split (DataSplitter.setup, np.random.RandomState(seed=None)) is drawn from OS entropy in every process: two fits of
the same tree, seed and node differ from the first batch on (found here, G1). That draw is pinned here, the same in both trees, by setting scvi.settings.seed; nothing in either tree changes.
    python bitproof_run.py <tree> <run_experiment.py arguments>
"""

import hashlib
import json
import os
import random
import sys
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
os.chdir(tree)
sys.path.insert(0, str(tree))
argv = sys.argv[2:]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import scvi  # noqa: E402

import chimera  # noqa: E402

_pin = os.environ.get("G1_PIN_SPLIT_SEED", "0")
PIN = None if _pin == "none" else int(_pin)  # "none": leave the split to the tree under test
if PIN is not None:
    scvi.settings.seed = PIN
from training import run_experiment  # noqa: E402

if not str(Path(chimera.__file__).resolve()).startswith(str(tree)):
    raise SystemExit(f"imported {chimera.__file__}, not {tree}")
sys.argv = ["run_experiment.py"] + argv
try:
    run_experiment.main()
except SystemExit:
    pass
run = argv[argv.index("--run-name") + 1]
rec = dict(tree=str(tree), scvi_split_seed=PIN, chimera_file=chimera.__file__, threads=torch.get_num_threads(),
           torch_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
           numpy_rng=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
           python_rng=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())
(tree / "training" / "runs" / run / "bitproof_rng.json").write_text(json.dumps(rec, indent=1))
print(rec, flush=True)
