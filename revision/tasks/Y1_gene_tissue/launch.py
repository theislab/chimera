"""Run a Y1 module under ``python -I`` (no cwd or script directory on sys.path; downloaded h5ads are data only).

    python -I <repo>/revision/tasks/Y1_gene_tissue/launch.py <module> [args]
"""

import os
import runpy
import sys
from pathlib import Path

if os.environ.get("Y1_WORKERS"):  # bootstrap draws run in threads; one BLAS thread each
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
mod = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_module(mod, run_name="__main__")
