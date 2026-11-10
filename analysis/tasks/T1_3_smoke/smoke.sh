#!/usr/bin/env bash
# T1.3: lock file, imports, tests, checkpoint load. Calibration runs are a separate array.
source analysis/slurm/_body.sh
OUT=results/analysis/T1_3; mkdir -p "$OUT"
L=analysis/envs/chimera_env.linux-64.lock.txt
{ echo "--index-url https://download.pytorch.org/whl/cpu"
  echo "--extra-index-url https://pypi.org/simple"
  echo "-f https://data.pyg.org/whl/torch-2.11.0+cpu.html"
  mamba run -n chimera_env pip freeze --exclude-editable; } > "$L"
mamba run --no-capture-output -n chimera_env python -c "import chimera, scvi, captum, esm, gseapy, harmonypy, statsmodels, pyarrow; print('imports OK')" > "$OUT/imports.txt" 2>&1
mamba run --no-capture-output -n chimera_env python -m pytest -q tests > "$OUT/pytest.txt" 2>&1; echo "pytest rc=$?" >> "$OUT/pytest.txt"
mamba run --no-capture-output -n chimera_env python analysis/tasks/T1_3_smoke/checkpoints.py "$OUT/checkpoint_smoke.tsv" > "$OUT/checkpoints.txt" 2>&1
echo done > "$OUT/DONE_smoke"
