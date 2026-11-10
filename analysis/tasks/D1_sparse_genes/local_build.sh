#!/usr/bin/env bash
# Inside an existing allocation: remaining forward margins, then gene tables, four cohorts at a time.
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH PYTHONPATH=$PWD OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 SLURM_CPUS_PER_TASK=3
PY="mamba run --no-capture-output -n chimera_env python"
for c in islet bone_marrow; do [ -f results/analysis/D1/fwd/$c.parquet ] || $PY -m analysis.tasks.D1_sparse_genes.fwd --cohort $c > results/analysis/D1/logs/fwd_$c.log 2>&1 & done
wait
printf '%s\n' liv_hmouse liv_hmonkey liv_hzeb liv_4sp test_hmouse test_hmac test_3sp kidney islet bone_marrow vz_hmouse vz_hmmul vz_primates vz_full | xargs -P 4 -I{} sh -c "$PY -m analysis.tasks.D1_sparse_genes.build {} > results/analysis/D1/logs/build_{}.log 2>&1"
