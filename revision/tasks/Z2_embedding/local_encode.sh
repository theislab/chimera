#!/usr/bin/env bash
# Existing models through encoders, skipping finished ones. liv_hmouse: healthy seeds and Y5 scVI; test_hmouse: healthy seeds and Y5 scVI.
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY="mamba run --no-capture-output -n chimera_env python"
for r in xds-hmouse-iter90-150ep-s42 xds-hmouse-iter90-150ep-s43 xds-hmouse-iter90-150ep-s44 rev-liv_hmouse-s45 rev-liv_hmouse-s46 rev-liv_hmouse-s47 rev-liv_hmouse-s48 rev-liv_hmouse-s49 rev-liv_hmouse-s50 rev-liv_hmouse-s51; do
  $PY -m revision.tasks.Z2_embedding.encode model liv_hmouse "$r" 2>&1 | grep -v -i "warn\|INFO\|^  " | tail -2
done
for s in 42 43 44 45 46; do $PY -m revision.tasks.Z2_embedding.encode model liv_hmouse "scvi:results/revision/Y5/scvi/liv_hmouse_s$s" 2>&1 | grep -v -i "warn\|INFO\|^  " | tail -2; done
for r in testis-hmouse-iter111-pclamp-coc-200ep-s42 rev-test_hmouse-s43 rev-test_hmouse-s44 rev-test_hmouse-s45 rev-test_hmouse-s46; do
  $PY -m revision.tasks.Z2_embedding.encode model test_hmouse "$r" 2>&1 | grep -v -i "warn\|INFO\|^  " | tail -2
done
for s in 42 43 44 45 46; do $PY -m revision.tasks.Z2_embedding.encode model test_hmouse "scvi:results/revision/Y5/scvi/test_hmouse_s$s" 2>&1 | grep -v -i "warn\|INFO\|^  " | tail -2; done
$PY -m revision.tasks.Z2_embedding.diag_encoder 2>&1 | grep -v -i "warn\|INFO" | tail -60
