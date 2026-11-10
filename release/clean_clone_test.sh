#!/usr/bin/env bash
# Clean-clone test: can the results be reproduced from the release alone?
#
# The test takes only what a reader gets: a fresh `git clone` of the code, an environment built
# from the lock file, and the staged Zenodo bundles. It unpacks the bundles into the clone and
# runs the checks there, while the authors' working tree is hidden from the test processes. A
# check that passes therefore depends on nothing that is not released.
#
# Usage (from anywhere; run the steps in this order, or `all`):
#   bash release/clean_clone_test.sh [options] <step> [<step> ...]
#
# Steps:
#   clone     git clone of the code into <work>/chimera (refuses to overwrite; see --fresh)
#   env       build the environment <env> from analysis/envs/chimera_env.linux-64.lock.txt of the clone
#   unpack    verify SHA256SUMS of every staged bundle, then unpack it into the clone
#   smoke     analysis/reproduce_smoke.sh in the clone (tests, tables, one checkpoint, one notebook)
#   extra     release/clean_clone/extra_checks.py in the clone (every run has its training object;
#             one model of each bundle family reloads on CPU and returns its archived alpha)
#   retrain   train one recipe again in the clone and compare it with the archived runs
#             (not part of `all`; about one hour on 16 threads)
#   all       clone env unpack smoke extra
#
# Options (environment variable in brackets):
#   --staging DIR   staging area written by release/zenodo/stage.py   [CHIMERA_ZENODO_STAGING, default ../_zenodo_staging]
#   --work DIR      work folder of the test, disposable                [CHIMERA_CLEAN_CLONE, default ../_clean_clone]
#   --env NAME      name of the test environment                       [default chimera_clean_env]
#   --source SRC    what to clone: a path or URL                       [default: the repository this script is in]
#   --ref REF       branch or tag to clone                             [default: the current branch of the source]
#   --hide DIR      working tree to hide from the test                 [default: the repository this script is in]
#   --no-hide       run without hiding anything (when the system forbids user namespaces)
#   --fresh         with `clone`: delete an existing <work>/chimera first; with `env`: remove <env> first
#   --threads N     threads for the checks and the retrain             [default 4]
#   --cohort C --seed S --gate A|M     the retrain: recipe key of configs/canonical_recipes.tsv, seed, and gate
#                   (A = protein-feature gate, the canonical recipes; M = per-gene gate)   [default test_hmouse 42 A]
#   -h, --help      this text
#
# How the working tree is hidden: each test command runs inside `unshare -Urm`, a private user
# and mount namespace that needs no root rights. Inside it an empty folder is bind-mounted over
# the working tree, so every path into the working tree (also through links) shows an empty
# folder. The test checks that the folder is empty before it starts the command. Outside that
# process nothing changes.
#
# Logs go to <work>/logs/<step>_<date>.log. The exit code is 0 only if every requested step passed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
STAGING="${CHIMERA_ZENODO_STAGING:-$REPO/../_zenodo_staging}"
WORK="${CHIMERA_CLEAN_CLONE:-$REPO/../_clean_clone}"
ENV_NAME=chimera_clean_env
SOURCE="$REPO"; REF=""; HIDE="$REPO"; NO_HIDE=0; FRESH=0; THREADS=4
COHORT=test_hmouse; SEED=42; GATE=A
STEPS=()

usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --staging) STAGING="$2"; shift 2 ;;
    --work)    WORK="$2"; shift 2 ;;
    --env)     ENV_NAME="$2"; shift 2 ;;
    --source)  SOURCE="$2"; shift 2 ;;
    --ref)     REF="$2"; shift 2 ;;
    --hide)    HIDE="$2"; shift 2 ;;
    --no-hide) NO_HIDE=1; shift ;;
    --fresh)   FRESH=1; shift ;;
    --threads) THREADS="$2"; shift 2 ;;
    --cohort)  COHORT="$2"; shift 2 ;;
    --seed)    SEED="$2"; shift 2 ;;
    --gate)    GATE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    all)       STEPS+=(clone env unpack smoke extra); shift ;;
    clone|env|unpack|smoke|extra|retrain) STEPS+=("$1"); shift ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ ${#STEPS[@]} -gt 0 ]] || { usage; exit 2; }

mkdir -p "$WORK/logs" "$WORK/empty"
chmod 700 "$WORK"
WORK="$(cd "$WORK" && pwd)"
CLONE="$WORK/chimera"
STAMP="$(date +%Y%m%d_%H%M%S)"
CONDA="${CONDA_EXE:-$(command -v mamba || command -v conda)}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg

in_env() { "$CONDA" run --no-capture-output -n "$ENV_NAME" "$@"; }

# hidden <command...>: run the command with the working tree hidden (see the header).
hidden() {
  if [[ $NO_HIDE == 1 || ! -d $HIDE || "$(cd "$HIDE" && pwd -P)" == "$(cd "$CLONE" && pwd -P)" ]]; then
    echo "[working tree not hidden]"
    "$@"
    return
  fi
  unshare -Urm true 2>/dev/null || {
    echo "this system does not allow unprivileged user namespaces (unshare -Urm); rerun with --no-hide" >&2
    return 2
  }
  echo "[working tree hidden: $HIDE]"
  unshare -Urm bash -c 'mount --bind "$0" "$1" && [[ -z $(ls -A "$1") ]] && shift && exec "$@"' \
    "$WORK/empty" "$HIDE" "$@"
}

need_clone() { [[ -d $CLONE/.git ]] || { echo "no clone at $CLONE; run the step 'clone' first" >&2; return 2; }; }

step_clone() {
  if [[ -e $CLONE ]]; then
    [[ $FRESH == 1 ]] || { echo "$CLONE exists; pass --fresh to delete it and clone again" >&2; return 2; }
    rm -rf "$CLONE"
  fi
  git clone ${REF:+--branch "$REF"} "$SOURCE" "$CLONE"
  git -C "$CLONE" log --oneline -1
}

step_env() {
  need_clone
  if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    [[ $FRESH == 1 ]] || { echo "environment $ENV_NAME exists; it is used as it is (pass --fresh to rebuild it)"; return 0; }
    "$CONDA" env remove -y -n "$ENV_NAME"
  fi
  bash "$CLONE/analysis/envs/build_env_from_lock.sh" "$ENV_NAME"
}

step_unpack() {
  need_clone
  local found=0 bundle
  for sums in "$STAGING"/*/upload/SHA256SUMS; do
    [[ -f $sums ]] || continue
    bundle="$(basename "$(dirname "$(dirname "$sums")")")"
    echo "== $bundle"
    (cd "$(dirname "$sums")" && sha256sum -c --quiet SHA256SUMS && echo "sha256 ok")
    cat "$(dirname "$sums")/chimera_$bundle".tar.gz* | tar -xzf - -C "$CLONE"
    found=$((found + 1))
  done
  [[ $found -gt 0 ]] || { echo "no staged bundle under $STAGING (run release/zenodo/stage.py first)" >&2; return 2; }
  echo "$found bundle(s) unpacked into $CLONE"
  # Everything the bundles bring is ignored by git, so the clone must still be clean.
  local dirty; dirty="$(git -C "$CLONE" status --short | head -20)"
  [[ -z $dirty ]] || { echo "the clone is not clean after unpacking:"; echo "$dirty"; return 1; }
  echo "git status of the clone: clean"
}

step_smoke() {
  need_clone
  git -C "$CLONE" log --oneline -1
  hidden env -C "$CLONE" CHIMERA_ROOT="$CLONE" RUN_CHECKPOINT=1 RUN_FIGURE=1 \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" bash analysis/reproduce_smoke.sh
}

step_extra() {
  need_clone
  hidden env -C "$CLONE" CHIMERA_ROOT="$CLONE" \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" python -I release/clean_clone/extra_checks.py
}

step_retrain() {
  need_clone
  local run="clean-$COHORT-s$SEED"
  [[ $GATE == A ]] || run="clean-$COHORT-m-s$SEED"
  # run_experiment.py exits 1 when an expectation gate fails; the run can still be complete.
  hidden env -C "$CLONE" CHIMERA_ROOT="$CLONE" CONFIG="$GATE" \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" bash analysis/retrain_from_recipe.sh "$COHORT" "$SEED" "$run" cpu \
    || echo "training command exit $? (1 = an expectation gate failed; the comparison below shows the state of the run)"
  hidden env -C "$CLONE" CHIMERA_ROOT="$CLONE" \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" python -I release/clean_clone/compare_retrain.py \
    "$run" --cohort "$COHORT" --gate "$GATE" --out "$WORK/logs/retrain_compare_$run.csv"
}

status=0
for step in "${STEPS[@]}"; do
  log="$WORK/logs/${step}_$STAMP.log"
  echo "######## $step $(date +%T)   log: $log"
  rc=0
  ( "step_$step" ) > >(tee "$log") 2>&1 || rc=$?
  echo "######## $step exit $rc $(date +%T)"
  if [[ $rc != 0 ]]; then status=$rc; break; fi
done
[[ $status == 0 ]] && echo "clean-clone test passed: ${STEPS[*]}" || echo "clean-clone test FAILED at step $step"
exit "$status"
