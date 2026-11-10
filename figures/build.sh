#!/usr/bin/env bash
# Build the figures of the paper from the panel files in figures/panels/.
#
#   figures/build.sh                  compose what is not current, then merge, crop, manifest, checks
#   figures/build.sh KEY [KEY ...]    compose these figures again, then merge, crop, manifest, checks
#   figures/build.sh --check          write nothing; exit 1 if a composed figure, figures.pdf or the manifest is not current
#   figures/build.sh --strict         also fail on the figures listed in figures/specs/_allow.py
#   figures/build.sh --export DIR     after the build, copy figures.pdf, manifest.tsv and manifest.json to DIR
#                                     (the figures/ folder of the manuscript)
#
# Needs the environment chimera_figures_env (figures/environment.yml) and pdflatex. Nothing is recomputed: the build
# reads panel files only. Panels are drawn by their own scripts (figures/README.md, "Changing a figure").
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
ENV_NAME="${CHIMERA_FIGURES_ENV:-chimera_figures_env}"

EXPORT=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --export) EXPORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

command -v pdflatex >/dev/null || { echo "pdflatex not found" >&2; exit 2; }
if command -v mamba >/dev/null && mamba env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
  RUN=(mamba run --no-capture-output -n "$ENV_NAME")
else
  RUN=()   # assume the environment is already active
fi

cd "$REPO"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 "${RUN[@]}" python -m figures.build ${ARGS[@]+"${ARGS[@]}"}

if [ -n "$EXPORT" ]; then
  OUT="${CHIMERA_FIGURES_OUT:-$HERE}"
  mkdir -p "$EXPORT"
  cp "$OUT/figures.pdf" "$OUT/manifest.tsv" "$OUT/manifest.json" "$EXPORT/"
  echo "exported figures.pdf, manifest.tsv, manifest.json to $EXPORT"
fi
