#!/usr/bin/env bash
#SBATCH --job-name=c1_esmw
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
# ESM-2 t48 15B weights and regression head (5120 dimensions, the SATURN ESM2 files in data/raw/saturn_esm2).
D=data/analysis/new_usecases/esm/hub/checkpoints
for f in models/esm2_t48_15B_UR50D.pt regression/esm2_t48_15B_UR50D-contact-regression.pt; do
  curl -sS -f -L -C - --retry 5 -o "$D/$(basename $f)" "https://dl.fbaipublicfiles.com/fair-esm/$f" || exit 1
done
ls -la $D
