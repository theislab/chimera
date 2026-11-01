#!/usr/bin/env bash
# P13 downloads. Usage: fetch_kidney.sh <chimera_repo>
# CELLxGENE collection 9c9d04c4-8899-417f-bb6f-6107dcadf14f (Lake, Jain et al. 2025,
# "Cellular and Spatial Drivers of Unresolved Injury and Functional Decline in the Human Kidney"):
# human snRNA v2.0 (dataset 7ff0197b-d175-49bf-b4fa-150fe0995d93) and mouse snRNA v2.0
# (dataset 9ac3d74d-1d94-4ff9-abf9-fe5abe4bd2a6). Files are written under data/ only.
set -u
C="$1"; F="$C/revision/tasks/T4_1_gold/fetch_item.sh"
D="$C/data/revision/disease_kidney/raw"; M="$D/MANIFEST.tsv"
mkdir -p "$D"
CX=https://datasets.cellxgene.cziscience.com
"$F" KPMP_MOUSE_V2 "$CX/19aab2ef-654a-4047-8f60-3529c041851d.h5ad" "$D/mouse_kidney_v2.h5ad" "$M" &
"$F" KPMP_HUMAN_V2 "$CX/06b8d9c2-0bde-48bd-88d9-730993fc95e2.h5ad" "$D/human_kidney_v2.h5ad" "$M" &
wait
# External anchor E-K3 (plan section 8c): Cippa et al. 2018 JCI Insight 3:e123151, supplementary
# tables S1-S12 (workbook linked from https://insight.jci.org/articles/view/123151/sd/2).
"$F" CIPPA2018_SDT "https://df6sxcketz7bb.cloudfront.net/manuscripts/123000/123151/jci.insight.123151.sdt1-11.xlsx" \
    "$D/cippa2018/jci.insight.123151.sdt1-11.xlsx" "$M"
