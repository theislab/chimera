# T11.8 Van Zyl: class filter sensitivity (rule written before the filtered models are trained)

Cohort checklist item 5, as on kidney and islet: a class enters only with at least 50 cells in each species that has it. Van Zyl build 3 has 10 class x species cells below 50.

**Filtered object** (`build_vanzyl.py --class-min 50 --out data/revision/van_zyl_f50`; human + mouse by `training/build_subset.py` from it).
Removed, 202 cells: B cell M. fascicularis 25; NK/T mouse 40; unresolved corneal mouse 42; corneal endothelium pig 34; myoepithelium mouse 40; neuron human 21.
Kept by exception, reason: V2 and V3 compare species over shared outflow types (in van Zyl's conventional or uveoscleral outflow groups and in all five species): canal endothelium mouse 25; ciliary muscle mouse 30, pig 41; Schwann mouse 27.

**Models.** The recipes in use, unchanged except the object: full cohort T11.2 grid 1, human + mouse T11.2b grid 1. Seeds 42 and 43 (`manifest.tsv`): rev-vz-f50-s<seed>, rev-vz-f50-hmouse-s<seed>. A seed failing health (NaN, collapse, frozen) is replaced by the next seed until 2 pass.

**Decision rule (per dataset; both must hold to keep the current models).** On genes shared by the filtered and current objects:
- A_cross = mean Spearman of Delta-alpha (alpha - prior as anchored) over all pairs (filtered seed, current seed in D);
- A_within = mean Spearman of Delta-alpha over all pairs of current seeds in D.
If A_cross >= A_within on both datasets, the current models stay and the filter is reported as a checked sensitivity. Otherwise every usable Van Zyl dataset is rebuilt and retrained on filtered objects, and those are used. Single-seed pairs on both sides, so a mean over seeds is not compared with a single seed. Spearman of the seed means and the same statistics on the C-residual scale are reported beside and do not decide.

**Timing, stated.** This rule reads only Delta-alpha, no gene list. It was written after V1-V3 on the current models had been computed and read by the analyst, so it cannot be described as preceding the glaucoma readout.
