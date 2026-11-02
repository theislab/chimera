"""Check on the capacity table: does the encoder at load reproduce final_state z_mean; how large is posterior variance against the spread of z_mean, by cell quantile; does the decoder survive it? python -m revision.tasks.Z2_embedding.check_post"""
import numpy as np, pandas as pd
from revision.tasks.Z2_embedding import common as K, rules as RU, snr as S
rows = []
for c, run in (("liv_4sp", "rev-liv_4sp-s43"), ("liv_4sp", "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42"), ("bone_marrow", "rev-bm2-s46"), ("liv_hmouse", "rev-liv_hmouse-s47"), ("test_hmouse", "rev-test_hmouse-s43")):
    zm, lv, kl = S.chimera_post(c, run)
    zf = K.load_z(RU.RUNS / run / "final_state.npz")
    sd_mean = zm.std(0)
    q = np.quantile(np.exp(0.5 * lv), [0.1, 0.5, 0.9, 0.99], axis=0)
    rows.append(dict(cohort=c, run=run, max_abs_diff_final=float(np.abs(zm - zf).max()), zm_sd_mean=float(sd_mean.mean()), zm_sd_min=float(sd_mean.min()), zm_sd_max=float(sd_mean.max()), post_sd_p10=float(np.median(q[0])), post_sd_p50=float(np.median(q[1])), post_sd_p90=float(np.median(q[2])), post_sd_p99=float(np.median(q[3])), kl=kl))
    print(rows[-1], flush=True)
