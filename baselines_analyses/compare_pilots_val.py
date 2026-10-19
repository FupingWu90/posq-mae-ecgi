"""Paired per-recording comparison of pilots on the VALIDATION recordings (commercial and all-valid
references) under ONE deployment layout.
  python baselines_analyses/compare_pilots_val.py [--layout template|kmeans] REF ID [ID ...]     (differences = ID - REF)
  -> results/val_percase/paired_<REF>_vs_<IDs>[_kmeans].json ; prints aggregates only."""
import os, sys, json, argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(V2, 'results', 'val_percase')
ap = argparse.ArgumentParser()
ap.add_argument('--layout', default='template', choices=['template', 'kmeans'])
ap.add_argument('ref'); ap.add_argument('ids', nargs='+')
a = ap.parse_args()
ld = lambda i: pd.read_csv(os.path.join(D, f'{i}_vc16_val_{a.layout}.csv'))
ref_id, ids = a.ref, a.ids
R = ld(ref_id)
out = {'layout': a.layout}
for i in ids:
    X = ld(i)
    print(f'== {i} minus {ref_id} (validation, {a.layout} layout, n recordings paired)')
    for refname in ('commercial', 'src2_allvalid'):
        a_ = X[(X.source == 'src4_recon_valid') & (X.ref == refname)].set_index('sid'); b = R[(R.source == 'src4_recon_valid') & (R.ref == refname)].set_index('sid')
        common = a_.index.intersection(b.index); a_, b = a_.loc[common], b.loc[common]
        pat = pd.Series([s.split('__')[1] for s in common], index=common)
        txt = []
        for m in ('AT_mae', 'RT_mae', 'ARI_mae', 'hsp_cc', 'AT_cc'):
            x = pd.to_numeric(a_[m], errors='coerce'); y = pd.to_numeric(b[m], errors='coerce'); ok = x.notna() & y.notna(); d = (x - y)[ok]
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            per_pat = d.groupby(pat[ok]).mean()
            sign = int((per_pat < 0).sum()) if m.endswith('mae') else int((per_pat > 0).sum())
            out[f'{i}|{refname}|{m}'] = dict(mean=float(x[ok].mean()), ref=float(y[ok].mean()), med_diff=float(d.median()), p=float(p), n=int(ok.sum()), patients_better=sign, n_patients=int(per_pat.size))
            txt.append(f"{m} {x[ok].mean():.3f} vs {y[ok].mean():.3f} (median diff {d.median():+.3f}, p={p:.2g}, better in {sign}/{per_pat.size} patients)")
        print(f'   vs {refname}: ' + ' | '.join(txt))
suffix = '' if a.layout == 'template' else f'_{a.layout}'
json.dump(out, open(os.path.join(D, f'paired_{ref_id}_vs_' + '_'.join(ids) + suffix + '.json'), 'w'), indent=1)
