"""Paired comparison of deployment layouts for the SAME model (v1 final model, N=16, commercial reference).
Reference layout = per-recording k-means on the valid electrodes (the v1 paper's protocol).  Aggregates only."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from paths import V1_GENERATED
DG = os.path.join(V2, 'results', 'diagnostics_20261004')
LAYOUTS = [('fixed template (one layout for all patients)', os.path.join(DG, 'fixed_layout_vc16_{s}.csv'), 'n_region_fallback'),
           ('spread placement from the geometry alone, k-means', os.path.join(DG, 'geomonly_kmean_vc16_{s}.csv'), 'n_template_sub'),
           ('spread placement from the geometry alone, farthest-point', os.path.join(DG, 'geomonly_greedy_vc16_{s}.csv'), 'n_template_sub')]
NAME = {'src4_recon_valid': 'Ours@16', 'src1_vc': 'raw 16-electrode inverse'}
res = {}
for split in ('test_internal', 'test_external'):
    k = pd.read_csv(os.path.join(V1_GENERATED, 'vc_percase', f'vc16_{split}.csv'))
    print(f'== {split}')
    kk = k[(k.source == 'src4_recon_valid') & (k.ref == 'commercial')]
    print(f"   reference: per-recording k-means on valid electrodes | Ours AT {kk.AT_mae.mean():.2f} RT {kk.RT_mae.mean():.2f} HSP CC {kk.hsp_cc.mean():.3f} | "
          f"raw inverse AT {k[(k.source == 'src1_vc') & (k.ref == 'commercial')].AT_mae.mean():.2f}")
    res[split] = {}
    for lab, path, subcol in LAYOUTS:
        p = path.format(s=split)
        if not os.path.exists(p):
            print(f'   {lab}: (not available yet)'); continue
        f = pd.read_csv(p)
        if len(f[(f.source == 'src4_recon_valid') & (f.ref == 'commercial')]) < 200:
            print(f'   {lab}: (incomplete)'); continue
        out = {}
        for src in ('src4_recon_valid', 'src1_vc'):
            a = f[(f.source == src) & (f.ref == 'commercial')].set_index('sid'); b = k[(k.source == src) & (k.ref == 'commercial')].set_index('sid')
            common = a.index.intersection(b.index); a, b = a.loc[common], b.loc[common]
            for m in ('AT_mae', 'RT_mae', 'hsp_cc'):
                x = pd.to_numeric(a[m], errors='coerce').to_numpy(float); y = pd.to_numeric(b[m], errors='coerce').to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y); d = x[ok] - y[ok]
                out[f'{src}|{m}'] = dict(mean=float(x[ok].mean()), ref=float(y[ok].mean()), med_diff=float(np.median(d)),
                                         p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, n=int(ok.sum()))
        s = f[(f.source == 'src4_recon_valid') & (f.ref == 'commercial')]
        sub = pd.to_numeric(s[subcol], errors='coerce') if subcol in s.columns else pd.Series([np.nan])
        bsp = pd.to_numeric(s['bsp_cc'], errors='coerce').mean() if 'bsp_cc' in s.columns else float('nan')
        out['substitutions_mean'] = float(sub.mean()); out['bsp_cc'] = float(bsp)
        res[split][lab] = out
        o = out
        print(f"   {lab}: Ours AT {o['src4_recon_valid|AT_mae']['mean']:.2f} (paired median diff {o['src4_recon_valid|AT_mae']['med_diff']:+.2f}, p={o['src4_recon_valid|AT_mae']['p']:.2g}) | "
              f"RT {o['src4_recon_valid|RT_mae']['mean']:.2f} ({o['src4_recon_valid|RT_mae']['med_diff']:+.2f}, p={o['src4_recon_valid|RT_mae']['p']:.2g}) | "
              f"HSP CC {o['src4_recon_valid|hsp_cc']['mean']:.3f} | BSP CC {bsp:.3f} | raw inverse AT {o['src1_vc|AT_mae']['mean']:.2f} | substituted electrodes per recording {sub.mean():.2f}")
json.dump(res, open(os.path.join(DG, 'compare_layouts_v1_vc16.json'), 'w'), indent=1)
