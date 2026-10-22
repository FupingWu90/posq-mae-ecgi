"""Paired per-recording comparison between runs on ONE split under ONE deployment layout (commercial and
all-valid references); generalises compare_pilots_val.py to the test cohorts.
  python baselines_analyses/compare_pilots.py --split val|test_internal|test_external [--layout template|kmeans] REF ID [ID ...]
IDs: pilot ids or 'v1'.  val CSVs: results/val_percase/<ID>_vc16_val_<layout>.csv ; test CSVs:
baselines_analyses/generated/percase_candidate/<ID>_vc16_<split>_<layout>.csv (v1: percase_template / v1 vc_percase).
Prints aggregates only; writes results/paired_tests/<split>_<layout>_<REF>_vs_<IDs>.json"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from paths import V1_GENERATED, V2_GENERATED, assert_v2


def csv_path(i, split, layout, vc=16):
    if split == 'val':
        return os.path.join(V2, 'results', 'val_percase', f'{i}_vc{vc}_val_{layout}.csv')
    if i == 'v1':
        return (os.path.join(V2_GENERATED, 'percase_template', f'ours-v1_vc{vc}_{split}.csv') if layout == 'template'
                else os.path.join(V1_GENERATED, 'vc_percase', f'vc{vc}_{split}.csv'))
    return os.path.join(V2_GENERATED, 'percase_candidate', f'{i}_vc{vc}_{split}_{layout}.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val', choices=['val', 'test_internal', 'test_external'])
    ap.add_argument('--layout', default='template', choices=['template', 'kmeans'])
    ap.add_argument('--vc', type=int, default=16)
    ap.add_argument('ref'); ap.add_argument('ids', nargs='+')
    a = ap.parse_args()
    R = pd.read_csv(csv_path(a.ref, a.split, a.layout, a.vc))
    out = {'split': a.split, 'layout': a.layout, 'ref': a.ref}
    for i in a.ids:
        X = pd.read_csv(csv_path(i, a.split, a.layout, a.vc))
        print(f'== {i} minus {a.ref} ({a.split}, {a.layout} layout, N={a.vc}, n recordings paired)')
        for refname in ('commercial', 'src2_allvalid'):
            x_ = X[(X.source == 'src4_recon_valid') & (X.ref == refname)].set_index('sid'); r_ = R[(R.source == 'src4_recon_valid') & (R.ref == refname)].set_index('sid')
            common = x_.index.intersection(r_.index); x_, r_ = x_.loc[common], r_.loc[common]
            pat = pd.Series([s.split('__')[1] for s in common], index=common)
            txt = []
            for m in ('AT_mae', 'RT_mae', 'ARI_mae', 'hsp_cc', 'AT_cc', 'bsp_cc'):
                if m not in x_.columns or m not in r_.columns:
                    continue
                x = pd.to_numeric(x_[m], errors='coerce'); y = pd.to_numeric(r_[m], errors='coerce'); ok = x.notna() & y.notna()
                if ok.sum() < 10:
                    continue
                d = (x - y)[ok]
                p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                per_pat = d.groupby(pat[ok]).mean()
                sign = int((per_pat < 0).sum()) if m.endswith('mae') else int((per_pat > 0).sum())
                out[f'{i}|{refname}|{m}'] = dict(mean=float(x[ok].mean()), ref=float(y[ok].mean()), med_diff=float(d.median()), p=float(p), n=int(ok.sum()),
                                                 patients_better=sign, n_patients=int(per_pat.size))
                txt.append(f"{m} {x[ok].mean():.3f} vs {y[ok].mean():.3f} (median diff {d.median():+.3f}, p={p:.2g}, better in {sign}/{per_pat.size} patients)")
            print(f'   vs {refname}: ' + ' | '.join(txt))
    od = assert_v2(os.path.join(V2, 'results', 'paired_tests')); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f'{a.split}_{a.layout}_vc{a.vc}_{a.ref}_vs_' + '_'.join(a.ids) + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
