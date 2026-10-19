"""Test-cohort paired comparison of every finished pilot against the base decoder DEC1 (and v1 against DEC1),
both cohorts, both layouts, commercial reference: downstream AT MAE (mean; paired median difference; Wilcoxon p).
Pilot decisions were taken on validation before these were looked at.  -> results/pilots_test_paired.md (aggregates only)."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'baselines_analyses')); sys.path.insert(0, os.path.join(V2, 'training_sweeps'))
from compare_pilots import csv_path
from pilots import PILOTS, flags_for, result

IDS = ['v1'] + [pid for pid in PILOTS if result(pid, flags_for(pid, decide=False)) is not None]
SPLITS = [('test_internal', 'int'), ('test_external', 'ext')]; LAYOUTS = ['template', 'kmeans']


def load(i, split, lay, metric='AT_mae'):
    p = csv_path(i, split, lay)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p); d = d[(d.source == 'src4_recon_valid') & (d.ref == 'commercial')].set_index('sid')
    return pd.to_numeric(d[metric], errors='coerce')


def main():
    md = ['# Pilots on the TEST cohorts: AT MAE vs. commercial (ms), paired against DEC1 (both layouts)', '',
          'cell = mean | median paired difference vs DEC1 (negative = better) | Wilcoxon p.  Decisions were taken on validation first.', '',
          '| ID | int tmpl | int km | ext tmpl | ext km |', '|---|---|---|---|---|']
    out = {}
    base = {(s, l): load('DEC1', s, l) for s, _ in SPLITS for l in LAYOUTS}
    for i in IDS:
        cells = []
        for s, tag in SPLITS:
            for l in LAYOUTS:
                x, b = load(i, s, l), base[(s, l)]
                if x is None or b is None:
                    cells.append('--'); continue
                common = x.index.intersection(b.index); d = (x.loc[common] - b.loc[common]); ok = d.notna(); d = d[ok]
                p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
                out[f'{i}|{s}|{l}'] = dict(mean=float(x.loc[common][ok].mean()), med_diff=float(d.median()), p=p, n=int(ok.sum()))
                cells.append(f"{x.loc[common][ok].mean():.2f} | {d.median():+.2f} | {p:.2g}" if i != 'DEC1' else f"{x.mean():.2f} (base)")
        md.append(f"| {i} | " + ' | '.join(cells) + ' |')
    open(os.path.join(V2, 'results', 'pilots_test_paired.md'), 'w').write('\n'.join(md) + '\n')
    json.dump(out, open(os.path.join(V2, 'results', 'paired_tests', 'pilots_test_vs_DEC1.json'), 'w'), indent=1)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
