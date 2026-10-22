"""Aggregate the train x deploy downstream matrix (out_matrix/*.csv) into 2x2
tables: for source 4 (model reconstruction) and source 1 (VC-only baseline), the
downstream HSP/AT/RT metrics for each (training mode) x (deployment selection).

Directly answers: is it best to TRAIN cluster-greedy but DEPLOY cluster-kmean?
"""
import os, csv
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_matrix')
TRAIN = ['greedy', 'kmean']
DEPLOY = ['greedy', 'kmean']


def _finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def load(train, deploy, split):
    f = os.path.join(OUT, f"train-{train}_deploy-{deploy}_{split}.csv")
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []


def cell(rows, source, ref, metric):
    """Return (mean, median) over recordings for this source/ref/metric."""
    v = [float(r[metric]) for r in rows
         if r['source'] == source and r['ref'] == ref and _finite(r.get(metric))]
    if not v:
        return float('nan'), float('nan')
    return float(np.mean(v)), float(np.median(v))


METRICS = [('pot_cc', 'pot-CC', 3), ('AT_mae', 'AT-MAE', 2),
           ('RT_mae', 'RT-MAE(T)', 2), ('RT_mae_all', 'RT-MAE(all)', 2),
           ('ARI_mae', 'ARI-MAE', 2)]


def main():
    print("SOURCE 4 (model reconstruction) downstream, rows=TRAIN mode, cols=DEPLOY selection")
    print("lower AT/RT/ARI-MAE (ms) better; higher pot-CC better")
    print("each cell = MEAN (MEDIAN) over recordings")
    for split in ['test_internal', 'test_external']:
        for ref in ['src2_allvalid', 'commercial']:
            print(f"\n{'='*80}\n[{split}]  vs {ref}       cells = mean (median)")
            for metric, lab, p in METRICS:
                print(f"  {lab:12s}" + "".join(f"deploy-{d:13s}" for d in DEPLOY))
                for train in TRAIN:
                    cells = [cell(load(train, d, split), 'src4_recon_valid', ref, metric) for d in DEPLOY]
                    fmt = (lambda mn, md: f"{mn:.{p}f} ({md:.{p}f})")
                    print(f"    train {train:7s}" + "".join(f"{fmt(mn, md):>19s}" for mn, md in cells))
            # src1 baseline (VC-only inverse; depends only on deploy selection)
            b_at = [cell(load('greedy', d, split), 'src1_vc', ref, 'AT_mae') for d in DEPLOY]
            b_rt = [cell(load('greedy', d, split), 'src1_vc', ref, 'RT_mae') for d in DEPLOY]
            print("  src1 baseline (mean[median]): " +
                  "  ".join(f"deploy-{d}: AT={a[0]:.2f}[{a[1]:.2f}] RT={r[0]:.2f}[{r[1]:.2f}]"
                            for d, a, r in zip(DEPLOY, b_at, b_rt)))


if __name__ == '__main__':
    main()
