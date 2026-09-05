"""Per-recording PAIRED significance tests on the train x deploy matrix, for
source 4 (model reconstruction). Independently re-reads the raw out_matrix CSVs
(a cross-check on the JSON aggregation) and pairs by recording id.

  TRAIN effect : greedy vs kmean, holding DEPLOY + split + ref fixed
                 (same recordings, same visible channels) -> is one model better?
  DEPLOY effect: greedy vs kmean, holding TRAIN + split + ref fixed
                 -> which electrode selection to deploy?

Wilcoxon signed-rank on the per-recording metric differences. Reports median of
each arm, median paired delta, and p. Negative delta = first arm lower (better)
for MAE metrics.
"""
import os, csv
import numpy as np
try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_matrix')
SPLITS = ['test_internal', 'test_external']
REFS = ['src2_allvalid', 'commercial']
SRC = 'src4_recon_valid'


def load(train, deploy, split):
    """Return {sid: row} for src4 rows, one dict per ref (keyed sid->row)."""
    f = os.path.join(OUT, f"train-{train}_deploy-{deploy}_{split}.csv")
    rows = []
    if os.path.exists(f):
        rows = list(csv.DictReader(open(f)))
    return rows


def paired(a_rows, b_rows, ref, metric):
    """Return (arr_a, arr_b) over recordings present in BOTH, matching src4/ref,
    with finite metric in both. a_rows/b_rows are full row lists."""
    ai = {r['sid']: r for r in a_rows if r['source'] == SRC and r['ref'] == ref}
    bi = {r['sid']: r for r in b_rows if r['source'] == SRC and r['ref'] == ref}
    xa, xb = [], []
    for sid, ra in ai.items():
        rb = bi.get(sid)
        if rb is None:
            continue
        try:
            va, vb = float(ra[metric]), float(rb[metric])
        except Exception:
            continue
        if np.isfinite(va) and np.isfinite(vb):
            xa.append(va); xb.append(vb)
    return np.array(xa), np.array(xb)


def test(a_rows, b_rows, ref, metric, lower_better=True):
    xa, xb = paired(a_rows, b_rows, ref, metric)
    if len(xa) < 5:
        return None
    d = xa - xb
    p = np.nan
    if HAVE_SCIPY and np.any(d != 0):
        try:
            p = wilcoxon(xa, xb, zero_method='wilcox').pvalue
        except Exception:
            p = np.nan
    win = ('A' if np.median(xa) < np.median(xb) else 'B') if lower_better \
        else ('A' if np.median(xa) > np.median(xb) else 'B')
    return dict(n=len(xa), medA=np.median(xa), medB=np.median(xb),
                meanA=np.mean(xa), meanB=np.mean(xb),
                d_med=np.median(d), d_mean=np.mean(d), p=p, win=win)


def stars(p):
    if not np.isfinite(p):
        return '   '
    return '***' if p < 1e-3 else '** ' if p < 1e-2 else '*  ' if p < 5e-2 else '   '


METRICS = [('AT_mae', True), ('RT_mae', True), ('RT_mae_all', True),
           ('ARI_mae', True), ('pot_cc', False)]


def main():
    print(f"source4 paired tests  (scipy={'yes' if HAVE_SCIPY else 'NO -> p blank'})")
    print("Each arm shown as  MEDIAN[mean]  over recordings.  d = median(A-B) per recording.")
    print("p = Wilcoxon signed-rank on paired per-recording differences (a MEDIAN-based test).")
    print("MAE lower=better, CC higher=better\n")

    for split in SPLITS:
        # ---- TRAIN effect (A=greedy, B=kmean) at each deploy ----
        for deploy in ['greedy', 'kmean']:
            A = load('greedy', deploy, split); B = load('kmean', deploy, split)
            print(f"[{split}] TRAIN greedy(A) vs kmean(B)  | deploy={deploy}")
            for ref in REFS:
                for m, lb in METRICS:
                    r = test(A, B, ref, m, lb)
                    if r:
                        print(f"    {ref:13s} {m:11s} n={r['n']:3d}  "
                              f"greedy={r['medA']:.2f}[{r['meanA']:.2f}]  "
                              f"kmean={r['medB']:.2f}[{r['meanB']:.2f}]  "
                              f"d={r['d_med']:+6.2f}  p={r['p']:.1e} {stars(r['p'])} "
                              f"-> {'kmean' if r['win']=='B' else 'greedy'}")
            print()
        # ---- DEPLOY effect (A=greedy, B=kmean) at each train ----
        for train in ['greedy', 'kmean']:
            A = load(train, 'greedy', split); B = load(train, 'kmean', split)
            print(f"[{split}] DEPLOY greedy(A) vs kmean(B)  | train={train}")
            for ref in REFS:
                for m, lb in METRICS:
                    r = test(A, B, ref, m, lb)
                    if r:
                        print(f"    {ref:13s} {m:11s} n={r['n']:3d}  "
                              f"depG={r['medA']:.2f}[{r['meanA']:.2f}]  "
                              f"depK={r['medB']:.2f}[{r['meanB']:.2f}]  "
                              f"d={r['d_med']:+6.2f}  p={r['p']:.1e} {stars(r['p'])} "
                              f"-> {'kmean' if r['win']=='B' else 'greedy'}")
            print()


if __name__ == '__main__':
    main()
