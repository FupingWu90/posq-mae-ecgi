"""Rank the four (train x deploy) combos across the FULL new metric set to test
whether 'train greedy, deploy kmean' is still the best / consistent choice.

For source 4 (model reconstruction), per split x reference, computes the
conventional mean-mean (cohort mean of per-recording spatial mean) for each
metric, marks the best combo, and tallies wins. Also flags where the robust
median-median disagrees.
"""
import os, csv
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_matrix')
SPLITS = ['test_internal', 'test_external']
REFS = ['src2_allvalid', 'commercial']
COMBOS = [('greedy', 'greedy'), ('greedy', 'kmean'), ('kmean', 'greedy'), ('kmean', 'kmean')]
CLAB = ['tG/dG', 'tG/dK', 'tK/dG', 'tK/dK']   # train/deploy
# (label, spatial-mean col, spatial-median col, +1 higher-better / -1 lower-better)
METS = [('HSP-CC',  'hsp_cc',       'hsp_cc_md',       +1),
        ('HSP-MAE', 'hsp_mae',      'hsp_mae_md',      -1),
        ('pot_cc',  'pot_cc',       'pot_cc',          +1),
        ('AT ref',  'AT_mae',       'AT_mdae',         -1),
        ('RT ref',  'RT_mae',       'RT_mdae',         -1),
        ('ARI ref', 'ARI_mae',      'ARI_mdae',        -1),
        ('AT all',  'AT_mae_allv',  'AT_mdae_allv',    -1),
        ('RT all',  'RT_mae_allv',  'RT_mdae_allv',    -1),
        ('ARI all', 'ARI_mae_allv', 'ARI_mdae_allv',   -1)]


def col(train, deploy, split, ref, key):
    f = os.path.join(OUT, f"train-{train}_deploy-{deploy}_{split}.csv")
    if not os.path.exists(f):
        return np.array([])
    v = []
    for r in csv.DictReader(open(f)):
        if r['source'] != 'src4_recon_valid' or r['ref'] != ref:
            continue
        try:
            fx = float(r.get(key, ''))
        except (TypeError, ValueError):
            continue
        if np.isfinite(fx):
            v.append(fx)
    return np.asarray(v, float)


def best_idx(vals, sign):
    arr = np.array([v if np.isfinite(v) else (-np.inf * sign) for v in vals])
    return int(np.argmax(sign * arr))


def main():
    tally = {c: 0 for c in CLAB}
    tally_split = {s: {c: 0 for c in CLAB} for s in SPLITS}
    flips = 0
    for split in SPLITS:
        for ref in REFS:
            print(f"\n{'='*72}\n[{split}] vs {ref}   (src4, mean-mean; * = best; [MdMd best] if differs)")
            print(f"  {'metric':8s} " + "".join(f"{c:>10s}" for c in CLAB) + "   best")
            for lab, mk, dk, sign in METS:
                mm = [float(np.mean(col(t, d, split, ref, mk))) if col(t, d, split, ref, mk).size else float('nan')
                      for t, d in COMBOS]
                md = [float(np.median(col(t, d, split, ref, dk))) if col(t, d, split, ref, dk).size else float('nan')
                      for t, d in COMBOS]
                bi = best_idx(mm, sign); bmd = best_idx(md, sign)
                tally[CLAB[bi]] += 1; tally_split[split][CLAB[bi]] += 1
                p = 3 if 'cc' in mk else 2
                cells = "".join((f"*{mm[i]:.{p}f}" if i == bi else f" {mm[i]:.{p}f}").rjust(10)
                                for i in range(4))
                extra = "" if bmd == bi else f"  [MdMd:{CLAB[bmd]}]"
                if bmd != bi:
                    flips += 1
                print(f"  {lab:8s} {cells}   {CLAB[bi]}{extra}")
    print(f"\n{'='*72}\nWIN TALLY (mean-mean best across {len(METS)} metrics x 2 refs x 2 splits = {len(METS)*4}):")
    for c in CLAB:
        print(f"  {c:8s}: {tally[c]:2d}   (internal {tally_split['test_internal'][c]}, "
              f"external {tally_split['test_external'][c]})")
    print(f"  median-median disagreed with mean-mean on {flips} of {len(METS)*4} cells")


if __name__ == '__main__':
    main()
