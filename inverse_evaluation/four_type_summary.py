"""Report the four two-stage aggregations of the downstream metrics, so the
mean/median choice at each level is explicit.

  level 1 (over epicardial vertices, within a recording): _mae/_mn = spatial MEAN,
                                                           _mdae/_md = spatial MEDIAN
  level 2 (over recordings): mean / median (+ SD of the mean)

  mean-mean    = cohort mean   of per-recording spatial MEAN  (conventional)
  mean-median  = cohort median of per-recording spatial MEAN
  median-mean  = cohort mean   of per-recording spatial MEDIAN (robust within map)
  median-median= cohort median of per-recording spatial MEDIAN (fully robust)

Covers HSP potential accuracy (MAE/RMSE/CC) and AT/RT/ARI on both the
REFERENCE-gated vertex set and ALL vertices (the two source-independent, fair
sets). Also prints coverage (reference-gate retained fraction + each source's own
valid-beat fraction).
  python four_type_summary.py [train] [deploy]     (default: greedy greedy)
"""
import os, sys, csv
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_matrix')
SPLITS = ['test_internal', 'test_external']
REFS = ['src2_allvalid', 'commercial']

# (label, spatial-mean column, spatial-median column)
HSP = [('HSP-MAE', 'hsp_mae', 'hsp_mae_md'),
       ('HSP-RMSE', 'hsp_rmse', 'hsp_rmse_md'),
       ('HSP-CC', 'hsp_cc', 'hsp_cc_md')]
GATED = [('AT', 'AT_mae', 'AT_mdae'), ('RT', 'RT_mae', 'RT_mdae'),
         ('ARI', 'ARI_mae', 'ARI_mdae')]
ALLV = [('AT', 'AT_mae_allv', 'AT_mdae_allv'), ('RT', 'RT_mae_allv', 'RT_mdae_allv'),
        ('ARI', 'ARI_mae_allv', 'ARI_mdae_allv')]


def load(train, deploy, split, source='src4_recon_valid'):
    f = os.path.join(OUT, f"train-{train}_deploy-{deploy}_{split}.csv")
    if not os.path.exists(f):
        return []
    return [r for r in csv.DictReader(open(f)) if r['source'] == source]


def col(rows, ref, key):
    v = []
    for r in rows:
        if r['ref'] != ref:
            continue
        try:
            fx = float(r.get(key, ''))
        except (TypeError, ValueError):
            continue
        if np.isfinite(fx):
            v.append(fx)
    return np.asarray(v, float)


def four(rows, ref, mean_key, med_key):
    pm = col(rows, ref, mean_key)      # per-recording spatial MEAN
    pd = col(rows, ref, med_key)       # per-recording spatial MEDIAN
    if pm.size == 0:
        return None
    return dict(n=pm.size,
                mm=pm.mean(), mm_sd=pm.std(ddof=1), mMd=np.median(pm),
                Mdm=pd.mean(), Mdm_sd=pd.std(ddof=1), MdMd=np.median(pd))


def block(rows, ref, spec, unit):
    print(f"  {'metric':10s} {'mean-mean±SD':>16s} {'mean-median':>12s} "
          f"{'median-mean±SD':>16s} {'median-median':>13s}   n")
    for lab, mk, dk in spec:
        d = four(rows, ref, mk, dk)
        if d is None:
            continue
        p = 3 if unit == 'cc' else 2
        print(f"  {lab:10s} {d['mm']:8.{p}f}±{d['mm_sd']:<6.{p}f} {d['mMd']:12.{p}f} "
              f"{d['Mdm']:8.{p}f}±{d['Mdm_sd']:<6.{p}f} {d['MdMd']:13.{p}f}  {d['n']:3d}")


def main():
    train = sys.argv[1] if len(sys.argv) > 1 else 'greedy'
    deploy = sys.argv[2] if len(sys.argv) > 2 else 'greedy'
    print(f"DOWNSTREAM FOUR-TYPE SUMMARY — source 4 (model recon), "
          f"train=cluster-{train}, deploy=cluster-{deploy}")
    print("level1 = over vertices (mean/median); level2 = over recordings (mean/median)")
    for split in SPLITS:
        for ref in REFS:
            rows = load(train, deploy, split)
            print(f"\n{'='*76}\n[{split}]  vs {ref}")
            print(" HSP potential accuracy (MAE/RMSE amplitude-sensitive; CC scale-free):")
            block(rows, ref, HSP[:2], 'mae'); block(rows, ref, HSP[2:], 'cc')
            print(" AT/RT/ARI error (ms) on REFERENCE-gated vertices (reliable-beat vertices):")
            block(rows, ref, GATED, 'ms')
            print(" AT/RT/ARI error (ms) on ALL vertices (no exclusion):")
            block(rows, ref, ALLV, 'ms')
            # coverage
            rq, rt = col(rows, ref, 'ref_qrs_frac'), col(rows, ref, 'ref_t_frac')
            s1 = [r for r in load(train, deploy, split, 'src1_vc') if r['ref'] == ref]
            s4 = rows
            sq1 = col(s1, ref, 'src_t_frac'); sq4 = col(s4, ref, 'src_t_frac')
            print(f" coverage (mean frac of vertices): ref QRS-good={rq.mean():.2f} "
                  f"ref T-good={rt.mean():.2f} | own T-good: VC(src1)={sq1.mean():.2f} "
                  f"recon(src4)={sq4.mean():.2f}")
    print("\n conventional: mean-mean±SD.  robust for RT/ARI: median-median.  "
          "fairness: reference-gated & all-vertices sets are source-independent.")


if __name__ == '__main__':
    main()
