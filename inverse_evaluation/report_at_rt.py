"""
Post-training downstream AT/RT report for a trained FixdElectrodsN_Recon model.

Runs the downstream HSP evaluation (eval_downstream) on the given test splits,
using the model's own VC config (read from args.json) and the precomputed
reference cache (cache_references.py), then writes at_rt_results.json into the
run dir and prints a focused summary.

Headline framing: source 1 (VC-only baseline) vs source 3 (recon-whole) vs
source 4 (recon-valid) — both model reconstructions — each scored against
source 2 (all-valid) and the commercial .potential.
`report()` is called automatically at the end of train.py (--downstream_eval),
or standalone:  python report_at_rt.py --run_dir <run> [--splits test_internal test_external]
                                       [--deploy_region all|front|back] [--out_json ...]

Deployment REGION resolution: explicit `deploy_region` > args.json['select_region']
(a region-trained model) > 'all'. It is recorded as 'deploy_region' in the JSON.
NOTE: the default out_json is <run_dir>/at_rt_results.json regardless of region --
pass --out_json when deploying a full-torso model on one region so the run's
main report is not overwritten.
"""
import os, sys, json, argparse, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_downstream import evaluate, COMPARE, REFS

# _mae = spatial MEAN over vertices (per recording); _mdae = spatial MEDIAN.
# At the cohort level each is summarised by mean/std/median (see _agg), so all
# four two-stage combinations are available: e.g. AT_mae_mean = mean-mean,
# AT_mdae_median = median-median.
METRICS = ['success', 'pot_cc', 'scale_ratio',
           'hsp_mae', 'hsp_mae_md', 'hsp_rmse', 'hsp_rmse_md', 'hsp_cc', 'hsp_cc_md',
           'AT_mae', 'AT_mdae', 'AT_cc',
           'RT_mae', 'RT_mdae', 'RT_cc',
           'RT_mae_all', 'RT_mdae_all', 'RT_cc_all',
           'ARI_mae', 'ARI_mdae', 'ARI_cc',
           'AT_mae_allv', 'AT_mdae_allv',
           'RT_mae_allv', 'RT_mdae_allv',
           'ARI_mae_allv', 'ARI_mdae_allv',
           'ref_qrs_frac', 'ref_t_frac', 'src_qrs_frac', 'src_t_frac']
# methods to print, in order: VC baseline, then the two model reconstructions
HIGHLIGHT = ['src1_vc', 'src3_recon_whole', 'src4_recon_valid']


def _agg(rows, s, r):
    """Aggregate per-recording rows -> cohort mean / std / median for each metric,
    stored under explicit keys '<metric>_mean' / '<metric>_std' / '<metric>_median'
    so the record is never ambiguous about which statistic a number is. Combined
    with the spatial mean (_mae) vs median (_mdae) per-recording summaries, this
    exposes all four two-stage aggregations plus the cohort SD."""
    sub = [x for x in rows if x['source'] == s and x['ref'] == r]
    out = {'n': len(sub)}
    for m in METRICS:
        vals = [x.get(m) for x in sub if isinstance(x.get(m), (int, float)) and np.isfinite(x.get(m))]
        arr = np.asarray(vals, float)
        out[f"{m}_mean"] = float(arr.mean()) if vals else None
        out[f"{m}_std"] = float(arr.std(ddof=1)) if len(vals) > 1 else (0.0 if vals else None)
        out[f"{m}_median"] = float(np.median(arr)) if vals else None
    return out


def report(run_dir, splits=('test_internal', 'test_external'), n=None, out_json=None,
           deploy_mode=None, deploy_cluster_method=None, deploy_region=None):
    args = json.load(open(os.path.join(run_dir, 'args.json')))
    vc, seed = args['vc'], args.get('seed', 42)
    tmode, tcm = args['select_mode'], args['cluster_method']
    # DEPLOY/test selection: explicit arg > val override in args.json > train selection.
    # So a model trained greedy but validated/deployed kmean is scored under kmean.
    mode = deploy_mode or args.get('val_select_mode') or tmode
    cm = deploy_cluster_method or args.get('val_cluster_method') or tcm
    # DEPLOY region: explicit arg > the region the model was trained with > 'all'.
    train_region = args.get('select_region') or 'all'
    region = deploy_region or train_region
    _lab = lambda m, c: f"{m}-{c}" if m == 'cluster' else m
    results = {'run': os.path.basename(run_dir), 'vc': vc,
               'train_mode': _lab(tmode, tcm), 'deploy_mode': _lab(mode, cm),
               'select_mode': _lab(mode, cm),
               'train_region': train_region, 'deploy_region': region,
               'region_fallback_n_rec': {}, 'splits': {}}

    for split in splits:
        rows = evaluate(run_dir, split=split, n=n, vc=vc, mode=mode,
                        cluster_method=cm, seed=seed, verbose=False, region=region)
        results['splits'][split] = {f"{s}__vs__{r}": _agg(rows, s, r)
                                    for s in COMPARE for r in REFS}
        # recordings whose region ran short of vc and needed out-of-region electrodes
        results['region_fallback_n_rec'][split] = len(
            {x['sid'] for x in rows if x.get('n_region_fallback', 0)})

    # a region deployment that differs from the training region gets its own
    # file by default, so the paper's at_rt_results.json is never clobbered
    if out_json is None:
        out_json = os.path.join(run_dir, 'at_rt_results.json' if region == train_region
                                else f'at_rt_results_reg-{region}.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # ---- printed summary: HSP potential accuracy + AT/RT/ARI (src1 vs src3/src4) ----
    print("\n" + "=" * 78)
    print(f"DOWNSTREAM HSP + AT/RT/ARI REPORT  vc={vc}  "
          f"train={results['train_mode']}  deploy={results['deploy_mode']}"
          + (f"  region={region}" if region != 'all' else ""))
    print("  (src1 = VC-only baseline; src3 = recon-whole vest, src4 = recon-valid "
          "subset; both model reconstructions)")
    for split, agg in results['splits'].items():
        print(f"\n[{split}]  HSP potential (CC higher / MAE lower better) & AT/RT/ARI error to "
              f"reference (ms, lower better; RT T-gated; ARI=RT-AT)")
        print(f"  cells = mean-mean +/- SD (median-median) over recordings")
        print(f"  {'method':18s} {'ref':13s} {'succ':>4s} {'HSP-CC':>7s} {'HSP-MAE':>8s} "
              f"{'AT mm+/-sd(MdMd)':>19s} {'RT mm+/-sd(MdMd)':>19s} {'ARI mm+/-sd(MdMd)':>20s}")
        for s in HIGHLIGHT:
            for r in REFS:
                a = agg.get(f"{s}__vs__{r}")
                if not a or not a['n']:
                    continue
                pc = f"{a['hsp_cc_mean']:.3f}" if a.get('hsp_cc_mean') is not None else '  --'
                hm = f"{a['hsp_mae_mean']:.3f}" if a.get('hsp_mae_mean') is not None else '  --'
                # p = 'AT'/'RT'/'ARI'; mean-mean +/- SD from _mae; median-median from _mdae
                def cell(p):
                    mm, sd, md = a.get(f'{p}_mae_mean'), a.get(f'{p}_mae_std'), a.get(f'{p}_mdae_median')
                    return f"{mm:.1f}+/-{sd:.1f} ({md:.1f})" if mm is not None else '--'
                print(f"  {s:18s} {r:13s} {a['success_mean']:4.2f} {pc:>7s} {hm:>8s} "
                      f"{cell('AT'):>19s} {cell('RT'):>19s} {cell('ARI'):>20s}")
    print("=" * 78)
    print(f"[downstream] wrote {out_json}")
    return results


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--splits', nargs='+', default=['test_internal', 'test_external'])
    p.add_argument('--n', type=int, default=None)
    p.add_argument('--deploy_region', default=None, choices=['all', 'front', 'back'],
                   help="deployment torso half; default: args.json['select_region'] or 'all'")
    p.add_argument('--out_json', default=None,
                   help='output JSON (default <run_dir>/at_rt_results.json)')
    a = p.parse_args()
    report(a.run_dir, splits=tuple(a.splits), n=a.n, out_json=a.out_json,
           deploy_region=a.deploy_region)
