"""Compare the four select_mode models' DOWNSTREAM AT/RT (source 4 = model recon)
against their BSP-selection metric (val MAE), to check whether the downstream
HSP/AT/RT ranking is consistent with the CC/BSP-based model selection.
"""
import os, json, glob
import numpy as np
_PROJ = os.environ.get('ECGI_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS = _PROJ + "/runs/data-cohortA+cohortB_vs_external"
MODELS = ['cluster-greedy', 'cluster-kmean', 'patch', 'random']


def rundir(m):
    h = glob.glob(os.path.join(RUNS, f"fixedn_ecgi_base_vc16_{m}_nm-gl_*"))
    return h[0] if h else None


def load(m):
    d = rundir(m)
    if not d:
        return None
    out = {'dir': d}
    ts = os.path.join(d, 'training_stats.json')
    if os.path.exists(ts):
        out['val_mae'] = json.load(open(ts)).get('val_best_mae')
    ar = os.path.join(d, 'at_rt_results.json')
    out['at_rt'] = json.load(open(ar)) if os.path.exists(ar) else None
    return out


def cell(at_rt, split, source, ref, metric, stat='mean'):
    """Read an aggregated metric from at_rt_results.json. Values are stored under
    '<metric>_mean' / '<metric>_median'; `stat` selects which (default mean).
    Falls back to a legacy bare '<metric>' key if present."""
    if not at_rt:
        return None
    a = at_rt['splits'].get(split, {}).get(f"{source}__vs__{ref}")
    if not a:
        return None
    return a.get(f"{metric}_{stat}", a.get(metric))


def main():
    data = {m: load(m) for m in MODELS}
    print("=" * 96)
    print("MODEL SELECTION CONSISTENCY: BSP val-MAE (selection metric) vs DOWNSTREAM AT/RT (source 4 recon)")
    print("  lower val-MAE / AT-MAE / RT-MAE better;  higher pot-CC better")
    print("=" * 96)
    for split in ['test_internal', 'test_external']:
        for ref in ['src2_allvalid', 'commercial']:
            print(f"\n[{split}]  source4 (model recon)  vs  {ref}   (cells = MEAN over recordings)")
            print(f"  {'model':16s} {'val_MAE':>8s} | {'pot_cc':>7s} {'AT_mae':>7s} "
                  f"{'AT_cc':>6s} {'RT_mae':>7s} {'RT_all':>7s}")
            rows = []
            for m in MODELS:
                d = data.get(m)
                if not d:
                    continue
                vm = d.get('val_mae')
                pc = cell(d['at_rt'], split, 'src4_recon_valid', ref, 'pot_cc')
                am = cell(d['at_rt'], split, 'src4_recon_valid', ref, 'AT_mae')
                ac = cell(d['at_rt'], split, 'src4_recon_valid', ref, 'AT_cc')
                rm = cell(d['at_rt'], split, 'src4_recon_valid', ref, 'RT_mae')
                ra = cell(d['at_rt'], split, 'src4_recon_valid', ref, 'RT_mae_all')
                rows.append((m, vm, pc, am, ac, rm, ra))
                f = lambda x, p=3: (f"{x:.{p}f}" if isinstance(x, (int, float)) else '  --')
                mark = ''
                print(f"  {m:16s} {f(vm,4):>8s} | {f(pc):>7s} {f(am,2):>7s} "
                      f"{f(ac):>6s} {f(rm,2):>7s} {f(ra,2):>7s}{mark}")
            # rank agreement note
            valid = [(m, vm, am) for (m, vm, pc, am, ac, rm, ra) in rows
                     if isinstance(vm, (int, float)) and isinstance(am, (int, float))]
            if len(valid) >= 2:
                best_val = min(valid, key=lambda r: r[1])[0]
                best_at = min(valid, key=lambda r: r[2])[0]
                print(f"    -> best by val-MAE: {best_val};  best by AT-MAE: {best_at}"
                      f"{'  (CONSISTENT)' if best_val == best_at else '  (differs)'}")
    print("=" * 96)


if __name__ == '__main__':
    main()
