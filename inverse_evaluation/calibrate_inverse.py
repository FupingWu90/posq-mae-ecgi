"""
Calibrate the MFS inverse (alpha_in, alpha_out, lam_floor) on the VAL split for
THIS dataset (cohort B), by comparing source 2 (all-valid MFS HSP) to the commercial
.potential. Computes ALL THREE agreement metrics for every config:
    cc      median per-vertex temporal CC        (higher better)
    at_mae  median AT map MAE vs commercial (ms)  (lower better)
    rt_mae  median RT map MAE vs commercial (ms)  (lower better)
Ranks by --objective (cc | at | rt | atrt) but REPORTS all three for the top
configs, so the CC-vs-AT/RT tradeoff is visible (potential-CC rewards
over-regularization, which blurs the T-wave and hurts RT).

Writes calibration.json (loaded by inverse.py).
  python calibrate_inverse.py --run_dir <run> [--n 15] [--objective atrt]
"""
import os, sys, json, argparse, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction')
sys.path.insert(0, _CODE)

from build_sources import build_context, CROP
from inverse import assemble_solver, solve_from_field, load_commercial
from validate_inverse import per_vertex_temporal_cc
from metrics import map_mae_cc
from at_rt import extract_sample, good_vertex_mask
from dataset import collect_split

CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.json')
ALPHA_IN_GRID = [0.9, 0.95]
ALPHA_OUT_GRID = [1.8, 2.2, 2.6]
LAM_FLOOR_GRID = [1e-3, 3e-3, 1e-2]


def _sample(recs, n):
    recs = [r for r in recs if r[2] is not None]
    if len(recs) <= n:
        return recs
    return [recs[i] for i in np.linspace(0, len(recs) - 1, n).astype(int)]


def _atrt(phi):
    if phi is None or not np.all(np.isfinite(phi)):
        return None
    try:
        return extract_sample(phi)
    except Exception:
        return None


def calibrate(run_dir, n=15, split='val', objective='atrt'):
    args = json.load(open(os.path.join(run_dir, 'args.json')))
    keep_idx = np.load(os.path.join(run_dir, 'keep_idx.npy')).astype(np.int64)
    buckets, _ = collect_split(seed=args.get('seed', 42))
    recs = _sample(buckets[split], n)
    print(f"calibrating on {len(recs)} {split} recordings "
          f"({len(set(r[0].split('__')[1] for r in recs))} patients), objective={objective}")

    ctxs = []
    for sid, ecg, tor in recs:
        try:
            ctx = build_context(ecg, tor, keep_idx, args)
            if ctx is not None:
                ctxs.append((sid, ecg, os.path.dirname(tor), ctx))
        except Exception as e:
            print(f"  skip {sid[:30]}: {e}")

    solver_cache = {}
    com_cache = {}     # sid -> (com_phi, com_atrt, com_good); alpha/lam-independent
    results = {}
    for ai in ALPHA_IN_GRID:
        for ao in ALPHA_OUT_GRID:
            acc = {lf: {'cc': [], 'at': [], 'rt': [], 'rt_all': []} for lf in LAM_FLOOR_GRID}
            for sid, ecg, gd, ctx in ctxs:
                key = (gd, ai, ao)
                if key not in solver_cache:
                    solver_cache[key] = assemble_solver(gd, alpha_in=ai, alpha_out=ao)
                solver, tci, hV, hF = solver_cache[key]
                nh = hV.shape[0]
                tw = (ctx['crop_start'], ctx['crop_start'] + CROP)
                if sid not in com_cache:
                    com = load_commercial(os.path.splitext(ecg)[0] + '.potential', nh, time_window=tw)
                    r = _atrt(com)
                    com_cache[sid] = (com, r,
                                      good_vertex_mask(r['sigavg'], *r['windows']) if r else None)
                com, com_r, com_good = com_cache[sid]
                if com is None or com_r is None:
                    continue
                for lf in LAM_FLOOR_GRID:
                    phi2 = solve_from_field(solver, tci, *ctx['src2_allvalid'], lam_floor=lf)[0]
                    if phi2 is None:
                        continue
                    acc[lf]['cc'].append(float(np.nanmedian(per_vertex_temporal_cc(phi2, com))))
                    r2 = _atrt(phi2)
                    if r2 is not None:
                        acc[lf]['at'].append(map_mae_cc(r2['AT'], com_r['AT'], com_good['qrs'])[0])
                        acc[lf]['rt'].append(map_mae_cc(r2['RT'], com_r['RT'], com_good['t'])[0])
                        acc[lf]['rt_all'].append(map_mae_cc(r2['RT'], com_r['RT'], com_good['qrs'])[0])
            for lf in LAM_FLOOR_GRID:
                med = lambda x: float(np.nanmedian(x)) if x else float('nan')
                results[(ai, ao, lf)] = dict(cc=med(acc[lf]['cc']), at=med(acc[lf]['at']),
                                             rt=med(acc[lf]['rt']), rt_all=med(acc[lf]['rt_all']),
                                             n=len(acc[lf]['at']))
            print(f"  a_in={ai} a_out={ao}: " + "  ".join(
                f"lf{lf:.0e}[cc{results[(ai,ao,lf)]['cc']:.2f} at{results[(ai,ao,lf)]['at']:.1f} "
                f"rt{results[(ai,ao,lf)]['rt']:.1f}/{results[(ai,ao,lf)]['rt_all']:.1f}]"
                for lf in LAM_FLOOR_GRID), flush=True)

    keyfn = {'cc': lambda r: -r['cc'], 'at': lambda r: r['at'],
             'rt': lambda r: r['rt'], 'atrt': lambda r: r['at'] + r['rt']}[objective]
    ranked = sorted((k for k in results if np.isfinite(keyfn(results[k]))),
                    key=lambda k: keyfn(results[k]))
    print(f"\n=== TOP CONFIGS by objective '{objective}' (RT_mae=T-gated, RT_all=un-gated) ===")
    print(f"  {'alpha_in':>8s} {'alpha_out':>9s} {'lam_floor':>9s} | {'CC':>6s} "
          f"{'AT_mae':>7s} {'RT_mae':>7s} {'RT_all':>7s}  n")
    for k in ranked[:8]:
        ai, ao, lf = k; r = results[k]
        print(f"  {ai:8.2f} {ao:9.2f} {lf:9.0e} | {r['cc']:6.3f} {r['at']:7.2f} "
              f"{r['rt']:7.2f} {r['rt_all']:7.2f}  {r['n']}")

    bai, bao, blf = ranked[0]; br = results[ranked[0]]
    cal = {'alpha_in': bai, 'alpha_out': bao, 'lam_floor': blf, 'objective': objective,
           'val_cc': br['cc'], 'val_at_mae': br['at'], 'val_rt_mae': br['rt'],
           'val_rt_mae_all': br['rt_all'], 'n_recordings': br['n'], 'split': split}
    with open(CAL_PATH, 'w') as f:
        json.dump(cal, f, indent=2)
    print(f"\nBEST ({objective}): alpha_in={bai} alpha_out={bao} lam_floor={blf:.0e} "
          f"-> CC={br['cc']:.3f}  AT_mae={br['at']:.2f}ms  RT_mae={br['rt']:.2f}ms")
    print(f"wrote {CAL_PATH}")
    return cal


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--n', type=int, default=15)
    p.add_argument('--split', default='val')
    p.add_argument('--objective', default='atrt', choices=['cc', 'at', 'rt', 'atrt'])
    a = p.parse_args()
    calibrate(a.run_dir, n=a.n, split=a.split, objective=a.objective)
