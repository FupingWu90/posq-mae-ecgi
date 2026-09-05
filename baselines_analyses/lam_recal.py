"""Lambda-fairness experiment for the VC-only baseline (src1).

The pipeline calibrates (alpha_in, alpha_out, lambda_floor) ONCE on the
all-valid inverse; a reviewer may object that the VC-only baseline is
handicapped by a calibration not suited to few electrodes. This script
re-calibrates the L-curve floor PER electrode count N for the VC-only inverse
itself and re-evaluates it on both test cohorts:

  --calibrate       grid-search lambda_floor per N on the validation split,
                    minimising the VC-only AT+RT MAE vs the commercial
                    reference  -> generated/lam_per_n.json
  --eval --split S [--shard k --nshards m]
                    re-evaluate src1 at every N with its own floor
                    -> generated/lam_recal/<split>_shard<k>.csv

alpha_in/alpha_out stay at the pipeline values (0.9 / 2.2): they define the
source geometry, which is independent of how many electrodes constrain it;
only the regularisation strength is electrode-count dependent.
"""
import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np

from common import GEN, PROJ, run_dir

sys.path.insert(0, os.path.join(PROJ, 'inverse_evaluation'))
sys.path.insert(0, os.path.join(PROJ, 'bsp_reconstruction'))
from build_sources import build_context                                   # noqa: E402
from channel_select import select_visible_channels                        # noqa: E402
from build_sources import _to256                                          # noqa: E402
from inverse import assemble_solver, solve_from_field                     # noqa: E402
from at_rt import extract_sample                                          # noqa: E402
from metrics import map_mae_cc                                            # noqa: E402
from cache_references import CACHE_ROOT                                   # noqa: E402
from dataset import collect_split                                         # noqa: E402

warnings.filterwarnings('ignore')

VCS = [4, 8, 16, 24, 32, 48, 64]
FLOORS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
LAMJ = os.path.join(GEN, 'lam_per_n.json')
OUTDIR = os.path.join(GEN, 'lam_recal')
os.makedirs(OUTDIR, exist_ok=True)


def _setup():
    rd = run_dir(model='fixedn_ecgi_wide', vc=16, ws='0', wt='3', rw=100)
    args = json.load(open(os.path.join(rd, 'args.json')))
    keep_idx = np.load(os.path.join(rd, 'keep_idx.npy')).astype(np.int64)
    return args, keep_idx


def _ref(split, sid):
    p = os.path.join(CACHE_ROOT, split, 'refs', sid.replace('/', '_') + '.npz')
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


def _src1_field(ctx, keep_idx, vc):
    vis = select_visible_channels(ctx['coords'], ctx['active252'], vc,
                                  'cluster', 'kmean', np.random.default_rng(42))
    return _to256(ctx['prep252'], vis, keep_idx)


def calibrate(n_recs=20):
    args, keep_idx = _setup()
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets['val'] if r[2]]
    recs = [recs[i] for i in np.linspace(0, len(recs) - 1, n_recs).astype(int)]

    acc = {(vc, lf): [] for vc in VCS for lf in FLOORS}
    solver_cache = {}
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        ref = _ref('val', sid)
        if ref is None or 'ATcom' not in ref.files:
            continue
        atc, rtc = ref['ATcom'].astype(float), ref['RTcom'].astype(float)
        qc = ref['qrs_goodcom'].astype(bool)
        tc = ref['t_goodcom'].astype(bool)
        gdir = os.path.dirname(torsopath)
        if gdir not in solver_cache:
            solver_cache[gdir] = assemble_solver(gdir)
        solver, tci, hV, hF = solver_cache[gdir]
        ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16,
                            mode='cluster', cluster_method='kmean', seed=42)
        if ctx is None:
            continue
        for vc in VCS:
            V256, a256 = _src1_field(ctx, keep_idx, vc)
            for lf in FLOORS:
                phi = solve_from_field(solver, tci, V256, a256, lam_floor=lf)[0]
                if phi is None or not np.all(np.isfinite(phi)):
                    continue
                r = extract_sample(phi)
                if r is None:
                    continue
                at = map_mae_cc(r['AT'].astype(float), atc, qc)[0]
                rt = map_mae_cc(r['RT'].astype(float), rtc, tc)[0]
                if np.isfinite(at) and np.isfinite(rt):
                    acc[(vc, lf)].append(at + rt)
        print(f'[cal] {i+1}/{len(recs)}', flush=True)

    best = json.load(open(LAMJ)) if os.path.exists(LAMJ) else {}   # merge (per-N extension)
    for vc in VCS:
        scores = {lf: float(np.mean(acc[(vc, lf)])) for lf in FLOORS
                  if acc[(vc, lf)]}
        lf = min(scores, key=scores.get)
        best[str(vc)] = dict(lam_floor=lf, val_at_plus_rt=scores[lf],
                             all_scores=scores, n=len(acc[(vc, lf)]))
        print(f'N={vc:2d}: best floor {lf:g} '
              + ' '.join(f'{k:g}:{v:.1f}' for k, v in sorted(scores.items())))
    json.dump(best, open(LAMJ, 'w'), indent=2)
    print('wrote', LAMJ)


def evaluate(split, shard, nshards, vcs_tag=''):
    args, keep_idx = _setup()
    best = json.load(open(LAMJ))
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets[split] if r[2]][shard::nshards]
    rows = []
    solver_cache = {}
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        try:
            ref = _ref(split, sid)
            if ref is None:
                continue
            atc, rtc = ref['ATcom'].astype(float), ref['RTcom'].astype(float)
            at2, rt2 = ref['AT2'].astype(float), ref['RT2'].astype(float)
            qc, tc = ref['qrs_goodcom'].astype(bool), ref['t_goodcom'].astype(bool)
            q2, t2 = ref['qrs_good2'].astype(bool), ref['t_good2'].astype(bool)
            gdir = os.path.dirname(torsopath)
            if gdir not in solver_cache:
                solver_cache[gdir] = assemble_solver(gdir)
            solver, tci, hV, hF = solver_cache[gdir]
            ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16,
                                mode='cluster', cluster_method='kmean', seed=42)
            if ctx is None:
                continue
            for vc in VCS:
                lf = best[str(vc)]['lam_floor']
                V256, a256 = _src1_field(ctx, keep_idx, vc)
                phi = solve_from_field(solver, tci, V256, a256, lam_floor=lf)[0]
                if phi is None or not np.all(np.isfinite(phi)):
                    continue
                r = extract_sample(phi)
                if r is None:
                    continue
                AT, RT = r['AT'].astype(float), r['RT'].astype(float)
                rows.append(dict(sid=sid, vc=vc, lam_floor=lf,
                                 at_com=map_mae_cc(AT, atc, qc)[0],
                                 rt_com=map_mae_cc(RT, rtc, tc)[0],
                                 at_src2=map_mae_cc(AT, at2, q2)[0],
                                 rt_src2=map_mae_cc(RT, rt2, t2)[0],
                                 ari_com=map_mae_cc(RT - AT, rtc - atc, tc)[0],
                                 ari_src2=map_mae_cc(RT - AT, rt2 - at2, t2)[0]))
            if (i + 1) % 20 == 0:
                print(f'[{split} s{shard}] {i+1}/{len(recs)}', flush=True)
        except Exception as e:
            print(f'[{split} s{shard}] {sid[:40]} FAILED: {e}', flush=True)
    out = os.path.join(OUTDIR, f'{split}_shard{vcs_tag}{shard}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sid', 'vc', 'lam_floor', 'at_com',
                                          'rt_com', 'at_src2', 'rt_src2',
                                          'ari_com', 'ari_src2'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {out} ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--eval', action='store_true')
    p.add_argument('--split')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--n_recs', type=int, default=20)
    p.add_argument('--vcs', type=int, nargs='*', default=None,
                   help='restrict to these N (extension runs); eval output goes to '
                        '<split>_shard<N..>_<k>.csv and calibrate merges into lam_per_n.json')
    a = p.parse_args()
    vcs_tag = ''
    if a.vcs:
        VCS = list(a.vcs)
        vcs_tag = '-'.join(str(v) for v in VCS) + '_'
    if a.calibrate:
        calibrate(a.n_recs)
    if a.eval:
        evaluate(a.split, a.shard, a.nshards, vcs_tag)
