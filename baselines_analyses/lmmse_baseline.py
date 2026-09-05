"""Population linear-MMSE (ridge) reconstruction baseline (Lux 1978; Nelwan 2004).

The classical limited-lead estimator: learn, from the TRAINING recordings, the
252x252 spatial covariance of the (preprocessed, per-recording scale-normalised)
body-surface potentials; at deployment reconstruct every channel from the N
visible ones with the ridge-regularised linear MMSE estimator

    x_hat = Sigma[:,V] (Sigma[V,V] + lambda I)^{-1} x_V .

Because the estimator is linear, the per-recording scale cancels at deployment;
the normalisation matters only when POOLING recordings to fit Sigma. Channel
identity is the vest index (fixed-layout population prior). Same k-means
electrode selection, inverse, references and gating as every other baseline.

  --fit                 pooled pairwise-complete covariance over the training
                        split -> generated/lmmse_cov.npz
  --calibrate           per-N ridge lambda (relative to mean diag) on the
                        validation split, minimising held-out BSP MAE
                        -> generated/lmmse_lam.json
  --eval --split S --shard k --nshards m
                        -> generated/interp/<split>_shardlm_<k>.csv
                           (method='lmmse'; same columns as the interpolation
                            shards, so vc_interp_agg/vc_stats pick it up)
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
from build_sources import build_context, _to256                           # noqa: E402
from channel_select import select_visible_channels                        # noqa: E402
from inverse import assemble_solver, solve_from_field                     # noqa: E402
from at_rt import extract_sample                                          # noqa: E402
from metrics import map_mae_cc                                            # noqa: E402
from cache_references import CACHE_ROOT                                   # noqa: E402
from dataset import collect_split                                         # noqa: E402

warnings.filterwarnings('ignore')

VCS = [4, 8, 16, 24, 32, 48, 64, 72]
LAM_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
COVF = os.path.join(GEN, 'lmmse_cov.npz')
LAMF = os.path.join(GEN, 'lmmse_lam.json')
T_STRIDE = 4                      # time downsampling for covariance fitting


def _setup():
    rd = run_dir(model='fixedn_ecgi_wide', vc=16, ws='0', wt='3', rw=100)
    args = json.load(open(os.path.join(rd, 'args.json')))
    keep_idx = np.load(os.path.join(rd, 'keep_idx.npy')).astype(np.int64)
    return args, keep_idx


def _ctx(rec, keep_idx, args):
    sid, ecgpath, torsopath = rec
    return build_context(ecgpath, torsopath, keep_idx, args, vc=16,
                         mode='cluster', cluster_method='kmean', seed=42)


def fit():
    args, keep_idx = _setup()
    C = len(keep_idx)
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets['train'] if r[2]]
    S = np.zeros((C, C)); N = np.zeros((C, C)); n_used = 0
    for i, rec in enumerate(recs):
        ctx = _ctx(rec, keep_idx, args)
        if ctx is None:
            continue
        act = ctx['active252'].astype(bool)
        x = ctx['prep252'][:, ::T_STRIDE].astype(np.float64)
        sd = x[act].std(axis=1)
        g = np.median(sd[sd > 1e-8]) if (sd > 1e-8).any() else 1.0
        x = x / max(g, 1e-8)
        x -= x.mean(axis=1, keepdims=True)
        x[~act] = 0.0
        a = act.astype(np.float64)
        S += x @ x.T
        N += np.outer(a, a) * x.shape[1]
        n_used += 1
        if (i + 1) % 100 == 0:
            print(f'[fit] {i+1}/{len(recs)}', flush=True)
    cov = np.where(N > 0, S / np.maximum(N, 1), 0.0)
    np.savez_compressed(COVF, cov=cov, n_recordings=n_used)
    print(f'wrote {COVF}  ({n_used} recordings, mean diag {np.diag(cov).mean():.4f})')


def _predict(cov, prep, visible, lam_rel):
    """Linear-MMSE prediction of ALL channels from the visible ones (physical
    units; the per-recording scale cancels for a linear estimator)."""
    V = np.where(visible)[0]
    Svv = cov[np.ix_(V, V)]
    lam = lam_rel * float(np.diag(Svv).mean())
    W = cov[:, V] @ np.linalg.solve(Svv + lam * np.eye(len(V)), np.eye(len(V)))
    return (W @ prep[V].astype(np.float64)).astype(np.float32)


def calibrate(n_recs=20):
    args, keep_idx = _setup()
    cov = np.load(COVF)['cov']
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets['val'] if r[2]]
    recs = [recs[i] for i in np.linspace(0, len(recs) - 1, n_recs).astype(int)]
    err = {(vc, lr): [] for vc in VCS for lr in LAM_GRID}
    for i, rec in enumerate(recs):
        ctx = _ctx(rec, keep_idx, args)
        if ctx is None:
            continue
        act = ctx['active252'].astype(bool); prep = ctx['prep252']
        sd = prep[act].std(axis=1)
        g = np.median(sd[sd > 1e-8]) if (sd > 1e-8).any() else 1.0
        for vc in VCS:
            vis = select_visible_channels(ctx['coords'], act, vc, 'cluster',
                                          'kmean', np.random.default_rng(42))
            held = act & ~vis
            if held.sum() == 0:
                continue
            for lr in LAM_GRID:
                pred = _predict(cov, prep, vis, lr)
                err[(vc, lr)].append(
                    float(np.abs(pred[held] - prep[held]).mean() / max(g, 1e-8)))
        print(f'[cal] {i+1}/{len(recs)}', flush=True)
    best = {}
    for vc in VCS:
        scores = {lr: float(np.mean(err[(vc, lr)])) for lr in LAM_GRID
                  if err[(vc, lr)]}
        lr = min(scores, key=scores.get)
        best[str(vc)] = dict(lam_rel=lr, val_bsp_mae=scores[lr], all_scores=scores)
        print(f'N={vc:2d}: best lam_rel {lr:g}  '
              + ' '.join(f'{k:g}:{v:.4f}' for k, v in sorted(scores.items())))
    json.dump(best, open(LAMF, 'w'), indent=2)
    print('wrote', LAMF)


def evaluate(split, shard, nshards):
    args, keep_idx = _setup()
    cov = np.load(COVF)['cov']
    best = json.load(open(LAMF))
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets[split] if r[2]][shard::nshards]
    refs_dir = os.path.join(CACHE_ROOT, split, 'refs')
    rows = []; solver_cache = {}
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        try:
            ref = np.load(os.path.join(refs_dir, sid.replace('/', '_') + '.npz'),
                          allow_pickle=False)
            at2, atc = ref['AT2'].astype(float), ref['ATcom'].astype(float)
            rt2, rtc = ref['RT2'].astype(float), ref['RTcom'].astype(float)
            q2, t2 = ref['qrs_good2'].astype(bool), ref['t_good2'].astype(bool)
            qc, tc_ = ref['qrs_goodcom'].astype(bool), ref['t_goodcom'].astype(bool)
            gdir = os.path.dirname(torsopath)
            if gdir not in solver_cache:
                solver_cache[gdir] = assemble_solver(gdir)
            solver, tci, hV, hF = solver_cache[gdir]
            ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16,
                                mode='cluster', cluster_method='kmean', seed=42)
            if ctx is None:
                continue
            act = ctx['active252'].astype(bool); prep = ctx['prep252']
            sd = prep[act].std(axis=1)
            g = np.median(sd[sd > 1e-8]) if (sd > 1e-8).any() else 1.0
            for vc in VCS:
                vis = select_visible_channels(ctx['coords'], act, vc, 'cluster',
                                              'kmean', np.random.default_rng(42))
                pred = _predict(cov, prep, vis, best[str(vc)]['lam_rel'])
                f252 = pred.copy()
                f252[vis] = prep[vis]                 # keep the measured leads
                held = act & ~vis
                bsp_mae = float(np.abs(pred[held] - prep[held]).mean()
                                / max(g, 1e-8)) if held.any() else np.nan
                ccs = []
                for c in np.where(held)[0]:
                    a_, b_ = pred[c], prep[c]
                    if a_.std() > 1e-8 and b_.std() > 1e-8:
                        ccs.append(float(np.corrcoef(a_, b_)[0, 1]))
                bsp_cc = float(np.mean(ccs)) if ccs else np.nan
                V256, a256 = _to256(f252, act, keep_idx)
                phi = solve_from_field(solver, tci, V256, a256)[0]
                if phi is None or not np.all(np.isfinite(phi)):
                    continue
                r = extract_sample(phi)
                if r is None:
                    continue
                AT, RT = r['AT'].astype(float), r['RT'].astype(float)
                rows.append(dict(sid=sid, vc=vc, method='lmmse',
                                 at_com=map_mae_cc(AT, atc, qc)[0],
                                 at_src2=map_mae_cc(AT, at2, q2)[0],
                                 rt_com=map_mae_cc(RT, rtc, tc_)[0],
                                 rt_src2=map_mae_cc(RT, rt2, t2)[0],
                                 ari_com=map_mae_cc(RT - AT, rtc - atc, tc_)[0],
                                 ari_src2=map_mae_cc(RT - AT, rt2 - at2, t2)[0],
                                 bsp_mae=round(bsp_mae, 5), bsp_cc=round(bsp_cc, 5)))
            if (i + 1) % 10 == 0:
                print(f'[{split} s{shard}] {i+1}/{len(recs)}', flush=True)
        except Exception as e:
            print(f'[{split} s{shard}] {sid[:40]} FAILED: '
                  f'{type(e).__name__}: {e}', flush=True)
    out = os.path.join(GEN, 'interp', f'{split}_shardlm_{shard}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sid', 'vc', 'method', 'at_com',
                                          'at_src2', 'rt_com', 'rt_src2',
                                          'ari_com', 'ari_src2',
                                          'bsp_mae', 'bsp_cc'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {out} ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fit', action='store_true')
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--eval', action='store_true')
    p.add_argument('--split')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--n_recs', type=int, default=20)
    a = p.parse_args()
    if a.fit:
        fit()
    if a.calibrate:
        calibrate(a.n_recs)
    if a.eval:
        evaluate(a.split, a.shard, a.nshards)
