"""Classical completion baselines (mesh-Laplacian, thin-plate-spline RBF, population
linear-MMSE) evaluated under a DEPLOYMENT LAYOUT of choice -- primarily the v2 fixed
template layout (mode='template'); mode='cluster'/'kmean' reproduces the frozen v1
numbers (parity check).  Port of the frozen v1 scripts
    baselines_analyses/{interp_baseline,lmmse_baseline,fidelity_baselines}.py
with the field assembly, the MFS inverse, the references and the metric definitions
kept verbatim; only the electrode selection (and, for LMMSE, the per-N ridge lambda
re-calibrated on the validation split under the same layout) differ.

Per recording x method x N one record holds BOTH the fidelity metrics of
fidelity_baselines.py (bsp_cc / hsp_cc / hsp_mae vs commercial, hsp_cc_src2 /
hsp_mae_src2 vs the all-valid inverse) AND the downstream AT/RT/ARI MAE of
interp_baseline / lmmse_baseline (at_com, rt_com, ari_com, at_src2, ...), plus the
BSP MAE in learned-model units (bsp_mae_baselines.py convention: g = median SD of
the visible electrodes).

  --calibrate  [--layout template]     per-N ridge lambda on 20 validation recordings
                                       -> generated/classical_template/lmmse_lam_<layout>.json
  --split S --shard k --nshards m      per-recording cache
                                       -> generated/classical_template/cache_<layout>/<split>/<sid>.json
  --summary                            -> generated/classical_template/summary_<layout>.json
                                          + per-recording CSV (interp_baseline columns + fidelity columns)

The covariance of LMMSE (training-set statistic, selection independent) is read from
the frozen v1 generated/lmmse_cov.npz.  CPU only; every path written lies in v2.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse
import csv
import json
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, 'bsp_reconstruction'))
sys.path.insert(0, os.path.join(PROJ, 'inverse_evaluation'))
from paths import V1_RUNS, V1_GENERATED, V2_GENERATED, assert_v2            # noqa: E402
from build_sources import build_context, _to256, CROP                      # noqa: E402
from channel_select import select_visible_channels                         # noqa: E402
from inverse import assemble_solver, solve_from_field, load_commercial     # noqa: E402
from at_rt import extract_sample                                           # noqa: E402
from metrics import map_mae_cc, hsp_metrics                                # noqa: E402
from mesh_io import load_surface, find_patient_files                       # noqa: E402
from cache_references import cache_lookup                                  # noqa: E402
from dataset import collect_split                                          # noqa: E402

warnings.filterwarnings('ignore')

VCS = [4, 8, 16, 32, 64]
METHODS = ['laplacian', 'tps', 'lmmse']
SPLITS = ['test_internal', 'test_external', 'holdout']   # 'holdout' (2026-10-20): patients listed in --holdout_file
LAM_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
OUTDIR = assert_v2(os.path.join(V2_GENERATED, 'classical_template'))
COVF = os.path.join(V1_GENERATED, 'lmmse_cov.npz')          # frozen v1, read-only
LAMF_V1 = os.path.join(V1_GENERATED, 'lmmse_lam.json')      # v1 k-means calibration
MAX_GEOM_CACHE = 4
V1_WIDE = os.path.join(V1_RUNS, 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_'
                       'nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800')

LAYOUTS = {   # layout name -> (mode, cluster_method)
    'template': ('template', 'greedy'),
    'kmeans': ('cluster', 'kmean'),
}


def lam_file(layout):
    return os.path.join(OUTDIR, f'lmmse_lam_{layout}.json')


def _setup():
    """args + keep_idx of the v1 final model (preprocessing / coordinate conventions)."""
    args = json.load(open(os.path.join(V1_WIDE, 'args.json')))
    keep_idx = np.load(os.path.join(V1_WIDE, 'keep_idx.npy')).astype(np.int64)
    return args, keep_idx


def _select(coords, active, vc, layout):
    mode, cm = LAYOUTS[layout]
    vis, info = select_visible_channels(coords, active, vc, mode, cm,
                                        np.random.default_rng(42), return_info=True)
    return vis, int(info.get('n_template_sub', 0))


# ------------------------------------------------------------ verbatim v1 pieces --
def mesh_laplacian(torso_V, torso_F):
    """Graph Laplacian with inverse-edge-length weights (dense, n_t x n_t)."""
    n = torso_V.shape[0]
    W = np.zeros((n, n))
    for tri in torso_F:
        for a in range(3):
            i, j = int(tri[a]), int(tri[(a + 1) % 3])
            w = 1.0 / (np.linalg.norm(torso_V[i] - torso_V[j]) + 1e-9)
            W[i, j] = W[j, i] = max(W[i, j], w)
    return np.diag(W.sum(1)) - W


def laplacian_interp(L, known_idx, known_vals, n_vert):
    u = np.setdiff1d(np.arange(n_vert), known_idx)
    phi = np.zeros((n_vert, known_vals.shape[1]), np.float64)
    phi[known_idx] = known_vals
    A = L[np.ix_(u, u)]
    B = -L[np.ix_(u, known_idx)] @ known_vals
    phi[u] = np.linalg.solve(A + 1e-9 * np.eye(len(u)), B)
    return phi


def tps_interp(pts_known, vals_known, pts_eval):
    from scipy.interpolate import RBFInterpolator
    for kw in (dict(kernel='thin_plate_spline'),
               dict(kernel='thin_plate_spline', smoothing=1e-6),
               dict(kernel='linear')):
        try:
            return RBFInterpolator(pts_known, vals_known, **kw)(pts_eval)
        except Exception:
            continue
    return None


def _predict(cov, prep, visible, lam_rel):
    V = np.where(visible)[0]
    Svv = cov[np.ix_(V, V)]
    lam = lam_rel * float(np.diag(Svv).mean())
    W = cov[:, V] @ np.linalg.solve(Svv + lam * np.eye(len(V)), np.eye(len(V)))
    return (W @ prep[V].astype(np.float64)).astype(np.float32)


def _bsp_cc(pairs):
    ccs = []
    for a_, b_ in pairs:
        if a_.std() > 1e-8 and b_.std() > 1e-8:
            ccs.append(float(np.corrcoef(a_, b_)[0, 1]))
    return (float(np.mean(ccs)) if ccs else float('nan')), len(ccs)


def _hsp(phi, phiref):
    if (phi is None or phiref is None or not np.all(np.isfinite(phi))
            or not np.all(np.isfinite(phiref))):
        return float('nan'), float('nan')
    h = hsp_metrics(phi, phiref)
    return h['cc'], h['mae']


def _g_visible(prep, visible):
    """engine.global_norm: median over the visible electrodes of the unbiased SD."""
    import torch
    g = float(torch.from_numpy(prep[visible]).std(dim=1).median())
    return g if g >= 1e-8 else 1.0


def _r(x, nd=5):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ----------------------------------------------------------------- calibration --
def calibrate(layout, n_recs=20):
    """Per-N ridge lambda minimising held-out BSP MAE (lmmse_baseline.calibrate,
    with the layout's selection); same 20 evenly spaced validation recordings."""
    args, keep_idx = _setup()
    cov = np.load(COVF)['cov']
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets['val'] if r[2]]
    recs = [recs[i] for i in np.linspace(0, len(recs) - 1, n_recs).astype(int)]
    err = {(vc, lr): [] for vc in VCS for lr in LAM_GRID}
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16,
                            mode='cluster', cluster_method='kmean', seed=42)
        if ctx is None:
            continue
        act = ctx['active252'].astype(bool); prep = ctx['prep252']
        sd = prep[act].std(axis=1)
        g = np.median(sd[sd > 1e-8]) if (sd > 1e-8).any() else 1.0
        for vc in VCS:
            vis, _ = _select(ctx['coords'], act, vc, layout)
            held = act & ~vis
            if held.sum() == 0:
                continue
            for lr in LAM_GRID:
                pred = _predict(cov, prep, vis, lr)
                err[(vc, lr)].append(float(np.abs(pred[held] - prep[held]).mean() / max(g, 1e-8)))
        print(f'[cal {layout}] {i+1}/{len(recs)}', flush=True)
    best = {}
    v1 = json.load(open(LAMF_V1))
    for vc in VCS:
        scores = {lr: float(np.mean(err[(vc, lr)])) for lr in LAM_GRID if err[(vc, lr)]}
        lr = min(scores, key=scores.get)
        best[str(vc)] = dict(lam_rel=lr, val_bsp_mae=scores[lr], all_scores=scores,
                             lam_rel_v1_kmeans=v1[str(vc)]['lam_rel'])
        print(f'N={vc:2d}: best lam_rel {lr:g} (v1 k-means {v1[str(vc)]["lam_rel"]:g})  '
              + ' '.join(f'{k:g}:{v:.4f}' for k, v in sorted(scores.items())))
    _atomic_json(lam_file(layout), best)
    print('wrote', lam_file(layout))


# ------------------------------------------------------------- per recording ----
def process_recording(rec, split, layout, keep_idx, args, cov, lam_best, geom_cache):
    sid, ecgpath, torsopath = rec
    sid_safe = sid.replace('/', '_')
    gdir = os.path.dirname(torsopath)
    if gdir not in geom_cache:
        while len(geom_cache) >= MAX_GEOM_CACHE:
            geom_cache.pop(next(iter(geom_cache)))
        solver_t = assemble_solver(gdir)
        tV, tF, tci = load_surface(find_patient_files(gdir)['torso'])
        geom_cache[gdir] = (solver_t, mesh_laplacian(tV, tF), tci)
    (solver, torso_ci, hV, hF), L, chidx = geom_cache[gdir]
    n_vert, n_h = L.shape[0], hV.shape[0]

    ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16,
                        mode='cluster', cluster_method='kmean', seed=42)
    if ctx is None:
        return {'sid': sid, 'failed': 'build_context returned None'}
    prep, coords, active = ctx['prep252'], ctx['coords'], ctx['active252']
    coord_ok = np.abs(coords).sum(1) > 1e-6
    pos252 = np.full(256, -1, np.int64)
    pos252[keep_idx] = np.arange(len(keep_idx))
    v2c = pos252[chidx]
    c2v = np.full(len(keep_idx), -1, np.int64)
    ok = v2c >= 0
    c2v[v2c[ok]] = np.where(ok)[0]
    on_mesh = c2v >= 0

    # references: cached AT/RT + all-valid inverse (v2 cache first, then frozen v1)
    rp = cache_lookup(split, 'refs', sid_safe + '.npz')
    if rp is None:
        return {'sid': sid, 'failed': 'no refs cache entry'}
    with np.load(rp, allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}
    at2, atc = ref['AT2'].astype(float), ref['ATcom'].astype(float)
    rt2, rtc = ref['RT2'].astype(float), ref['RTcom'].astype(float)
    q2, t2 = ref['qrs_good2'].astype(bool), ref['t_good2'].astype(bool)
    qc, tc_ = ref['qrs_goodcom'].astype(bool), ref['t_goodcom'].astype(bool)
    phi2 = np.asarray(ref['phi2'], np.float64) if 'phi2' in ref else None
    tw = (ctx['crop_start'], ctx['crop_start'] + CROP)
    phicom = load_commercial(os.path.splitext(ecgpath)[0] + '.potential', n_h, time_window=tw)

    res = {m: {} for m in METHODS}
    nsub = {}
    for vc in VCS:
        visible, n_sub = _select(coords, active, vc, layout)
        nsub[str(vc)] = n_sub
        g_vis = _g_visible(prep, visible)
        entries = {}      # method -> (bsp_cc, n_cc, bsp_mae, f252, amask)

        kn_c = np.where(visible & on_mesh)[0]
        if len(kn_c) >= 4:
            phiV = laplacian_interp(L, c2v[kn_c], prep[kn_c].astype(np.float64), n_vert)
            f252 = prep.copy().astype(np.float32)
            fill = active & on_mesh & ~visible
            f252[fill] = phiV[c2v[fill]].astype(np.float32)
            cc, n_cc = _bsp_cc((phiV[c2v[c]], prep[c]) for c in np.where(fill)[0])
            mae = float(np.abs(f252[fill] - prep[fill]).mean() / g_vis) if fill.any() else np.nan
            entries['laplacian'] = (cc, n_cc, mae, f252, active & (visible | on_mesh))

        kn = visible & coord_ok
        ev = active & coord_ok & ~visible
        if kn.sum() >= 4 and ev.sum() > 0:
            vals = tps_interp(coords[kn].astype(np.float64), prep[kn].astype(np.float64),
                              coords[ev].astype(np.float64))
            if vals is not None:
                f252 = prep.copy().astype(np.float32)
                f252[ev] = vals.astype(np.float32)
                cc, n_cc = _bsp_cc(zip(vals.astype(np.float32), prep[ev]))
                mae = float(np.abs(f252[ev] - prep[ev]).mean() / g_vis)
                entries['tps'] = (cc, n_cc, mae, f252, active & (visible | coord_ok))

        pred = _predict(cov, prep, visible, lam_best[str(vc)]['lam_rel'])
        f252 = pred.copy()
        f252[visible] = prep[visible]
        held = active & ~visible
        cc, n_cc = _bsp_cc((pred[c], prep[c]) for c in np.where(held)[0])
        mae = float(np.abs(pred[held] - prep[held]).mean() / g_vis) if held.any() else np.nan
        entries['lmmse'] = (cc, n_cc, mae, f252, active)

        for meth, (bcc, n_cc, bmae, f252, amask) in entries.items():
            V256, a256 = _to256(f252, amask, keep_idx)
            phi = solve_from_field(solver, torso_ci, V256, a256)[0]
            hcc, hmae = _hsp(phi, phicom)
            hcc2, hmae2 = _hsp(phi, phi2)
            ent = dict(bsp_cc=_r(bcc), bsp_cc_n=n_cc, bsp_mae=_r(bmae),
                       hsp_cc=_r(hcc), hsp_mae=_r(hmae, 6),
                       hsp_cc_src2=_r(hcc2), hsp_mae_src2=_r(hmae2, 6))
            if phi is not None and np.all(np.isfinite(phi)):
                r = extract_sample(phi)
                if r is not None:
                    AT, RT = r['AT'].astype(float), r['RT'].astype(float)
                    ent.update(at_com=_r(map_mae_cc(AT, atc, qc)[0], 6),
                               at_src2=_r(map_mae_cc(AT, at2, q2)[0], 6),
                               rt_com=_r(map_mae_cc(RT, rtc, tc_)[0], 6),
                               rt_src2=_r(map_mae_cc(RT, rt2, t2)[0], 6),
                               ari_com=_r(map_mae_cc(RT - AT, rtc - atc, tc_)[0], 6),
                               ari_src2=_r(map_mae_cc(RT - AT, rt2 - at2, t2)[0], 6))
                    if int(np.asarray(tc_, bool).sum()) < 3:    # T-gate rule (2026-10-20): RT/ARI undefined vs the commercial reference
                        ent.update(rt_com=None, ari_com=None)
                    if int(np.asarray(t2, bool).sum()) < 3:
                        ent.update(rt_src2=None, ari_src2=None)
            res[meth][str(vc)] = ent
    return {'sid': sid, 'layout': layout, 'has_commercial': phicom is not None,
            'has_src2': phi2 is not None, 'n_template_sub': nsub, 'results': res}


def _holdout(holdout_file):
    return set(l.strip() for l in open(holdout_file) if l.strip()) if holdout_file else None


def fit_cov(out, holdout_file=None, t_stride=4):
    """LMMSE covariance (port of the frozen v1 lmmse_baseline.fit, T_STRIDE=4) on the training recordings, optionally
    without the patients of --holdout_file (so that the hold-out split is evaluated with a covariance that has never
    seen those patients; 2026-10-20)."""
    args, keep_idx = _setup()
    C = len(keep_idx)
    buckets, _ = collect_split(seed=42, holdout=_holdout(holdout_file))
    recs = [r for r in buckets['train'] if r[2]]
    S = np.zeros((C, C)); N = np.zeros((C, C)); n_used = 0
    for i, rec in enumerate(recs):
        ctx = build_context(rec[1], rec[2], keep_idx, args, vc=16, mode='cluster', cluster_method='kmean', seed=42)
        if ctx is None:
            continue
        act = ctx['active252'].astype(bool)
        x = ctx['prep252'][:, ::t_stride].astype(np.float64)
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
            print(f'[fit_cov] {i+1}/{len(recs)}', flush=True)
    cov = np.where(N > 0, S / np.maximum(N, 1), 0.0)
    np.savez_compressed(assert_v2(out), cov=cov, n_recordings=n_used)
    print(f'wrote {out}  ({n_used} training recordings, mean diag {np.diag(cov).mean():.4f})', flush=True)


def run_shard(split, layout, shard, nshards, n_max, retry_failed, holdout_file=None, cov_file=None):
    args, keep_idx = _setup()
    cov = np.load(cov_file or COVF)['cov']
    lam_best = json.load(open(lam_file(layout)))
    cache_dir = assert_v2(os.path.join(OUTDIR, f'cache_{layout}', split))
    os.makedirs(cache_dir, exist_ok=True)
    hold = _holdout(holdout_file)
    if split == 'holdout' and not hold:
        raise SystemExit('split=holdout needs --holdout_file')
    buckets, _ = collect_split(seed=42, holdout=hold)
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    recs = [r for r in buckets[split] if r[2] and _dedup_ok(r[0])][:n_max][shard::nshards]
    geom_cache = {}
    n_done = n_skip = n_fail = 0
    for i, rec in enumerate(recs):
        sid = rec[0]
        cpath = os.path.join(cache_dir, sid.replace('/', '_') + '.json')
        if os.path.exists(cpath):
            try:
                prev = json.load(open(cpath))
            except Exception:
                prev = None
            if prev is not None and ('results' in prev or ('failed' in prev and not retry_failed)):
                n_skip += 1
                continue
        try:
            out = process_recording(rec, split, layout, keep_idx, args, cov, lam_best, geom_cache)
        except Exception as e:
            out = {'sid': sid, 'failed': f'{type(e).__name__}: {e}'}
        if 'failed' in out:
            n_fail += 1
            print(f'[{split} {layout} s{shard}] rec#{i} FAILED: {out["failed"]}', flush=True)
        else:
            n_done += 1
        _atomic_json(cpath, out)
        if (n_done + n_fail) % 5 == 0:
            print(f'[{split} {layout} s{shard}] {i+1}/{len(recs)} (done {n_done}, skip {n_skip}, fail {n_fail})', flush=True)
    print(f'[{split} {layout} s{shard}] COMPLETE: done {n_done}, skipped {n_skip}, failed {n_fail} of {len(recs)}', flush=True)


# ---------------------------------------------------------------------- summary --
KEYS = ['bsp_cc', 'bsp_mae', 'hsp_cc', 'hsp_mae', 'hsp_cc_src2', 'hsp_mae_src2',
        'at_com', 'rt_com', 'ari_com', 'at_src2', 'rt_src2', 'ari_src2']


def summarize(layout):
    summary = {'meta': dict(layout=layout, selection=LAYOUTS[layout], vcs=VCS, methods=METHODS,
                            lam=json.load(open(lam_file(layout))) if os.path.exists(lam_file(layout)) else None,
                            definitions='verbatim v1: fidelity_baselines (bsp_cc/hsp_*), interp_baseline & '
                                        'lmmse_baseline (at/rt/ari MAE on reference-gated vertices, ms), '
                                        'bsp_mae_baselines (bsp_mae in units of the visible-electrode g)',
                            aggregation='cohort mean over recordings of finite per-recording values'),
               'splits': {}}
    csv_rows = []
    for split in SPLITS:
        cache_dir = os.path.join(OUTDIR, f'cache_{layout}', split)
        files = sorted(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else []
        acc = {(m, vc): {k: [] for k in KEYS} for m in METHODS for vc in VCS}
        nsub = {str(vc): [] for vc in VCS}
        n_ok = n_fail = 0; failed = []
        for fn in files:
            if not fn.endswith('.json'):
                continue
            d = json.load(open(os.path.join(cache_dir, fn)))
            if 'failed' in d:
                n_fail += 1; failed.append(d['failed']); continue
            n_ok += 1
            for vc in VCS:
                nsub[str(vc)].append(d.get('n_template_sub', {}).get(str(vc), 0))
            for m in METHODS:
                for vc in VCS:
                    e = d['results'].get(m, {}).get(str(vc))
                    if not e:
                        continue
                    csv_rows.append(dict(split=split, sid=d['sid'], vc=vc, method=m,
                                         **{k: e.get(k) for k in KEYS}))
                    for k in KEYS:
                        if e.get(k) is not None:
                            acc[(m, vc)][k].append(e[k])
        tab = {}
        for m in METHODS:
            tab[m] = {}
            for vc in VCS:
                cell = {}
                for k, v in acc[(m, vc)].items():
                    cell[k] = round(float(np.mean(v)), 5) if v else None
                    cell[k + '_n'] = len(v)
                tab[m][str(vc)] = cell
        summary['splits'][split] = dict(n_ok=n_ok, n_failed=n_fail, failed=failed, methods=tab,
                                        n_template_sub_mean={k: (round(float(np.mean(v)), 3) if v else None)
                                                             for k, v in nsub.items()})
    out_json = os.path.join(OUTDIR, f'summary_{layout}.json')
    _atomic_json(out_json, summary)
    out_csv = os.path.join(OUTDIR, f'percase_{layout}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'sid', 'vc', 'method'] + KEYS)
        w.writeheader(); w.writerows(csv_rows)
    print('wrote', out_json, 'and', out_csv, f'({len(csv_rows)} rows)')
    for split in SPLITS:
        s = summary['splits'][split]
        print(f"== {split} [{layout}]  ok {s['n_ok']}  failed {s['n_failed']}  "
              f"template subs {s['n_template_sub_mean']} ==")
        print(f"{'method':10s} {'N':>3s} {'at_com':>7s} {'rt_com':>7s} {'ari_com':>7s} "
              f"{'bsp_cc':>7s} {'hsp_cc':>7s} {'cc_src2':>7s} {'bsp_mae':>7s} {'n':>4s}")
        for m in METHODS:
            for vc in VCS:
                e = s['methods'][m][str(vc)]
                f_ = lambda k: (e[k] if e.get(k) is not None else float('nan'))
                print(f"{m:10s} {vc:3d} {f_('at_com'):7.2f} {f_('rt_com'):7.2f} {f_('ari_com'):7.2f} "
                      f"{f_('bsp_cc'):7.4f} {f_('hsp_cc'):7.4f} {f_('hsp_cc_src2'):7.4f} "
                      f"{f_('bsp_mae'):7.4f} {e.get('at_com_n', 0):4d}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--layout', default='template', choices=sorted(LAYOUTS))
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--n_recs', type=int, default=20)
    p.add_argument('--split', choices=SPLITS)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--n', type=int, default=10**9)
    p.add_argument('--retry_failed', action='store_true')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--holdout_file', default=None, help='patient folders of the hold-out role (one per line); required for --split holdout')
    p.add_argument('--cov', default=None, help='alternative LMMSE covariance npz (default: frozen v1 lmmse_cov.npz)')
    p.add_argument('--fit_cov', default=None, metavar='OUT_NPZ', help='fit the LMMSE covariance on the training recordings (minus --holdout_file) and write it')
    a = p.parse_args()
    if a.fit_cov:
        fit_cov(a.fit_cov, a.holdout_file)
    if a.calibrate:
        calibrate(a.layout, a.n_recs)
    if a.split:
        run_shard(a.split, a.layout, a.shard, a.nshards, a.n, a.retry_failed, a.holdout_file, a.cov)
    if a.summary:
        summarize(a.layout)
