"""Deployment-selection robustness of the N=16 downstream comparison.

The deployment protocol selects the N visible electrodes with k-means medoids
seeded by RNG seed 42. This script quantifies how the downstream AT/RT results
of EVERY compared method change when that selection seed changes
(seeds 42 / 7 / 123) — inference only, no training.

Methods (all share the SAME per-(recording, seed) visible set, produced by
build_context(seed=S) exactly as in the main pipeline):

  ours       fixedn_ecgi_wide (wt=3-rw100)      src4_recon_valid
  pe_index   same run + '_pe-index' suffix       src4_recon_valid
  mcma       mcma_unet (same naming pattern)     src4_recon_valid
  lmmse      population linear-MMSE (lmmse_baseline._predict, per-N lambda)
  laplacian  mesh-graph Dirichlet interpolation (interp_baseline conventions)
  tps        3-D thin-plate-spline RBF          (interp_baseline conventions)
  vc_only    src1_vc (the N measured electrodes only)

Each assembled field goes through the standard MFS inverse
(inverse.assemble_solver + solve_from_field), AT/RT extraction
(at_rt.extract_sample) and is scored (map_mae_cc, spatial-mean MAE) against
the commercial reference AT/RT on its gates (cache_hsp refs: ATcom/RTcom,
qrs_goodcom/t_goodcom).

Per-recording results are cached idempotently under
generated/seed_robustness/cache/<split>/<sid>_s<seed>.json ; aggregate output
goes to generated/seed_robustness.json plus a printed table with per-seed
cohort means, across-seed range/SD, and the across-seed SD of the paired
difference (ours - baseline) with a ranking-flip check.

Usage:
  python seed_robustness.py [--n 80] [--seeds 42 7 123] [--agg_only]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

from common import GEN, PROJ, run_dir

sys.path.insert(0, os.path.join(PROJ, 'inverse_evaluation'))
sys.path.insert(0, os.path.join(PROJ, 'bsp_reconstruction'))
from build_sources import load_model, build_context, reconstruct_sources, _to256  # noqa: E402
from inverse import assemble_solver, solve_from_field                     # noqa: E402
from at_rt import extract_sample                                          # noqa: E402
from metrics import map_mae_cc                                            # noqa: E402
from cache_references import CACHE_ROOT                                   # noqa: E402
from dataset import collect_split                                         # noqa: E402
from mesh_io import load_surface, find_patient_files                      # noqa: E402
from lmmse_baseline import _predict, COVF, LAMF                           # noqa: E402
from interp_baseline import mesh_laplacian, laplacian_interp, tps_interp  # noqa: E402

warnings.filterwarnings('ignore')

VC = 16
SPLITS = ('test_internal', 'test_external')
METHODS = ('ours', 'pe_index', 'mcma', 'lmmse', 'laplacian', 'tps', 'vc_only')
MODEL_METHODS = ('ours', 'pe_index', 'mcma')
CACHE_DIR = os.path.join(GEN, 'seed_robustness', 'cache')
OUT_JSON = os.path.join(GEN, 'seed_robustness.json')


# ---------------------------------------------------------------- helpers ---
def cache_path(split, sid, seed):
    return os.path.join(CACHE_DIR, split, f"{sid.replace('/', '_')}_s{seed}.json")


def cache_ok(p):
    """Cache file exists and covers every method."""
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
        return set(METHODS) <= set(d.get('methods', {}))
    except Exception:
        return False


def _f(x):
    """NaN -> None for JSON."""
    return None if (x is None or not np.isfinite(x)) else float(x)


def _nan(x):
    return np.nan if x is None else float(x)


def load_models(device):
    """(name -> model), shared keep_idx, shared args (data config)."""
    rd_ours = run_dir(model='fixedn_ecgi_wide', vc=VC, ws='0', wt='3', rw=100)
    dirs = {'ours': rd_ours,
            'pe_index': rd_ours + '_pe-index',
            'mcma': run_dir(model='mcma_unet', vc=VC, ws='0', wt='3', rw=100)}
    models, keep_idx, args = {}, None, None
    for name, rd in dirs.items():
        m, ki, ar = load_model(rd, device)
        models[name] = m
        if keep_idx is None:
            keep_idx, args = ki, ar
        else:
            assert np.array_equal(keep_idx, ki), f'{name}: keep_idx mismatch'
    return models, keep_idx, args


def solve_and_score(solver, tci, field, atc, rtc, qc, tc):
    """field=(V256, a256) -> (at_mae, rt_mae) vs commercial (NaN on failure)."""
    phi = solve_from_field(solver, tci, *field)[0]
    if phi is None or not np.all(np.isfinite(phi)):
        return np.nan, np.nan
    r = extract_sample(phi)
    if r is None:
        return np.nan, np.nan
    AT, RT = r['AT'].astype(float), r['RT'].astype(float)
    return map_mae_cc(AT, atc, qc)[0], map_mae_cc(RT, rtc, tc)[0]


def eval_recording(rec, seed, models, keep_idx, args, cov, lam_rel,
                   solver_cache, device):
    """Evaluate every method for one (recording, seed).
    Returns cache dict or None if the context cannot be built."""
    sid, ecgpath, torsopath = rec
    gdir = os.path.dirname(torsopath)
    if gdir not in solver_cache:
        solver, tci, hV, hF = assemble_solver(gdir)
        tV, tF, chidx = load_surface(find_patient_files(gdir)['torso'])
        solver_cache[gdir] = (solver, tci, mesh_laplacian(tV, tF), chidx)
    solver, tci, L, chidx = solver_cache[gdir]

    ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=VC,
                        mode='cluster', cluster_method='kmean', seed=seed)
    if ctx is None:
        return None
    prep = ctx['prep252']
    coords = ctx['coords']
    active = ctx['active252'].astype(bool)
    visible = ctx['visible252'].astype(bool)
    coord_ok = np.abs(coords).sum(1) > 1e-6

    # torso-mesh channel bookkeeping (same as interp_baseline)
    pos252 = np.full(256, -1, np.int64)
    pos252[keep_idx] = np.arange(len(keep_idx))
    v2c = pos252[chidx]
    c2v = np.full(len(keep_idx), -1, np.int64)
    ok = v2c >= 0
    c2v[v2c[ok]] = np.where(ok)[0]
    on_mesh = c2v >= 0

    # ---- assemble one field per method (SAME visible set) ----
    fields = {}
    for name in MODEL_METHODS:
        fields[name] = reconstruct_sources(models[name], ctx,
                                           device)['src4_recon_valid']

    pred = _predict(cov, prep, visible, lam_rel)
    f252 = pred.copy()
    f252[visible] = prep[visible]                     # keep the measured leads
    fields['lmmse'] = _to256(f252, active, keep_idx)

    kn_c = np.where(visible & on_mesh)[0]
    if len(kn_c) >= 4:
        phiV = laplacian_interp(L, c2v[kn_c], prep[kn_c].astype(np.float64),
                                L.shape[0])
        f252 = prep.copy().astype(np.float32)
        fill = active & on_mesh & ~visible
        f252[fill] = phiV[c2v[fill]].astype(np.float32)
        fields['laplacian'] = _to256(f252, active & (visible | on_mesh), keep_idx)

    kn = visible & coord_ok
    ev = active & coord_ok & ~visible
    if kn.sum() >= 4 and ev.sum() > 0:
        vals = tps_interp(coords[kn].astype(np.float64),
                          prep[kn].astype(np.float64),
                          coords[ev].astype(np.float64))
        if vals is not None:
            f252 = prep.copy().astype(np.float32)
            f252[ev] = vals.astype(np.float32)
            fields['tps'] = _to256(f252, active & (visible | coord_ok), keep_idx)

    fields['vc_only'] = ctx['src1_vc']
    return ctx, fields, solver, tci


def run_split(split, recs, seeds, models, keep_idx, args, cov, lam_rel, device):
    refs_dir = os.path.join(CACHE_ROOT, split, 'refs')
    solver_cache = {}
    n_printed = 0
    t0 = time.time()
    for i, rec in enumerate(recs):
        sid = rec[0]
        todo = [s for s in seeds if not cache_ok(cache_path(split, sid, s))]
        show_overlap = n_printed < 2
        if not todo and not show_overlap:
            continue

        ref_p = os.path.join(refs_dir, sid.replace('/', '_') + '.npz')
        if not os.path.exists(ref_p):
            print(f'[{split}] {sid[:50]} SKIP: no cached reference', flush=True)
            continue
        ref = np.load(ref_p, allow_pickle=False)
        if not all(k in ref for k in ('ATcom', 'RTcom', 'qrs_goodcom', 't_goodcom')):
            print(f'[{split}] {sid[:50]} SKIP: reference lacks commercial AT/RT',
                  flush=True)
            continue
        atc, rtc = ref['ATcom'].astype(float), ref['RTcom'].astype(float)
        qc, tc = ref['qrs_goodcom'].astype(bool), ref['t_goodcom'].astype(bool)

        vis_by_seed = {}
        try:
            for s in seeds:
                cp = cache_path(split, sid, s)
                if cache_ok(cp) and s not in todo:
                    if show_overlap:
                        vis_by_seed[s] = np.array(json.load(open(cp))['visible'])
                    continue
                out = eval_recording(rec, s, models, keep_idx, args, cov,
                                     lam_rel, solver_cache, device)
                if out is None:
                    print(f'[{split}] {sid[:50]} s{s} SKIP: context failed',
                          flush=True)
                    continue
                ctx, fields, solver, tci = out
                res = {}
                for m in METHODS:
                    if m in fields:
                        at, rt = solve_and_score(solver, tci, fields[m],
                                                 atc, rtc, qc, tc)
                    else:
                        at, rt = np.nan, np.nan
                    res[m] = {'at': _f(at), 'rt': _f(rt)}
                vis_idx = np.where(ctx['visible252'])[0]
                vis_by_seed[s] = vis_idx
                d = dict(sid=sid, seed=s, n_vc=int(ctx['n_vc']),
                         n_active=int(ctx['n_active']),
                         visible=[int(v) for v in vis_idx], methods=res)
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                json.dump(d, open(cp, 'w'))
        except Exception as e:
            print(f'[{split}] {sid[:50]} FAILED: {type(e).__name__}: {e}',
                  flush=True)
            continue

        # ---- visible-set overlap sanity print (first 2 recordings) ----
        if show_overlap and len(vis_by_seed) == len(seeds):
            n_printed += 1
            print(f'[{split}] visible-set check  {sid}', flush=True)
            for s in seeds:
                print(f'    s{s:<4d}: {sorted(int(v) for v in vis_by_seed[s])}')
            for a in range(len(seeds)):
                for b in range(a + 1, len(seeds)):
                    sa, sb = seeds[a], seeds[b]
                    ov = len(set(vis_by_seed[sa].tolist())
                             & set(vis_by_seed[sb].tolist()))
                    print(f'    overlap s{sa} vs s{sb}: {ov}/{VC}')
        if (i + 1) % 5 == 0 or i + 1 == len(recs):
            dt = time.time() - t0
            print(f'[{split}] {i+1}/{len(recs)}  ({dt/60:.1f} min)', flush=True)


# ------------------------------------------------------------ aggregation ---
def aggregate(splits, seeds, recs_by_split):
    out = {'config': {'vc': VC, 'seeds': list(seeds),
                      'n_per_split': {s: len(recs_by_split[s]) for s in splits},
                      'selection': 'cluster-kmean medoids (deployment)'},
           'overlap': {}, 'per_seed': {}, 'summary': {}, 'diff': {},
           'degenerate': {}}
    lines = []
    for split in splits:
        sids, data, vis = [], {}, {}
        for rec in recs_by_split[split]:
            sid = rec[0]
            row, vrow, ok = {}, {}, True
            for s in seeds:
                cp = cache_path(split, sid, s)
                if not cache_ok(cp):
                    ok = False
                    break
                d = json.load(open(cp))
                row[s] = d['methods']
                vrow[s] = (d['visible'], d.get('n_vc'))
            if ok:
                sids.append(sid)
                data[sid] = row
                vis[sid] = vrow
        n = len(sids)

        # ---- visible-set overlap statistics over ALL recordings ----
        ov_stats, ident_cnt = {}, {}
        for a in range(len(seeds)):
            for b in range(a + 1, len(seeds)):
                sa, sb = seeds[a], seeds[b]
                ovs = [len(set(vis[sid][sa][0]) & set(vis[sid][sb][0]))
                       for sid in sids]
                key = f's{sa}_vs_s{sb}'
                ov_stats[key] = {'mean': float(np.mean(ovs)),
                                 'min': int(np.min(ovs)),
                                 'max': int(np.max(ovs)),
                                 'n_identical': int(sum(o == VC for o in ovs))}
                ident_cnt[key] = ov_stats[key]['n_identical']
        deg = [sid for sid in sids
               if any(vis[sid][s][1] is not None and vis[sid][s][1] < VC
                      for s in seeds)]
        out['overlap'][split] = ov_stats
        out['degenerate'][split] = {'n_vc_below_16': deg}

        # ---- per-method arrays: (seed -> per-sid metric) ----
        arr = {m: {s: np.array([_nan(data[sid][s][m]['at']) for sid in sids])
                   for s in seeds} for m in METHODS}
        arr_rt = {m: {s: np.array([_nan(data[sid][s][m]['rt']) for sid in sids])
                      for s in seeds} for m in METHODS}

        per_seed, summary = {}, {}
        for m in METHODS:
            entry = {}
            for met, a_ in (('at', arr[m]), ('rt', arr_rt[m])):
                fin = np.all([np.isfinite(a_[s]) for s in seeds], axis=0)
                means = {s: float(a_[s][fin].mean()) if fin.any() else np.nan
                         for s in seeds}
                vals = np.array(list(means.values()))
                entry[met] = {
                    'per_seed': {str(s): _f(means[s]) for s in seeds},
                    'n': int(fin.sum()),
                    'range': _f(vals.max() - vals.min()),
                    'sd': _f(vals.std(ddof=1)),
                }
            per_seed[m] = entry
        out['per_seed'][split] = per_seed

        # ---- paired difference ours - baseline ----
        diff = {}
        for m in METHODS:
            if m == 'ours':
                continue
            entry = {}
            for met, ao, ab in (('at', arr['ours'], arr[m]),
                                ('rt', arr_rt['ours'], arr_rt[m])):
                fin = np.all([np.isfinite(ao[s]) & np.isfinite(ab[s])
                              for s in seeds], axis=0)
                dm = {s: float((ao[s][fin] - ab[s][fin]).mean())
                      if fin.any() else np.nan for s in seeds}
                vals = np.array(list(dm.values()))
                signs = set(np.sign(vals[np.abs(vals) > 1e-12]).tolist())
                entry[met] = {
                    'per_seed': {str(s): _f(dm[s]) for s in seeds},
                    'n': int(fin.sum()),
                    'sd': _f(vals.std(ddof=1)),
                    'range': _f(vals.max() - vals.min()),
                    'flips': bool(len(signs) > 1),
                }
            diff[m] = entry
        out['diff'][split] = diff

        # ------------------------------------------------ printed table ----
        lines.append(f'\n=== {split}  (n={n} recordings with all seeds cached) ===')
        ov_txt = '  '.join(
            "{}: mean {:.1f} [{},{}] identical {}".format(
                k, v['mean'], v['min'], v['max'], v['n_identical'])
            for k, v in ov_stats.items())
        lines.append(f'visible-set overlap across seeds (out of {VC}): {ov_txt}')
        hdr = (f"{'method':<10}" +
               ''.join(f'  AT s{s:<4}' for s in seeds) +
               '   rng    sd  |' +
               ''.join(f'  RT s{s:<4}' for s in seeds) + '   rng    sd')
        lines.append(hdr)
        for m in METHODS:
            e = per_seed[m]
            row = f'{m:<10}'
            for met in ('at', 'rt'):
                for s in seeds:
                    v = e[met]['per_seed'][str(s)]
                    row += f"  {v:7.2f}" if v is not None else '     --'
                rng, sd = e[met]['range'], e[met]['sd']
                row += (f"  {rng:5.2f} {sd:5.2f}" if rng is not None
                        else '    --    --')
                if met == 'at':
                    row += ' |'
            lines.append(row)
        lines.append('--- paired diff (ours - baseline), cohort mean per seed ---')
        for m in METHODS:
            if m == 'ours':
                continue
            e = diff[m]
            row = f'{m:<10}'
            for met in ('at', 'rt'):
                for s in seeds:
                    v = e[met]['per_seed'][str(s)]
                    row += f"  {v:7.2f}" if v is not None else '     --'
                sd = e[met]['sd']
                flip = 'FLIP' if e[met]['flips'] else 'no  '
                row += (f"  sd {sd:5.2f} {flip}" if sd is not None
                        else '  sd    -- ??  ')
                if met == 'at':
                    row += ' |'
            lines.append(row)
        if deg:
            lines.append(f'degenerate selections (n_vc < {VC}): {deg}')
    return out, '\n'.join(lines)


# ------------------------------------------------------------------- main ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=80)
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 7, 123])
    p.add_argument('--splits', nargs='*', default=list(SPLITS))
    p.add_argument('--agg_only', action='store_true',
                   help='skip computation; aggregate existing cache')
    p.add_argument('--shard', type=int, default=0,
                   help='process only recordings [shard::nshards] (cache is '
                        'idempotent, so shards can run as parallel processes; '
                        'aggregate afterwards with --agg_only)')
    p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()

    buckets, _ = collect_split(seed=42)
    recs_by_split = {}
    for split in a.splits:
        recs = [r for r in buckets[split] if r[2]]
        idx = np.linspace(0, len(recs) - 1, a.n).astype(int)
        recs_by_split[split] = [recs[i] for i in idx]

    if not a.agg_only:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {device}', flush=True)
        models, keep_idx, args = load_models(device)
        cov = np.load(COVF)['cov']
        lam_rel = json.load(open(LAMF))[str(VC)]['lam_rel']
        print(f'models loaded; lmmse lam_rel={lam_rel}', flush=True)
        for split in a.splits:
            recs = recs_by_split[split][a.shard::a.nshards]
            run_split(split, recs, a.seeds, models, keep_idx,
                      args, cov, lam_rel, device)
        if a.nshards > 1:
            print(f'shard {a.shard}/{a.nshards} done; aggregate with '
                  f'--agg_only once every shard has finished', flush=True)
            return

    out, table = aggregate(a.splits, a.seeds, recs_by_split)
    json.dump(out, open(OUT_JSON, 'w'), indent=2)
    print(table)
    print(f'\nwrote {OUT_JSON}')


if __name__ == '__main__':
    main()
