"""Deployment-layout robustness of the N = 16 comparison (supplement S8, first table; main-text Robustness
item 3).  v2 port and extension of the frozen v1 seed_robustness.py: inference only, no training.

Every compared method is evaluated on the SAME 80 evenly spaced test recordings per cohort (the v1 subset)
under eight deployment conditions of the 16-electrode layout:
  per-recording k-means protocol   km_s42 (paper) / km_s7 / km_s123       -- selection RNG seed
  fixed-template protocol          tpl_main (paper)                        -- bsp_reconstruction/templates/template_vc16.npz
                                   tpl_s7 / tpl_s123                       -- k-means medoid seed of the template
                                   tpl_halfA / tpl_halfB                   -- template built from either half of the
                                                                              training geometries (disjoint halves)
Methods (same field assembly as the main pipeline; the inverse, references and metrics are the pipeline's):
  ours      PosQ-MAE final model (LOC3)                 stmem   ST-MEM adapted to the vest (frozen v1 index run)
  mcma      MCMA U-Net (frozen v1 run)                  lmmse   population linear-MMSE (per-layout lambda)
  laplacian mesh-graph Dirichlet interpolation          tps     3-D thin-plate spline        vc_only  the 16 electrodes
For the k-means conditions the rows of the methods that were not retrained are copied from the frozen v1 cache
(same recordings, same seeds) whenever the v2 selection reproduces the cached electrode set exactly (parity
check); only PosQ-MAE is computed.  Per-(recording, condition) rows are cached idempotently under
generated/deploy_robustness/cache/<split>/ ; aggregate output: generated/deploy_robustness_v2.json,
tab_deploy_kmeans_v2.tex, tab_deploy_template_v2.tex.  Prints aggregates only.

  python baselines_analyses/deploy_robustness.py --build_templates              # alternative templates + overlap stats
  python baselines_analyses/deploy_robustness.py --shard k --nshards m [--conds ...]   (parallel shards; idempotent)
  python baselines_analyses/deploy_robustness.py --agg_only                     # tables + JSON from the cache
"""
import os, sys, json, glob, argparse, time, warnings
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
sys.path.insert(0, os.path.join(V2, 'baselines_analyses'))
from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
import numpy as np
import torch
from scipy.stats import wilcoxon
from paths import V1_ROOT, V1_RUNS, V1_GENERATED, V2_GENERATED, V2_RUNS, assert_v2
import channel_select as CS
from channel_select import select_visible_channels, region_mask
from build_sources import load_model, build_context, reconstruct_sources, held_out_bsp_cc, _to256
from inverse import assemble_solver, solve_from_field
from at_rt import extract_sample
from metrics import map_mae_cc
from cache_references import cache_lookup
from mesh_io import load_surface, find_patient_files
import dataset as D
from classical_baselines_template import mesh_laplacian, laplacian_interp, tps_interp, _predict, COVF

warnings.filterwarnings('ignore')
VC = 16
SPLITS = ('test_internal', 'test_external')
METHODS = ('ours', 'stmem', 'mcma', 'lmmse', 'laplacian', 'tps', 'vc_only')
MODEL_METHODS = ('ours', 'stmem', 'mcma')
V1_KEY = {'ours': 'ppe_mae_v1', 'pe_index': 'stmem', 'mcma': 'mcma', 'lmmse': 'lmmse', 'laplacian': 'laplacian',
          'tps': 'tps', 'vc_only': 'vc_only'}
KM_CONDS = {'km_s42': 42, 'km_s7': 7, 'km_s123': 123}
TPL_CONDS = ('tpl_main', 'tpl_s7', 'tpl_s123', 'tpl_halfA', 'tpl_halfB')
GROUPS = {'kmeans': list(KM_CONDS), 'template': list(TPL_CONDS)}
OUTD = assert_v2(os.path.join(V2_GENERATED, 'deploy_robustness'))
CACHE = os.path.join(OUTD, 'cache'); TPLD = os.path.join(OUTD, 'templates')
V1_CACHE = os.path.join(V1_GENERATED, 'seed_robustness', 'cache')
OUT_JSON = os.path.join(V2_GENERATED, 'deploy_robustness_v2.json')
STD = 'cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800'
RUNS = {'ours': None, 'stmem': os.path.join(V1_RUNS, f'fixedn_ecgi_wide_vc{VC}_{STD}_pe-index'),
        'mcma': os.path.join(V1_RUNS, f'mcma_unet_vc{VC}_{STD}')}
LABEL = {'ours': 'Ours (PosQ-MAE)', 'stmem': 'ST-MEM', 'mcma': 'MCMA', 'lmmse': 'LMMSE', 'laplacian': 'Laplace',
         'tps': 'TPS', 'vc_only': 'VC-only'}
CLAB = {'km_s42': 's42', 'km_s7': 's7', 'km_s123': 's123', 'tpl_main': 'main', 'tpl_s7': 's7', 'tpl_s123': 's123',
        'tpl_halfA': 'half A', 'tpl_halfB': 'half B'}


def _f(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _nan(x):
    return np.nan if x is None else float(x)


def cache_path(split, sid, cond):
    return os.path.join(CACHE, split, f"{sid.replace('/', '_')}__{cond}.json")


def v1_cache_path(split, sid, seed):
    return os.path.join(V1_CACHE, split, f"{sid.replace('/', '_')}_s{seed}.json")


def recordings(n):
    buckets, _ = D.collect_split(seed=42)
    out = {}
    for split in SPLITS:
        recs = [r for r in buckets[split] if r[2]]
        idx = np.linspace(0, len(recs) - 1, n).astype(int)
        out[split] = [recs[i] for i in idx if _dedup_ok(recs[i][0])]   # the sub-sample is otherwise unchanged (2026-10-21)
    return out


# ----------------------------------------------------------------- templates ---
def build_templates():
    """Alternative fixed layouts (same recipe as bsp_reconstruction/build_templates.py; written under generated)."""
    os.makedirs(TPLD, exist_ok=True)
    b, _ = D.collect_split(42)
    keep = np.where(D.geometry_union(b['train']))[0].astype(np.int64)
    tors = sorted({tp for _, _, tp in b['train'] if tp})
    C = np.stack([D._coords_for(tp)[keep] for tp in tors])
    ok = np.abs(C).sum(-1) > 1e-6

    def mean_geom(sel):
        Cs, oks = C[sel], ok[sel]
        return ((Cs * oks[..., None]).sum(0) / np.maximum(oks.sum(0), 1)[:, None]).astype(np.float32)

    CS._TEMPLATES.pop(VC, None); main = CS.load_template(VC)
    allg = np.arange(len(tors)); perm = np.random.default_rng(0).permutation(len(tors))
    halves = {'tpl_halfA': perm[: len(tors) // 2], 'tpl_halfB': perm[len(tors) // 2:]}
    meta = {'n_train_geometries': len(tors), 'main_idx': main[0].tolist(), 'templates': {}}
    for cond in TPL_CONDS + ('tpl_front', 'tpl_back'):     # the last two: half-torso templates (S8 region table)
        if cond == 'tpl_main':
            idx, mxyz = main
        elif cond in ('tpl_front', 'tpl_back'):
            mxyz = mean_geom(allg)
            vis = select_visible_channels(mxyz, np.ones(len(keep), bool), VC, 'cluster', 'kmean', np.random.default_rng(42),
                                          region='front' if cond == 'tpl_front' else 'back')
            idx = np.where(vis)[0].astype(np.int64)
        else:
            seed = {'tpl_s7': 7, 'tpl_s123': 123}.get(cond, 42)
            mxyz = mean_geom(halves[cond]) if cond in halves else mean_geom(allg)
            vis = select_visible_channels(mxyz, np.ones(len(keep), bool), VC, 'cluster', 'kmean', np.random.default_rng(seed))
            idx = np.where(vis)[0].astype(np.int64)
        np.savez(assert_v2(os.path.join(TPLD, f'{cond}.npz')), idx=idx, mean_xyz=mxyz, keep_idx=keep)
        front = int(region_mask(mxyz, 'front')[idx].sum())
        shared = int(len(set(idx.tolist()) & set(main[0].tolist())))
        # nearest main-template electrode distance for the template's electrodes (in main spacing units)
        nn = np.linalg.norm(main[1][:, None] - main[1][None], axis=-1); np.fill_diagonal(nn, 9); spacing = float(np.median(nn.min(1)))
        d = np.linalg.norm(mxyz[idx][:, None] - main[1][main[0]][None], axis=-1).min(1) / spacing
        meta['templates'][cond] = {'idx252': idx.tolist(), 'n_anterior': front, 'shared_with_main': shared,
                                   'mean_dist_to_nearest_main_electrode_spacings': float(d.mean()),
                                   'n_geometries': int(len(halves[cond])) if cond in halves else len(tors)}
        print(f'{cond:10s}: shared with main {shared:2d}/{VC}, anterior {front:2d}, mean distance to nearest main electrode '
              f'{d.mean():.2f} spacings, geometries {meta["templates"][cond]["n_geometries"]}')
    json.dump(meta, open(os.path.join(TPLD, 'templates.json'), 'w'), indent=1)


def set_template(cond):
    """Install template `cond` as the vc16 layout seen by channel_select.template_select."""
    with np.load(os.path.join(TPLD, f'{cond}.npz')) as z:
        CS._TEMPLATES[VC] = (z['idx'].astype(np.int64), z['mean_xyz'].astype(np.float32))


def context_for(cond, rec, keep_idx, args):
    sid, ecgpath, torsopath = rec
    if cond in KM_CONDS:
        return build_context(ecgpath, torsopath, keep_idx, args, vc=VC, mode='cluster', cluster_method='kmean', seed=KM_CONDS[cond])
    set_template(cond)
    try:
        return build_context(ecgpath, torsopath, keep_idx, args, vc=VC, mode='template', cluster_method='greedy', seed=42)
    finally:
        CS._TEMPLATES.pop(VC, None)        # back to the on-disk main template for everything else


# ---------------------------------------------------------------- evaluation ---
def load_models(device):
    rd = sorted(glob.glob(os.path.join(V2_RUNS, 'data-cohortA+cohortB_vs_external', 'LOC3__*')))[0]
    RUNS['ours'] = rd
    models, keep_idx, args = {}, None, None
    for name in MODEL_METHODS:
        m, ki, ar = load_model(RUNS[name], device)
        models[name] = m
        if keep_idx is None:
            keep_idx, args = ki, ar
        assert np.array_equal(keep_idx, ki), name
    return models, keep_idx, args


def solve_and_score(solver, tci, field, atc, rtc, qc, tc):
    phi = solve_from_field(solver, tci, *field)[0]
    if phi is None or not np.all(np.isfinite(phi)):
        return np.nan, np.nan
    r = extract_sample(phi)
    if r is None:
        return np.nan, np.nan
    AT, RT = r['AT'].astype(float), r['RT'].astype(float)
    return map_mae_cc(AT, atc, qc)[0], map_mae_cc(RT, rtc, tc)[0]


def fields_for(ctx, models, keep_idx, cov, lam_rel, L, chidx, device):
    """method -> (V256, a256) [+ model bsp cc]; identical visible set for every method."""
    prep, coords = ctx['prep252'], ctx['coords']
    active = ctx['active252'].astype(bool); visible = ctx['visible252'].astype(bool)
    coord_ok = np.abs(coords).sum(1) > 1e-6
    fields, bsp = {}, {}
    for name in MODEL_METHODS:
        rec = reconstruct_sources(models[name], ctx, device)
        fields[name] = rec['src4_recon_valid']
        bsp[name] = _f(held_out_bsp_cc(ctx, rec['recon252'])[0])
    pred = _predict(cov, prep, visible, lam_rel)
    f252 = pred.copy(); f252[visible] = prep[visible]
    fields['lmmse'] = _to256(f252, active, keep_idx)
    pos252 = np.full(256, -1, np.int64); pos252[keep_idx] = np.arange(len(keep_idx))
    v2c = pos252[chidx]; c2v = np.full(len(keep_idx), -1, np.int64); okm = v2c >= 0; c2v[v2c[okm]] = np.where(okm)[0]
    on_mesh = c2v >= 0
    kn_c = np.where(visible & on_mesh)[0]
    if len(kn_c) >= 4:
        phiV = laplacian_interp(L, c2v[kn_c], prep[kn_c].astype(np.float64), L.shape[0])
        f252 = prep.copy().astype(np.float32); fill = active & on_mesh & ~visible
        f252[fill] = phiV[c2v[fill]].astype(np.float32)
        fields['laplacian'] = _to256(f252, active & (visible | on_mesh), keep_idx)
    kn = visible & coord_ok; ev = active & coord_ok & ~visible
    if kn.sum() >= 4 and ev.sum() > 0:
        vals = tps_interp(coords[kn].astype(np.float64), prep[kn].astype(np.float64), coords[ev].astype(np.float64))
        if vals is not None:
            f252 = prep.copy().astype(np.float32); f252[ev] = vals.astype(np.float32)
            fields['tps'] = _to256(f252, active & (visible | coord_ok), keep_idx)
    fields['vc_only'] = ctx['src1_vc']
    return fields, bsp


def run_shard(split, recs, conds, models, keep_idx, args, cov, lam, device, reuse_v1=True):
    geom = {}
    t0 = time.time(); n_reused = n_parity_fail = 0
    for i, rec in enumerate(recs):
        sid, ecgpath, torsopath = rec
        todo = [c for c in conds if not os.path.exists(cache_path(split, sid, c))]
        if not todo:
            continue
        rp = cache_lookup(split, 'refs', sid.replace('/', '_') + '.npz')
        if rp is None:
            print(f'[{split}] recording {i}: no cached reference, skipped'); continue
        with np.load(rp, allow_pickle=False) as z:
            if not all(k in z.files for k in ('ATcom', 'RTcom', 'qrs_goodcom', 't_goodcom')):
                print(f'[{split}] recording {i}: reference lacks commercial AT/RT, skipped'); continue
            atc, rtc = z['ATcom'].astype(float), z['RTcom'].astype(float)
            qc, tc = z['qrs_goodcom'].astype(bool), z['t_goodcom'].astype(bool)
        gdir = os.path.dirname(torsopath)
        if gdir not in geom:
            while len(geom) >= 3:
                geom.pop(next(iter(geom)))
            solver, tci, hV, hF = assemble_solver(gdir)
            tV, tF, chidx = load_surface(find_patient_files(gdir)['torso'])
            geom[gdir] = (solver, tci, mesh_laplacian(tV, tF), chidx)
        solver, tci, L, chidx = geom[gdir]
        for cond in todo:
            try:
                ctx = context_for(cond, rec, keep_idx, args)
                if ctx is None:
                    print(f'[{split}] recording {i} {cond}: context failed'); continue
                vis_idx = [int(v) for v in np.where(ctx['visible252'])[0]]
                res, source = {}, 'computed'
                v1row = None
                if reuse_v1 and cond in KM_CONDS:
                    p1 = v1_cache_path(split, sid, KM_CONDS[cond])
                    if os.path.exists(p1):
                        d1 = json.load(open(p1))
                        if sorted(int(v) for v in d1['visible']) == vis_idx:
                            v1row = d1['methods']
                        else:
                            n_parity_fail += 1
                lam_rel = lam['kmeans'] if cond in KM_CONDS else lam['template']
                if v1row is not None:        # only PosQ-MAE is new; the other methods' rows are copied
                    rec_o = reconstruct_sources(models['ours'], ctx, device)
                    at, rt = solve_and_score(solver, tci, rec_o['src4_recon_valid'], atc, rtc, qc, tc)
                    res['ours'] = {'at': _f(at), 'rt': _f(rt), 'bsp': _f(held_out_bsp_cc(ctx, rec_o['recon252'])[0])}
                    for k1, k2 in V1_KEY.items():
                        if k1 in v1row:
                            res[k2] = {'at': _f(_nan(v1row[k1]['at'])), 'rt': _f(_nan(v1row[k1]['rt']))}
                    source = 'v1-cache+ours'; n_reused += 1
                else:
                    fields, bsp = fields_for(ctx, models, keep_idx, cov, lam_rel, L, chidx, device)
                    for m in METHODS:
                        if m in fields:
                            at, rt = solve_and_score(solver, tci, fields[m], atc, rtc, qc, tc)
                        else:
                            at, rt = np.nan, np.nan
                        res[m] = {'at': _f(at), 'rt': _f(rt)}
                        if m in bsp:
                            res[m]['bsp'] = bsp[m]
                d = dict(sid=sid, cond=cond, n_vc=int(ctx['n_vc']), n_active=int(ctx['n_active']), visible=vis_idx,
                         n_template_sub=int(ctx.get('n_template_sub', 0)), source=source, methods=res)
                os.makedirs(os.path.dirname(cache_path(split, sid, cond)), exist_ok=True)
                json.dump(d, open(cache_path(split, sid, cond), 'w'))
            except Exception as e:
                print(f'[{split}] recording {i} {cond} FAILED: {type(e).__name__}: {e}', flush=True)
        if (i + 1) % 5 == 0 or i + 1 == len(recs):
            print(f'[{split}] {i+1}/{len(recs)}  ({(time.time()-t0)/60:.1f} min; v1 rows reused {n_reused}, parity failures {n_parity_fail})', flush=True)


# ---------------------------------------------------------------- aggregation ---
def aggregate(recs_by_split):
    out = {'config': {'vc': VC, 'n_per_split': {s: len(recs_by_split[s]) for s in SPLITS}, 'groups': GROUPS},
           'overlap': {}, 'per_cond': {}, 'diff': {}, 'n_recordings': {}, 'bsp': {}}
    lines = []
    for grp, conds in GROUPS.items():
        for split in SPLITS:
            data, vis = {}, {}
            for rec in recs_by_split[split]:
                sid = rec[0]; row, vrow, ok = {}, {}, True
                for c in conds:
                    p = cache_path(split, sid, c)
                    if not os.path.exists(p):
                        ok = False; break
                    d = json.load(open(p)); row[c] = d['methods']; vrow[c] = d['visible']
                if ok:
                    data[sid] = row; vis[sid] = vrow
            sids = list(data); n = len(sids)
            out['n_recordings'][f'{grp}|{split}'] = n
            if n == 0:
                continue
            ov = {}
            for a in range(len(conds)):
                for b_ in range(a + 1, len(conds)):
                    ca, cb = conds[a], conds[b_]
                    o = [len(set(vis[s][ca]) & set(vis[s][cb])) for s in sids]
                    ov[f'{ca}_vs_{cb}'] = {'mean': float(np.mean(o)), 'min': int(min(o)), 'max': int(max(o)), 'n_identical': int(sum(x == VC for x in o))}
            out['overlap'][f'{grp}|{split}'] = ov
            arr = {m: {c: np.array([_nan(data[s][c].get(m, {}).get('at')) for s in sids]) for c in conds} for m in METHODS}
            from tgate_rule import ok as _tgate_ok   # RT only where the commercial T-gate holds >= 3 vertices (2026-10-20)
            arr_rt = {m: {c: np.array([_nan(data[s][c].get(m, {}).get('rt')) if _tgate_ok(split, s, 'commercial') else np.nan for s in sids]) for c in conds} for m in METHODS}
            per = {}
            for m in METHODS:
                e = {}
                for met, A in (('at', arr[m]), ('rt', arr_rt[m])):
                    fin = np.all([np.isfinite(A[c]) for c in conds], axis=0)
                    means = {c: float(A[c][fin].mean()) if fin.any() else np.nan for c in conds}
                    v = np.array(list(means.values()))
                    e[met] = {'per_cond': {c: _f(means[c]) for c in conds}, 'n': int(fin.sum()), 'range': _f(v.max() - v.min()),
                              'sd': _f(v.std(ddof=1)), 'max_abs_dev_from_paper': _f(np.max(np.abs(v - v[0])))}
                per[m] = e
            out['per_cond'][f'{grp}|{split}'] = per
            bsp = {}
            for m in MODEL_METHODS:
                vals = {c: [data[s][c].get(m, {}).get('bsp') for s in sids] for c in conds}
                bsp[m] = {c: _f(np.nanmean([_nan(x) for x in vals[c]])) if any(x is not None for x in vals[c]) else None for c in conds}
            out['bsp'][f'{grp}|{split}'] = bsp
            diff = {}
            for m in METHODS:
                if m == 'ours':
                    continue
                e = {}
                for met, ao, ab in (('at', arr['ours'], arr[m]), ('rt', arr_rt['ours'], arr_rt[m])):
                    fin = np.all([np.isfinite(ao[c]) & np.isfinite(ab[c]) for c in conds], axis=0)
                    dm, pv = {}, {}
                    for c in conds:
                        dd = (ao[c] - ab[c])[fin]
                        dm[c] = float(dd.mean()) if fin.any() else np.nan
                        pv[c] = float(wilcoxon(dd).pvalue) if fin.sum() >= 10 and (dd != 0).any() else None
                    v = np.array(list(dm.values())); signs = set(np.sign(v[np.abs(v) > 1e-12]).tolist())
                    e[met] = {'per_cond': {c: _f(dm[c]) for c in conds}, 'p': pv, 'n': int(fin.sum()), 'sd': _f(v.std(ddof=1)),
                              'range': _f(v.max() - v.min()), 'flips': bool(len(signs) > 1)}
                diff[m] = e
            out['diff'][f'{grp}|{split}'] = diff
            lines.append(f'\n=== {grp} / {split}  (n={n} recordings with every condition cached) ===')
            lines.append('visible-set overlap (of 16): ' + '  '.join(f"{k}: {v['mean']:.1f} [{v['min']},{v['max']}] identical {v['n_identical']}" for k, v in ov.items()))
            hdr = f"{'method':<10}" + ''.join(f'  AT {CLAB[c]:<6}' for c in conds) + '   rng    sd  |' + ''.join(f'  RT {CLAB[c]:<6}' for c in conds) + '   rng    sd'
            lines.append(hdr)
            for m in METHODS:
                e = per[m]; row = f'{m:<10}'
                for met in ('at', 'rt'):
                    for c in conds:
                        v = e[met]['per_cond'][c]; row += f'  {v:9.2f}' if v is not None else '         --'
                    row += f"  {e[met]['range']:5.2f} {e[met]['sd']:5.2f}" + (' |' if met == 'at' else '')
                lines.append(row)
            lines.append('--- paired diff (ours - method), cohort mean per condition; FLIP = sign changes with the condition ---')
            for m in METHODS:
                if m == 'ours':
                    continue
                e = diff[m]; row = f'{m:<10}'
                for met in ('at', 'rt'):
                    for c in conds:
                        v = e[met]['per_cond'][c]; row += f'  {v:9.2f}' if v is not None else '         --'
                    row += f"  sd {e[met]['sd']:5.2f} {'FLIP' if e[met]['flips'] else 'no  '}" + (' |' if met == 'at' else '')
                lines.append(row)
    return out, '\n'.join(lines)


def write_tables(out):
    f2 = lambda v: ('%.2f' % v) if isinstance(v, (int, float)) and np.isfinite(v) else '--'
    for grp, name in (('kmeans', 'tab_deploy_kmeans_v2.tex'), ('template', 'tab_deploy_template_v2.tex')):
        conds = GROUPS[grp]; k = len(conds)
        L = [r'\begin{tabular}{l' + 'c' * (2 * (k + 1)) + '}', r'\toprule',
             r' & \multicolumn{%d}{c}{AT MAE (ms)} & \multicolumn{%d}{c}{RT MAE (ms)} \\' % (k + 1, k + 1),
             r'\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}' % (k + 1, k + 2, 2 * k + 2),
             'Method & ' + ' & '.join(CLAB[c] for c in conds) + ' & SD & ' + ' & '.join(CLAB[c] for c in conds) + r' & SD \\']
        for split, lab in (('test_internal', 'internal dataset'), ('test_external', 'external dataset')):
            per = out['per_cond'].get(f'{grp}|{split}')
            if not per:
                continue
            L.append(r'\midrule'); L.append(r'\multicolumn{%d}{l}{\emph{%s}} \\' % (2 * k + 3, lab))
            for m in METHODS:
                e = per[m]
                cells = [f2(e['at']['per_cond'][c]) for c in conds] + [f2(e['at']['sd'])] + [f2(e['rt']['per_cond'][c]) for c in conds] + [f2(e['rt']['sd'])]
                L.append(LABEL[m] + ' & ' + ' & '.join(cells) + r' \\')
        L += [r'\bottomrule', r'\end{tabular}']
        open(assert_v2(os.path.join(V2_GENERATED, name)), 'w').write('\n'.join(L) + '\n'); print('wrote', name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=80)
    ap.add_argument('--conds', nargs='*', default=list(KM_CONDS) + list(TPL_CONDS))
    ap.add_argument('--splits', nargs='*', default=list(SPLITS))
    ap.add_argument('--build_templates', action='store_true')
    ap.add_argument('--agg_only', action='store_true')
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--no_reuse_v1', action='store_true')
    a = ap.parse_args()
    if a.build_templates:
        build_templates(); return
    recs = recordings(a.n)
    if not a.agg_only:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models, keep_idx, args = load_models(device)
        cov = np.load(COVF)['cov']
        lam = {'kmeans': json.load(open(os.path.join(V1_GENERATED, 'lmmse_lam.json')))[str(VC)]['lam_rel'],
               'template': json.load(open(os.path.join(V2_GENERATED, 'classical_template', 'lmmse_lam_template.json')))[str(VC)]['lam_rel']}
        print(f'device {device}; models loaded; lmmse lam_rel {lam}', flush=True)
        for split in a.splits:
            run_shard(split, recs[split][a.shard::a.nshards], a.conds, models, keep_idx, args, cov, lam, device, reuse_v1=not a.no_reuse_v1)
        if a.nshards > 1:
            print(f'shard {a.shard}/{a.nshards} done', flush=True); return
    out, table = aggregate(recs)
    json.dump(out, open(assert_v2(OUT_JSON), 'w'), indent=1)
    print(table); write_tables(out); print('wrote', OUT_JSON)


if __name__ == '__main__':
    main()
