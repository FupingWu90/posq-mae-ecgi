"""Position-aware linear baseline (reviewer point 3, 2026-10-23): Gaussian-process / kriging completion of the body-surface
field with a stationary Matern-5/2 kernel on the normalised electrode coordinates and a common-average mean (ordinary
kriging), i.e. the best linear predictor under a stationary spatial covariance, between TPS (polyharmonic kernel, no
nugget) and LMMSE (empirical, data-driven covariance).  Same protocol as baselines_analyses/classical_baselines_template.py:
fixed template layout, N in 4..64, held-out BSP accuracy, inverse -> AT/RT against the commercial reference and the
full-vest inverse; hyperparameters (length-scale in units of the median electrode spacing, nugget) chosen per N on the
same 20 validation recordings by the held-out BSP MAE, as the LMMSE ridge weight was.  CPU only; aggregates only printed.
  --calibrate                       -> generated/gp_baseline/gp_hyper_template.json
  --split S --shard i --nshards n   -> generated/gp_baseline/<split>_shard<i>.csv
  --summary                         -> generated/gp_baseline_v2.json, tab_gp_baseline_v2.tex
"""
import os, sys, json, csv, glob, argparse
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '3')      # must precede the numpy import (BLAS reads it at load time; 2026-10-23 the shards ran at ~5 cores each)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(V2, 'baselines_analyses'))
import classical_baselines_template as CB                    # CPU-only env + every convention of the classical baselines
from classical_baselines_template import (_setup, _select, _bsp_cc, _g_visible, _hsp, build_context, _to256, CROP, assemble_solver,
                                          solve_from_field, load_commercial, extract_sample, map_mae_cc, cache_lookup, collect_split,
                                          VCS, assert_v2, V2_GENERATED)
from dedup import ok as dedup_ok
OUT = assert_v2(os.path.join(V2_GENERATED, 'gp_baseline')); os.makedirs(OUT, exist_ok=True)
HYPF = os.path.join(OUT, 'gp_hyper_template.json')
SPACING = json.load(open(os.path.join(V2, 'bsp_reconstruction', 'templates', 'templates.json')))['spacing_normalised']
ELL_GRID = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]      # length-scale / median electrode spacing
TAU_GRID = [1e-4, 1e-3, 1e-2, 1e-1]                        # nugget relative to the unit signal variance
OURS_ID = {4: 'P3N4', 8: 'P3N8', 16: 'LOC3', 32: 'P3N32', 64: 'P3N64'}   # as baselines_analyses/results_v2.py
COLS = ['sid', 'vc', 'n_template_sub', 'bsp_cc', 'bsp_cc_n', 'bsp_mae', 'hsp_cc', 'at_com', 'rt_com', 'ari_com', 'at_src2', 'rt_src2']


def matern52(D, ell):
    r = np.sqrt(5.0) * D / ell
    return (1.0 + r + r * r / 3.0) * np.exp(-r)


def predict(coords, visible, coord_ok, prep, ell, tau):
    """Ordinary kriging per time sample: common average of the visible channels + Matern-5/2 GP residual."""
    V = np.where(visible & coord_ok)[0]; Q = np.where(coord_ok)[0]
    Kvv = matern52(np.linalg.norm(coords[V][:, None] - coords[V][None], axis=-1), ell) + tau * np.eye(len(V))
    Kqv = matern52(np.linalg.norm(coords[Q][:, None] - coords[V][None], axis=-1), ell)
    y = prep[V].astype(np.float64); m = y.mean(axis=0, keepdims=True)
    W = Kqv @ np.linalg.solve(Kvv, np.eye(len(V)))
    pred = np.zeros(prep.shape, np.float64); pred[Q] = m + W @ (y - m)
    return pred.astype(np.float32)


def calibrate(n_recs=20):
    args, keep_idx = _setup()
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets['val'] if r[2]]
    recs = [recs[i] for i in np.linspace(0, len(recs) - 1, n_recs).astype(int)]
    err = {}
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16, mode='cluster', cluster_method='kmean', seed=42)
        if ctx is None:
            continue
        act = ctx['active252'].astype(bool); prep = ctx['prep252']; coords = ctx['coords']; coord_ok = np.abs(coords).sum(1) > 1e-6
        sd = prep[act].std(axis=1); g = np.median(sd[sd > 1e-8]) if (sd > 1e-8).any() else 1.0
        for vc in VCS:
            vis, _ = _select(coords, act, vc, 'template')
            held = act & coord_ok & ~vis
            if held.sum() == 0:
                continue
            for ell in ELL_GRID:
                for tau in TAU_GRID:
                    pred = predict(coords, vis, coord_ok, prep, ell * SPACING, tau)
                    err.setdefault((vc, ell, tau), []).append(float(np.abs(pred[held] - prep[held]).mean() / max(g, 1e-8)))
        print(f'[cal] {i + 1}/{len(recs)}', flush=True)
    best = {}
    for vc in VCS:
        scores = {(ell, tau): float(np.mean(err[(vc, ell, tau)])) for ell in ELL_GRID for tau in TAU_GRID if (vc, ell, tau) in err}
        ell, tau = min(scores, key=scores.get)
        best[str(vc)] = dict(ell_spacing=ell, tau=tau, val_bsp_mae=scores[(ell, tau)], spacing_normalised=SPACING,
                             all_scores={f'{e}|{t}': v for (e, t), v in scores.items()})
        print(f'N={vc:2d}: ell {ell:g} spacings, tau {tau:g}, held-out BSP MAE {scores[(ell, tau)]:.4f}')
    json.dump(best, open(HYPF, 'w'), indent=1); print('wrote', HYPF)


def run_shard(split, shard, nshards, n_max):
    args, keep_idx = _setup(); hyp = json.load(open(HYPF))
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets[split] if r[2] and dedup_ok(r[0])][:n_max][shard::nshards]
    out_csv = os.path.join(OUT, f'{split}_shard{shard}.csv')
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as fh:
            done = {row['sid'] for row in csv.DictReader(fh)}
    new = not os.path.exists(out_csv)
    geom_cache = {}; n_fail = 0
    with open(out_csv, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLS)
        for i, (sid, ecgpath, torsopath) in enumerate(recs):
            if sid in done:
                continue
            try:
                gdir = os.path.dirname(torsopath)
                if gdir not in geom_cache:
                    if len(geom_cache) >= 4:
                        geom_cache.pop(next(iter(geom_cache)))
                    geom_cache[gdir] = assemble_solver(gdir)
                solver, torso_ci, hV, hF = geom_cache[gdir]; n_h = hV.shape[0]
                ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16, mode='cluster', cluster_method='kmean', seed=42)
                if ctx is None:
                    raise RuntimeError('no context')
                prep, coords, active = ctx['prep252'], ctx['coords'], ctx['active252']; coord_ok = np.abs(coords).sum(1) > 1e-6
                rp = cache_lookup(split, 'refs', sid.replace('/', '_') + '.npz')
                if rp is None:
                    raise RuntimeError('no refs')
                with np.load(rp, allow_pickle=False) as z:
                    ref = {k: z[k] for k in z.files}
                atc, rtc = ref['ATcom'].astype(float), ref['RTcom'].astype(float); qc, tc_ = ref['qrs_goodcom'].astype(bool), ref['t_goodcom'].astype(bool)
                at2, rt2 = ref['AT2'].astype(float), ref['RT2'].astype(float); q2, t2 = ref['qrs_good2'].astype(bool), ref['t_good2'].astype(bool)
                tw = (ctx['crop_start'], ctx['crop_start'] + CROP)
                phicom = load_commercial(os.path.splitext(ecgpath)[0] + '.potential', n_h, time_window=tw)
                for vc in VCS:
                    visible, n_sub = _select(coords, active, vc, 'template')
                    g_vis = _g_visible(prep, visible)
                    ev = active & coord_ok & ~visible
                    h = hyp[str(vc)]
                    pred = predict(coords, visible, coord_ok, prep, h['ell_spacing'] * SPACING, h['tau'])
                    f252 = prep.copy().astype(np.float32); f252[ev] = pred[ev]
                    cc, n_cc = _bsp_cc((pred[c], prep[c]) for c in np.where(ev)[0])
                    mae = float(np.abs(pred[ev] - prep[ev]).mean() / g_vis) if ev.any() else float('nan')
                    V256, a256 = _to256(f252, active & (visible | coord_ok), keep_idx)
                    phi = solve_from_field(solver, torso_ci, V256, a256)[0]
                    row = dict(sid=sid, vc=vc, n_template_sub=n_sub, bsp_cc=cc, bsp_cc_n=n_cc, bsp_mae=mae)
                    hcc, _ = _hsp(phi, phicom); row['hsp_cc'] = hcc
                    if phi is not None and np.all(np.isfinite(phi)):
                        r = extract_sample(phi)
                        if r is not None:
                            AT, RT = r['AT'].astype(float), r['RT'].astype(float)
                            row.update(at_com=map_mae_cc(AT, atc, qc)[0], rt_com=map_mae_cc(RT, rtc, tc_)[0],
                                       ari_com=map_mae_cc(RT - AT, rtc - atc, tc_)[0], at_src2=map_mae_cc(AT, at2, q2)[0], rt_src2=map_mae_cc(RT, rt2, t2)[0])
                            if int(tc_.sum()) < 3:
                                row.update(rt_com=float('nan'), ari_com=float('nan'))
                            if int(t2.sum()) < 3:
                                row.update(rt_src2=float('nan'))
                    w.writerow([row.get(k, float('nan')) for k in COLS]); fh.flush()
            except Exception as e:
                n_fail += 1; print(f'  [{i + 1}] FAILED: {type(e).__name__}', flush=True)
            if (i + 1) % 10 == 0:
                print(f'[{split} shard {shard}] {i + 1}/{len(recs)} (failed {n_fail})', flush=True)
    print(f'[{split} shard {shard}] done, failed {n_fail}')


def summary():
    import pandas as pd
    from scipy.stats import wilcoxon
    cls = json.load(open(os.path.join(V2_GENERATED, 'classical_template', 'summary_template.json')))
    PR = os.path.join(V2_GENERATED, 'percase_rerun')
    out = {'hyper': json.load(open(HYPF))}
    L = [r'\begin{tabular}{llccccccccc}', r'\toprule',
         r' & & BSP CC & \multicolumn{4}{c}{AT MAE (ms)} & \multicolumn{4}{c}{RT MAE (ms)} \\', r'\cmidrule(lr){4-7}\cmidrule(lr){8-11}',
         r'Cohort & $N$ & GP & TPS & LMMSE & GP & PosQ-MAE & TPS & LMMSE & GP & PosQ-MAE \\']
    for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
        files = sorted(glob.glob(os.path.join(OUT, f'{split}_shard*.csv')))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files]); L.append(r'\midrule'); first = True
        for vc in VCS:
            d = df[df.vc == vc].set_index('sid')
            # PosQ-MAE and full-vest rows: the per-recording files of the paper's final model at every N (results_v2.OURS_ID)
            b = pd.read_csv(os.path.join(V2_GENERATED, 'percase_candidate', f'{OURS_ID[vc]}_vc{vc}_{split}_template.csv'))
            b = b[(b.ref == 'commercial') & (b.success == 1)]
            s4 = b[b.source == 'src4_recon_valid'].set_index('sid')
            cm = cls['splits'][split]['methods']
            rec = dict(n=int(len(d)), bsp_cc=float(np.nanmean(d.bsp_cc)), at=float(np.nanmean(d.at_com)), rt=float(np.nanmean(d.rt_com)),
                       at_src2=float(np.nanmean(d.at_src2)), rt_src2=float(np.nanmean(d.rt_src2)), hsp_cc=float(np.nanmean(d.hsp_cc)))
            cells = [f"{rec['bsp_cc']:.3f}"]
            for met, col4 in (('at', 'AT_mae'), ('rt', 'RT_mae')):
                for m in ('tps', 'lmmse'):
                    v = cm[m][str(vc)][f'{met}_com'] if str(vc) in cm[m] and f'{met}_com' in cm[m][str(vc)] else float('nan')
                    cells.append(f'{v:.1f}' if np.isfinite(v) else '--'); rec[f'{m}_{met}'] = float(v) if np.isfinite(v) else None
                j = d[[f'{met}_com']].join(s4[[col4]], how='inner').dropna(); dd = j[f'{met}_com'] - j[col4]
                p = wilcoxon(dd).pvalue if len(dd) > 5 and np.any(dd != 0) else float('nan')
                st = '' if not np.isfinite(p) else '^{***}' if p < 1e-3 else '^{**}' if p < 1e-2 else '^{*}' if p < 0.05 else '^{\\mathrm{ns}}'
                cells += [f"{rec[met]:.1f} (${dd.mean():+.2f}{st}$)", f"{float(np.nanmean(s4[col4])):.1f}"]
                rec[f'd_{met}_vs_posq'] = float(dd.mean()); rec[f'p_{met}_vs_posq'] = float(p) if np.isfinite(p) else None; rec[f'posq_{met}'] = float(np.nanmean(s4[col4]))
            if vc == 16:   # the full-vest inverse does not depend on N: stored once per cohort from the N=16 rerun file
                bb = pd.read_csv(os.path.join(PR, f'base_vc16_{split}_template.csv')); bb = bb[(bb.ref == 'commercial') & (bb.success == 1) & (bb.source == 'src2_allvalid')]
                out[f'{lab}|vest'] = dict(at=float(np.nanmean(bb.AT_mae)), rt=float(np.nanmean(bb.RT_mae)), n=int(len(bb)))
            out[f'{lab}|N{vc}'] = rec
            L.append(f"{lab if first else ''} & {vc} & " + ' & '.join(cells) + r' \\'); first = False
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_gp_baseline_v2.tex')), 'w').write('\n'.join(L) + '\n')
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'gp_baseline_v2.json')), 'w'), indent=1)
    for k, v in out.items():
        if k != 'hyper':
            print(f'{k:16s} ' + ' '.join(f'{a}={b:.3f}' if isinstance(b, float) else f'{a}={b}' for a, b in v.items()))
    print('wrote tab_gp_baseline_v2.tex, gp_baseline_v2.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--calibrate', action='store_true'); ap.add_argument('--split'); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1); ap.add_argument('--n', type=int, default=10 ** 9); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.split:
        run_shard(a.split, a.shard, a.nshards, a.n)
    if a.summary:
        summary()
