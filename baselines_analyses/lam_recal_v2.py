"""Lambda-fairness experiment for the VC-only baseline under a deployment layout of choice (supplement S6,
tab:lambda; v2 port of the frozen v1 lam_recal.py + the lambda part of vc_stats.py).  Model free: the pipeline
calibrates the L-curve floor once on the all-valid inverse; here the floor is re-calibrated PER electrode count
for the VC-only inverse itself (grid search on 20 validation recordings, minimising AT + RT MAE vs the commercial
reference; alpha_in / alpha_out unchanged) and the baseline is re-evaluated on both test cohorts.
  --calibrate [--layout template]      -> generated/lam_recal_<layout>/lam_per_n.json
  --eval --split S [--shard k --nshards m]  -> generated/lam_recal_<layout>/<split>_shard<k>.csv (idempotent)
  --table                              -> generated/tab_lambda{,_km}_v2.tex + lam_recal_<layout>_v2.json
     shared-floor rows = the standard VC-only rows (src1_vc, commercial) of the per-recording CSVs of the N-grid
     runs under the same layout (generated/percase_candidate/<P3N*|LOC3>_vc<N>_<split>_<layout>.csv).
Prints aggregates only."""
import argparse, csv, glob, json, os, sys, warnings
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(V2, 'inverse_evaluation')); sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from paths import V2_GENERATED, V2_RUNS, assert_v2
from build_sources import build_context, _to256
from channel_select import select_visible_channels
from inverse import assemble_solver, solve_from_field
from at_rt import extract_sample
from metrics import map_mae_cc
from cache_references import cache_lookup
from dataset import collect_split
warnings.filterwarnings('ignore')

VCS = [4, 8, 16, 32, 64]
FLOORS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SPLITS = ['test_internal', 'test_external']
OURS_ID = {4: 'P3N4', 8: 'P3N8', 16: 'LOC3', 32: 'P3N32', 64: 'P3N64'}
LAYOUTS = {'template': ('template', 'greedy'), 'kmeans': ('cluster', 'kmean')}
LAYOUT = 'template'


def outdir():
    d = assert_v2(os.path.join(V2_GENERATED, f'lam_recal_{LAYOUT}')); os.makedirs(d, exist_ok=True); return d


def _setup():
    rd = sorted(glob.glob(os.path.join(V2_RUNS, 'data-cohortA+cohortB_vs_external', 'LOC3__*')))[0]
    args = json.load(open(os.path.join(rd, 'args.json')))
    keep_idx = np.load(os.path.join(rd, 'keep_idx.npy')).astype(np.int64)
    return args, keep_idx


def _ref(split, sid):
    p = cache_lookup(split, 'refs', sid.replace('/', '_') + '.npz')
    return np.load(p, allow_pickle=False) if p else None


def _src1_field(ctx, keep_idx, vc):
    mode, cm = LAYOUTS[LAYOUT]
    vis = select_visible_channels(ctx['coords'], ctx['active252'], vc, mode, cm, np.random.default_rng(42))
    return _to256(ctx['prep252'], vis, keep_idx)


def _ctx(rec, keep_idx, args):
    mode, cm = LAYOUTS[LAYOUT]
    return build_context(rec[1], rec[2], keep_idx, args, vc=16, mode=mode, cluster_method=cm, seed=42)


def calibrate(n_recs=20):
    args, keep_idx = _setup()
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets['val'] if r[2]]
    recs = [recs[i] for i in np.linspace(0, len(recs) - 1, n_recs).astype(int)]
    acc = {(vc, lf): [] for vc in VCS for lf in FLOORS}
    solver_cache = {}
    for i, rec in enumerate(recs):
        ref = _ref('val', rec[0])
        if ref is None or 'ATcom' not in ref.files:
            continue
        atc, rtc = ref['ATcom'].astype(float), ref['RTcom'].astype(float)
        qc, tc = ref['qrs_goodcom'].astype(bool), ref['t_goodcom'].astype(bool)
        gdir = os.path.dirname(rec[2])
        if gdir not in solver_cache:
            solver_cache[gdir] = assemble_solver(gdir)
        solver, tci, hV, hF = solver_cache[gdir]
        ctx = _ctx(rec, keep_idx, args)
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
                at = map_mae_cc(r['AT'].astype(float), atc, qc)[0]; rt = map_mae_cc(r['RT'].astype(float), rtc, tc)[0]
                if np.isfinite(at) and np.isfinite(rt):
                    acc[(vc, lf)].append(at + rt)
        print(f'[cal] {i+1}/{len(recs)}', flush=True)
    best = {}
    for vc in VCS:
        scores = {lf: float(np.mean(acc[(vc, lf)])) for lf in FLOORS if acc[(vc, lf)]}
        lf = min(scores, key=scores.get)
        best[str(vc)] = dict(lam_floor=lf, val_at_plus_rt=scores[lf], all_scores={str(k): v for k, v in scores.items()}, n=len(acc[(vc, lf)]))
        print(f'N={vc:2d}: best floor {lf:g} ' + ' '.join(f'{k:g}:{v:.1f}' for k, v in sorted(scores.items())))
    json.dump(best, open(os.path.join(outdir(), 'lam_per_n.json'), 'w'), indent=2)


def evaluate(split, shard, nshards):
    out = os.path.join(outdir(), f'{split}_shard{shard}.csv')
    if os.path.exists(out):
        print('[skip]', out); return
    args, keep_idx = _setup()
    best = json.load(open(os.path.join(outdir(), 'lam_per_n.json')))
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets[split] if r[2]][shard::nshards]
    rows, solver_cache = [], {}
    for i, rec in enumerate(recs):
        try:
            ref = _ref(split, rec[0])
            if ref is None:
                continue
            atc, rtc = ref['ATcom'].astype(float), ref['RTcom'].astype(float)
            at2, rt2 = ref['AT2'].astype(float), ref['RT2'].astype(float)
            qc, tc = ref['qrs_goodcom'].astype(bool), ref['t_goodcom'].astype(bool)
            q2, t2 = ref['qrs_good2'].astype(bool), ref['t_good2'].astype(bool)
            gdir = os.path.dirname(rec[2])
            if gdir not in solver_cache:
                while len(solver_cache) >= 3:
                    solver_cache.pop(next(iter(solver_cache)))
                solver_cache[gdir] = assemble_solver(gdir)
            solver, tci, hV, hF = solver_cache[gdir]
            ctx = _ctx(rec, keep_idx, args)
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
                rows.append(dict(sid=rec[0], vc=vc, lam_floor=lf, at_com=map_mae_cc(AT, atc, qc)[0], rt_com=map_mae_cc(RT, rtc, tc)[0],
                                 at_src2=map_mae_cc(AT, at2, q2)[0], rt_src2=map_mae_cc(RT, rt2, t2)[0],
                                 ari_com=map_mae_cc(RT - AT, rtc - atc, tc)[0], ari_src2=map_mae_cc(RT - AT, rt2 - at2, t2)[0]))
            if (i + 1) % 20 == 0:
                print(f'[{split} s{shard}] {i+1}/{len(recs)}', flush=True)
        except Exception as e:
            print(f'[{split} s{shard}] recording {i} FAILED: {type(e).__name__}: {e}', flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sid', 'vc', 'lam_floor', 'at_com', 'rt_com', 'at_src2', 'rt_src2', 'ari_com', 'ari_src2'])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f'wrote {out} ({len(rows)} rows)', flush=True)


def table():
    from scipy.stats import wilcoxon
    stars = lambda p: '' if not np.isfinite(p) else ('$^{***}$' if p < 1e-3 else '$^{**}$' if p < 1e-2 else '$^{*}$' if p < 5e-2 else '$^{\\mathrm{ns}}$')
    best = json.load(open(os.path.join(outdir(), 'lam_per_n.json')))
    res = {'lam_per_n': best, 'layout': LAYOUT}
    L = [r'\begin{tabular}{ccccccc}', r'\toprule', r' & \multicolumn{3}{c}{internal (ms)} & \multicolumn{3}{c}{external (ms)} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
         r'$N$ & shared $\lambda_f$ & per-$N$ $\lambda_f$ & $\Delta$ (p) & shared $\lambda_f$ & per-$N$ $\lambda_f$ & $\Delta$ (p) \\']
    recal = {}
    for split in SPLITS:
        rows = [r for f in sorted(glob.glob(os.path.join(outdir(), f'{split}_shard*.csv'))) for r in csv.DictReader(open(f))]
        recal[split] = {vc: {r['sid']: r for r in rows if int(r['vc']) == vc} for vc in VCS}
    for met, col, key in (('AT', 'AT_mae', 'at_com'), ('RT', 'RT_mae', 'rt_com'), ('ARI', 'ARI_mae', 'ari_com')):
        L.append(r'\midrule'); L.append(r'\multicolumn{7}{l}{\emph{%s}} \\' % met)
        for vc in VCS:
            cells = []
            for split in SPLITS:
                pc = os.path.join(V2_GENERATED, 'percase_candidate', f'{OURS_ID[vc]}_vc{vc}_{split}_{LAYOUT}.csv')
                shared = {}
                for r in csv.DictReader(open(pc)):
                    if r['source'] == 'src1_vc' and r['ref'] == 'commercial':
                        try:
                            shared[r['sid']] = float(r[col])
                        except ValueError:
                            pass
                per = {s: float(r[key]) for s, r in recal[split][vc].items() if r[key] not in ('', 'nan')}
                common = [s for s in shared if s in per and np.isfinite(shared[s]) and np.isfinite(per[s])]
                a = np.array([shared[s] for s in common]); b = np.array([per[s] for s in common])
                p = float(wilcoxon(b - a).pvalue) if len(common) >= 10 and np.any(b != a) else np.nan
                res[f'{split}|{met}|{vc}'] = dict(shared=float(a.mean()), per_n=float(b.mean()), med_diff=float(np.median(b - a)), p=p, n=len(common),
                                                   lam_floor=best[str(vc)]['lam_floor'])
                cells += [f'{a.mean():.2f}', f'{b.mean():.2f}', f'${np.median(b - a):+.2f}$' + stars(p)]
            L.append(f'{vc} & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    name = 'tab_lambda_v2.tex' if LAYOUT == 'template' else 'tab_lambda_km_v2.tex'
    open(assert_v2(os.path.join(V2_GENERATED, name)), 'w').write('\n'.join(L) + '\n')
    json.dump(res, open(assert_v2(os.path.join(V2_GENERATED, f'lam_recal_{LAYOUT}_v2.json')), 'w'), indent=1)
    for k, v in res.items():
        if '|' in k:
            print(f"{k}: shared {v['shared']:.2f} per-N {v['per_n']:.2f} (floor {v['lam_floor']:g}) med diff {v['med_diff']:+.2f} p={v['p']:.2g} n={v['n']}")
    print('wrote', name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--layout', default='template', choices=list(LAYOUTS))
    p.add_argument('--calibrate', action='store_true'); p.add_argument('--eval', action='store_true'); p.add_argument('--table', action='store_true')
    p.add_argument('--split'); p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--n_recs', type=int, default=20)
    a = p.parse_args()
    LAYOUT = a.layout
    if a.calibrate:
        calibrate(a.n_recs)
    if a.eval:
        evaluate(a.split, a.shard, a.nshards)
    if a.table:
        table()
