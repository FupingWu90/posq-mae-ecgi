"""Electrode-position jitter (collaborators' item A4a, 2026-10-21): robustness of the physical-position completion to errors
in the recorded positions of the 16 measured electrodes.  The coordinate of every visible electrode fed to the model is
displaced by a random 3-D vector of fixed length d (5, 10, 20, 40 mm; uniformly random direction; one draw per recording
and level, seeded by the recording); the query positions of the other vest electrodes, the inverse and the vertex gating
are unchanged, so only the position embedding of the measured electrodes is perturbed (an index-embedding model is
unaffected by construction).  Scored on 80 evenly spaced test recordings per cohort (after the de-duplication rule):
body-surface completion CC / MAE on the held-out active electrodes and the downstream AT / RT error against the commercial
reference on the commercial-gated vertices, paired against d = 0 (median difference, Wilcoxon).  Fixed template layout.
Resumable (rows appended to the CSV).  Aggregates only.
  python baselines_analyses/position_jitter_v2.py [--id LOC3] [--n 80] [--levels 0 5 10 20 40] [--splits test_internal test_external]
  -> generated/position_jitter/<split>.csv, position_jitter_v2.json, tab_position_jitter_v2.tex"""
import os, sys, csv, json, glob, argparse, warnings, zlib
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('bsp_reconstruction', 'baselines_analyses', 'inverse_evaluation'):
    sys.path.insert(0, os.path.join(V2, d))
import torch
from scipy.stats import wilcoxon
from scipy.spatial import cKDTree
from paths import V2_GENERATED, V2_RUNS, assert_v2
from build_sources import load_model, build_context, reconstruct_sources
from inverse import assemble_solver, solve_from_field
from at_rt import extract_sample
from cache_references import cache_lookup
from torso_loader import load_torso_coords
from dataset import collect_split
from dedup import ok as dedup_ok
from tgate_rule import T_MIN
OUT = os.path.join(V2_GENERATED, 'position_jitter')
COLS = ['sid', 'level_mm', 'disp_spacings', 'bsp_cc', 'bsp_mae', 'AT_mae', 'RT_mae', 'AT_mae_ref2', 'n_vis']


def recordings(split, n, seed=42):
    b, _ = collect_split(seed=seed)
    recs = [r for r in b[split] if r[2] and dedup_ok(r[0])]
    return [recs[i] for i in np.linspace(0, len(recs) - 1, min(n, len(recs))).astype(int)]


def run(a):
    warnings.filterwarnings('ignore'); os.makedirs(OUT, exist_ok=True)
    run_dir = sorted(glob.glob(os.path.join(V2_RUNS, '*', f'{a.id}__*')))[0]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(a.threads)
    model, keep_idx, args = load_model(run_dir, dev); vc = int(args['vc'])
    for split in a.splits:
        out_csv = os.path.join(OUT, f'{split}.csv'); done = set()
        if os.path.exists(out_csv):
            done = {(r['sid'], r['level_mm']) for r in csv.DictReader(open(out_csv))}
        new_file = not os.path.exists(out_csv); f = open(out_csv, 'a', newline=''); w = csv.DictWriter(f, fieldnames=COLS)
        if new_file:
            w.writeheader()
        recs = recordings(split, a.n, args.get('seed', 42)); solver_cache = {}; n_new = n_fail = 0
        for i, (sid, ecgpath, torsopath) in enumerate(recs):
            if all((sid, str(l)) in done for l in a.levels):
                continue
            try:
                gdir = os.path.dirname(torsopath); sid_safe = sid.replace('/', '_')
                fr = cache_lookup(split, 'refs', sid_safe + '.npz')
                if fr is None:
                    n_fail += 1; continue
                with np.load(fr, allow_pickle=False) as zr:
                    if 'ATcom' not in zr.files:
                        n_fail += 1; continue
                    atc, rtc, at2 = zr['ATcom'].astype(float), zr['RTcom'].astype(float), zr['AT2'].astype(float)
                    qg, tg, qg2 = zr['qrs_goodcom'].astype(bool), zr['t_goodcom'].astype(bool), zr['qrs_good2'].astype(bool)
                if gdir not in solver_cache:
                    if len(solver_cache) > 3:
                        solver_cache.clear()
                    solver_cache[gdir] = assemble_solver(gdir)
                solver, tci, hV, hF = solver_cache[gdir]
                ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=vc, mode='template', cluster_method='greedy', seed=args.get('seed', 42))
                if ctx is None:
                    n_fail += 1; continue
                raw = load_torso_coords(torsopath).astype(np.float64); valid = ~np.isnan(raw[:, 0])
                scale_cm = float(np.linalg.norm(raw[valid] - raw[valid].mean(0), axis=1).mean())        # normalisation of torso_loader.normalize_coords
                spacing_cm = float(np.median(cKDTree(raw[valid]).query(raw[valid], k=2)[0][:, 1]))
                vis = ctx['visible252'].astype(bool); act = ctx['active252'].astype(bool); held = act & ~vis
                prep_n = ctx['prep252'] / ctx['g']
                for lev in a.levels:
                    if (sid, str(lev)) in done:
                        continue
                    rng = np.random.default_rng(zlib.crc32(sid.encode()) * 7 + int(lev))
                    coords_j = ctx['coords'].copy()
                    if lev > 0:
                        dirs = rng.normal(size=(int(vis.sum()), 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                        coords_j[vis] += (dirs * (lev / 10.0 / scale_cm)).astype(np.float32)
                    ctx2 = dict(ctx); ctx2['coords'] = coords_j
                    with torch.no_grad():
                        recon = reconstruct_sources(model, ctx2, dev)
                    pred = recon['recon252'] / ctx['g']
                    ccs = [float(np.corrcoef(pred[c], prep_n[c])[0, 1]) for c in np.where(held)[0] if pred[c].std() > 1e-8 and prep_n[c].std() > 1e-8]
                    row = {'sid': sid, 'level_mm': lev, 'disp_spacings': round(lev / 10.0 / spacing_cm, 4), 'bsp_cc': float(np.mean(ccs)) if ccs else np.nan,
                           'bsp_mae': float(np.abs(pred[held] - prep_n[held]).mean()) if held.any() else np.nan, 'n_vis': int(vis.sum())}
                    phi = solve_from_field(solver, tci, *recon['src4_recon_valid'])[0]
                    r = extract_sample(phi) if (phi is not None and np.all(np.isfinite(phi))) else None
                    if r is None:
                        row.update(AT_mae=np.nan, RT_mae=np.nan, AT_mae_ref2=np.nan)
                    else:
                        at4, rt4 = np.asarray(r['AT'], float), np.asarray(r['RT'], float)
                        row['AT_mae'] = float(np.abs(at4 - atc)[qg].mean()) if qg.sum() else np.nan
                        row['RT_mae'] = float(np.abs(rt4 - rtc)[tg].mean()) if tg.sum() >= T_MIN else np.nan
                        row['AT_mae_ref2'] = float(np.abs(at4 - at2)[qg2].mean()) if qg2.sum() else np.nan
                    w.writerow(row); f.flush(); n_new += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f'  fail: {type(e).__name__}: {str(e)[:80]}')
            if (i + 1) % 10 == 0:
                print(f'[{split}] {i + 1}/{len(recs)} rows {n_new} fail {n_fail}', flush=True)
        f.close(); print(f'[{split}] done: rows {n_new} fail {n_fail}')


def summary():
    res = {}
    for p in sorted(glob.glob(os.path.join(OUT, '*.csv'))):
        split = os.path.basename(p)[:-4]; rows = list(csv.DictReader(open(p)))
        by = {}
        for r in rows:
            by.setdefault(r['sid'], {})[float(r['level_mm'])] = r
        levels = sorted({float(r['level_mm']) for r in rows}); S = {}
        for lev in levels:
            vals = {k: np.array([float(by[s][lev][k]) for s in by if lev in by[s] and by[s][lev][k] not in ('', 'nan')]) for k in ('bsp_cc', 'bsp_mae', 'AT_mae', 'RT_mae', 'AT_mae_ref2', 'disp_spacings')}
            ent = {k: float(np.nanmean(v)) for k, v in vals.items() if v.size}; ent['n'] = int(vals['AT_mae'].size)
            if lev > 0 and 0.0 in levels:
                for k in ('bsp_cc', 'AT_mae', 'RT_mae'):
                    pairs = [(float(by[s][lev][k]), float(by[s][0.0][k])) for s in by if lev in by[s] and 0.0 in by[s] and by[s][lev][k] not in ('', 'nan') and by[s][0.0][k] not in ('', 'nan')]
                    d = np.array([x - y for x, y in pairs])
                    if d.size > 2:
                        try:
                            pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                        except ValueError:
                            pv = np.nan
                        ent[f'd_{k}'] = {'median': float(np.median(d)), 'mean': float(d.mean()), 'p': pv, 'n': int(d.size)}
            S[str(lev)] = ent
        res[split] = S
    json.dump(res, open(assert_v2(os.path.join(V2_GENERATED, 'position_jitter_v2.json')), 'w'), indent=1)
    st = lambda p: '' if not np.isfinite(p) else '$^{***}$' if p < 1e-3 else '$^{**}$' if p < 1e-2 else '$^{*}$' if p < 5e-2 else '$^{\\mathrm{ns}}$'
    L = [r'\begin{tabular}{lrrrrrrr}', r'\toprule', r'Cohort & jitter (mm) & spacings & BSP CC & AT MAE (ms) & $\Delta$AT & RT MAE (ms) & $\Delta$RT \\']
    for split in ('test_internal', 'test_external'):
        if split not in res:
            continue
        L.append(r'\midrule'); first = True
        for lev, ent in res[split].items():
            dA, dR = ent.get('d_AT_mae'), ent.get('d_RT_mae')
            L.append(f"{('internal' if 'internal' in split else 'external') if first else ''} & {float(lev):.0f} & {ent.get('disp_spacings', 0):.2f} & {ent['bsp_cc']:.3f} & {ent['AT_mae']:.2f} & "
                     + (f"${dA['median']:+.2f}${st(dA['p'])}" if dA else '--') + f" & {ent['RT_mae']:.2f} & " + (f"${dR['median']:+.2f}${st(dR['p'])}" if dR else '--') + r' \\'); first = False
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_position_jitter_v2.tex')), 'w').write('\n'.join(L) + '\n')
    for split, S in res.items():
        print(f'### {split}: ' + ' | '.join(f"{lev} mm: CC {e['bsp_cc']:.3f} AT {e['AT_mae']:.2f} RT {e['RT_mae']:.2f} (n {e['n']})" for lev, e in S.items()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', default='LOC3'); ap.add_argument('--n', type=int, default=80); ap.add_argument('--levels', type=float, nargs='+', default=[0, 5, 10, 20, 40])
    ap.add_argument('--splits', nargs='+', default=['test_internal', 'test_external']); ap.add_argument('--threads', type=int, default=6); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if not a.summary:
        run(a)
    summary()
