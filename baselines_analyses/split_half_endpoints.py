"""Split-half (odd / even beats) repeatability of the clinical endpoints (collaborators' item A3, noise floor; 2026-10-21).
For every test recording the completed field of the final model is rebuilt (as inverse_evaluation/at_maps.py), the MFS inverse
solved, and the activation / repolarisation times extracted three times: from all beats, from the odd beats and from the
even beats of the same recording (beats detected once on the full-vest potentials and shared by every source); the
full-vest inverse (phi2 of the reference cache) and the raw 16-electrode inverse (phi1 of the vc cache) are treated
identically.  The endpoints of baselines_analyses/clinical_endpoints.py are then computed per half on the commercial-gated vertices.
Summary: SD of the odd-even difference per source and endpoint; divided by sqrt(2) it is the disagreement two independent
measurements of the same recording by the same system would show (each half holds half of the beats, so this bounds the
noise of the full-recording endpoint from above), next to the SD of the Ours-minus-full-vest difference of the
full-recording endpoints.  Resumable (rows are appended to the CSV).  Aggregates only.
  python baselines_analyses/split_half_endpoints.py --split test_internal [--layout template] [--id LOC3] [--n N]
  -> generated/split_half/<split>_<layout>.csv, split_half_v2.json (summary over the available splits/layouts)"""
import os, sys, csv, json, glob, argparse, warnings
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('bsp_reconstruction', 'baselines_analyses', 'inverse_evaluation'):
    sys.path.insert(0, os.path.join(V2, d))
import torch
from paths import V2_GENERATED, V2_RUNS, assert_v2
from build_sources import load_model, build_context, reconstruct_sources
from inverse import assemble_solver, solve_from_field
from at_rt import extract_sample, detect_beats
from cache_references import cache_lookup, vc_tag
from dataset import collect_split
from dedup import ok as dedup_ok
from clinical_endpoints import geom_coords, sectors, endpoints, ENDPOINTS, LABEL, icc21, recordings as ce_recordings
LAYOUTS = {'template': ('template', 'greedy'), 'kmeans': ('cluster', 'kmean')}
OUT = os.path.join(V2_GENERATED, 'split_half')
SRC = ['ours', 'allvalid', 'vconly']


def ep_from(phi, rp, qg, tg, lv, rv):
    try:
        r = extract_sample(phi, r_peaks=np.asarray(rp))
    except Exception:
        r = None
    if r is None:
        return {k: np.nan for k in ENDPOINTS}
    return endpoints(np.asarray(r['AT'], float), np.asarray(r['RT'], float), qg, tg, lv, rv)


def run(a):
    warnings.filterwarnings('ignore'); os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, f'{a.split}_{a.layout}.csv')
    done = set()
    if os.path.exists(out_csv):
        done = {r['sid'] for r in csv.DictReader(open(out_csv))}
    rid = 'CARTOHO' if a.split == 'holdout' else a.id
    run_dir = sorted(glob.glob(os.path.join(V2_RUNS, '*', f'{rid}__*')))[0]
    dev = torch.device(a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu'); torch.set_num_threads(a.threads)
    model, keep_idx, args = load_model(run_dir, dev)
    vc = int(args['vc']); mode, cm = LAYOUTS[a.layout]
    hold = set(l.strip() for l in open(args['holdout_file']) if l.strip()) if args.get('holdout_file') else None
    b, _ = collect_split(seed=args.get('seed', 42), holdout=hold)
    _recs, pm = ce_recordings(a.split, rid)
    keep = {s for s, _g in _recs}
    recs = [r for r in b[a.split] if r[2] and r[0] in keep][:a.n]
    vtag = 'vc16_template_s42' if a.layout == 'template' else vc_tag(16, 'cluster', 'kmean', 42)
    cols = ['sid', 'patient', 'n_beats'] + [f'{s}_{e}_{h}' for s in SRC for e in ENDPOINTS for h in ('full', 'odd', 'even')]
    new_file = not os.path.exists(out_csv)
    f = open(out_csv, 'a', newline=''); w = csv.DictWriter(f, fieldnames=cols)
    if new_file:
        w.writeheader()
    solver_cache = {}; n_new = n_skip = n_fail = 0
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        if sid in done:
            n_skip += 1; continue
        try:
            gdir = os.path.dirname(torsopath); sid_safe = sid.replace('/', '_')
            fr = cache_lookup(a.split, 'refs', sid_safe + '.npz'); fv = cache_lookup(a.split, vtag, sid_safe + '.npz')
            if fr is None:
                n_fail += 1; continue
            if gdir not in solver_cache:
                if len(solver_cache) > 3:
                    solver_cache.clear()
                solver_cache[gdir] = assemble_solver(gdir)
            solver, tci, hV, hF = solver_cache[gdir]
            ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=vc, mode=mode, cluster_method=cm, seed=args.get('seed', 42))
            if ctx is None:
                n_fail += 1; continue
            with torch.no_grad():
                recon = reconstruct_sources(model, ctx, dev)
            phi4 = solve_from_field(solver, tci, *recon['src4_recon_valid'])[0]
            with np.load(fr, allow_pickle=False) as zr:
                if 'ATcom' not in zr.files or 'phi2' not in zr.files:
                    n_fail += 1; continue
                phi2 = zr['phi2'].astype(np.float32); qg = zr['qrs_goodcom'].astype(bool); tg = zr['t_goodcom'].astype(bool)
            phi1 = None
            if fv is not None:
                with np.load(fv, allow_pickle=False) as zv:
                    phi1 = zv['phi1'].astype(np.float32) if 'phi1' in zv.files else None
            if phi4 is None or not np.all(np.isfinite(phi4)) or phi4.shape != phi2.shape:
                n_fail += 1; continue
            th, _z = geom_coords(gdir)
            if th.shape[0] != qg.shape[0]:
                n_fail += 1; continue
            lv, rv = sectors(th)
            rp = detect_beats(phi2)
            if rp is None or len(rp) < 4:
                n_fail += 1; continue
            rp = np.asarray(rp); halves = {'full': rp, 'odd': rp[0::2], 'even': rp[1::2]}
            row = {'sid': sid, 'patient': pm.get(sid, ''), 'n_beats': int(len(rp))}
            for s, phi in (('ours', phi4), ('allvalid', phi2), ('vconly', phi1)):
                for h, rph in halves.items():
                    e = ep_from(phi, rph, qg, tg, lv, rv) if phi is not None else {k: np.nan for k in ENDPOINTS}
                    row.update({f'{s}_{k}_{h}': v for k, v in e.items()})
            w.writerow(row); f.flush(); n_new += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f'  fail: {type(e).__name__}: {str(e)[:80]}')
        if (i + 1) % 25 == 0:
            print(f'[{a.split} {a.layout}] {i + 1}/{len(recs)} new {n_new} skip {n_skip} fail {n_fail}', flush=True)
    f.close(); print(f'[{a.split} {a.layout}] done: new {n_new} skip {n_skip} fail {n_fail}')


def summary():
    res = {}
    for p in sorted(glob.glob(os.path.join(OUT, '*.csv'))):
        key = os.path.basename(p)[:-4]; rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        g = lambda c: np.array([float(r[c]) if r.get(c) not in ('', None, 'nan') else np.nan for r in rows])
        out = {'n': len(rows), 'median_beats': float(np.median(g('n_beats'))), 'min_beats': float(np.min(g('n_beats')))}; res.setdefault(key, {})['meta'] = out
        for e in ENDPOINTS:
            d_ov = g(f'ours_{e}_full') - g(f'allvalid_{e}_full'); m = np.isfinite(d_ov)
            ent = {'sd_ours_minus_fullvest': float(d_ov[m].std(ddof=1)) if m.sum() > 2 else np.nan, 'n_pairs': int(m.sum())}
            for s in SRC:
                o, ev = g(f'{s}_{e}_odd'), g(f'{s}_{e}_even'); k = np.isfinite(o) & np.isfinite(ev)
                if k.sum() > 2:
                    d = o[k] - ev[k]
                    ent[s] = {'n': int(k.sum()), 'sd_odd_even': float(d.std(ddof=1)), 'sd_independent_repeats': float(d.std(ddof=1) / np.sqrt(2)),
                              'bias_odd_even': float(d.mean()), 'icc_odd_even': icc21(o[k], ev[k])}
            res.setdefault(key, {})[e] = ent
    json.dump(res, open(assert_v2(os.path.join(V2_GENERATED, 'split_half_v2.json')), 'w'), indent=1)
    L = [r'\begin{tabular}{llrrrrr}', r'\toprule', r'Cohort & Endpoint & \multicolumn{3}{c}{odd $-$ even beats, SD$/\sqrt{2}$ (ms)} & \multicolumn{2}{c}{full recordings, SD (ms)} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}', r' & & completion & full vest & raw 16 & completion $-$ full vest & $n$ \\']
    for key in ('test_internal_template', 'test_external_template'):   # the table reports the fixed template layout (k-means stays in the JSON)
        if key not in res or res[key].get('lvat95', {}).get('n_pairs', 0) < 200:   # skip splits whose run is incomplete
            continue
        L.append(r'\midrule'); first = True; lab = key.replace('test_', '').replace('_template', '').replace('_kmeans', ' ($k$-means)')
        for e in ('tat', 'lvat95', 'lvdi', 'veu', 'at_sd', 'ari_mean'):
            ent = res[key].get(e)
            if not ent:
                continue
            cell = lambda s: f"{ent[s]['sd_independent_repeats']:.1f}" if s in ent else '--'
            L.append(f"{lab if first else ''} & {LABEL[e]} & {cell('ours')} & {cell('allvalid')} & {cell('vconly')} & {ent['sd_ours_minus_fullvest']:.1f} & {ent['n_pairs']} \\\\"); first = False
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_split_half_v2.tex')), 'w').write('\n'.join(L) + '\n')
    for key, d in res.items():
        print(f'### {key}: n={d["lvat95"]["n_pairs"] if "lvat95" in d else "?"}')
        for e in ('lvat95', 'tat', 'lvdi', 'veu', 'ari_mean'):
            if e in d:
                ent = d[e]; line = f'  {e:9s} SD(Ours-full vest) {ent["sd_ours_minus_fullvest"]:5.1f}'
                for s in SRC:
                    if s in ent:
                        line += f' | {s:8s} odd-even SD {ent[s]["sd_odd_even"]:5.1f} (/sqrt2 {ent[s]["sd_independent_repeats"]:4.1f}) ICC {ent[s]["icc_odd_even"]:.2f}'
                print(line)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test_internal'); ap.add_argument('--layout', default='template', choices=sorted(LAYOUTS))
    ap.add_argument('--id', default='LOC3'); ap.add_argument('--n', type=int, default=10**9); ap.add_argument('--device', default='cuda'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if not a.summary:
        run(a)
    summary()
