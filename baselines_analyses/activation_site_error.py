"""Patient-level localisation endpoints of the epicardial activation maps, scored against the commercial map on its
QRS-gated vertices: (i) earliest-activation site = centroid of the earliest 10 % of the gated vertices (breakthrough
or pacing site), (ii) latest-activation site = centroid of the latest 10 % (the region targeted by resynchronisation),
both as the distance in mm between the centroid of a source and that of the commercial map; (iii) total activation
time = 95th - 5th percentile of AT over the gated vertices, as the absolute difference to the commercial value (ms).
Sources: Ours (PosQ-MAE, N=16; per-vertex maps from inverse_evaluation/at_maps.py), the all-valid inverse (AT2) and the
VC-only inverse (AT1) from the reference caches.  Heart meshes are stored in cm and converted to mm.  Per-recording
CSV (sid column, never printed), cohort medians [IQR], per-patient means and paired Wilcoxon tests at recording and
patient level (baselines_analyses/patient_ids.py).  Aggregates only.
  python baselines_analyses/activation_site_error.py [--id LOC3] [--layouts template kmeans] [--frac 0.10]
  -> generated/site_error/<split>_<layout>.csv, site_error_v2.json, tab_site_error_v2.tex, tab_site_error_km_v2.tex
"""
import os, sys, json, argparse, csv
import numpy as np
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('bsp_reconstruction', 'baselines_analyses', 'inverse_evaluation'):
    sys.path.insert(0, os.path.join(V2, d))
from paths import V2_GENERATED, assert_v2
from cache_references import cache_lookup, vc_tag
from mesh_io import load_surface, find_patient_files
from dataset import collect_split
from patient_ids import patient_map

SPLITS = [('test_internal', 'internal'), ('test_external', 'external')]
SRC = [('ours', 'Ours ($N{=}16$)'), ('allvalid', 'all-valid'), ('vconly', 'VC-only')]
END = [('early', 'earliest-activation site (mm)'), ('late', 'latest-activation site (mm)'), ('tat', 'total activation time (ms)')]
MIN_GATE = 20


def stars(p):
    return '$^{***}$' if p < 1e-3 else '$^{**}$' if p < 1e-2 else '$^{*}$' if p < 5e-2 else '$^{\\mathrm{ns}}$'


def centroid(V, at, gate, frac, which):
    v = at[gate]; q = np.nanquantile(v, frac if which == 'early' else 1 - frac)
    sel = gate.copy(); sel[gate] = (v <= q) if which == 'early' else (v >= q)
    return V[sel].mean(0)


def measures(V, at, gate, frac):
    at = at.astype(float); ok = gate & np.isfinite(at)
    if ok.sum() < MIN_GATE:
        return None
    v = at[ok]
    return dict(early=centroid(V, at, ok, frac, 'early'), late=centroid(V, at, ok, frac, 'late'),
                tat=float(np.quantile(v, 0.95) - np.quantile(v, 0.05)))


def paired(d, pmap, sids):
    d = np.asarray(d, float); sids = list(sids)
    p_rec = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    pats = sorted({pmap[s] for s in sids}); idx = {p: i for i, p in enumerate(pats)}
    s = np.zeros(len(pats)); c = np.zeros(len(pats))
    for sid, di in zip(sids, d):
        s[idx[pmap[sid]]] += di; c[idx[pmap[sid]]] += 1
    dp = s / c
    p_pat = wilcoxon(dp).pvalue if np.any(dp != 0) else 1.0
    return dict(n_rec=int(len(d)), med_rec=float(np.median(d)), p_rec=float(p_rec), n_pat=int(len(pats)),
                med_pat=float(np.median(dp)), mean_pat=float(np.mean(dp)), p_pat=float(p_pat), pat_better=int((dp < 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', default='LOC3'); ap.add_argument('--layouts', nargs='+', default=['template', 'kmeans'])
    ap.add_argument('--frac', type=float, default=0.10)
    a = ap.parse_args()
    buckets, _ = collect_split(seed=42)
    od = assert_v2(os.path.join(V2_GENERATED, 'site_error')); os.makedirs(od, exist_ok=True)
    out, meshes = {}, {}
    for layout in a.layouts:
        vtag = 'vc16_template_s42' if layout == 'template' else vc_tag(16, 'cluster', 'kmean', 42)
        for split, lab in SPLITS:
            pmap = patient_map(split); rows = []
            n_miss = {'atmap': 0, 'vc': 0, 'ref': 0, 'gate': 0}
            for sid, _ecg, torso in buckets[split]:
                if not torso:
                    continue
                fa = os.path.join(V2_GENERATED, 'atmaps', a.id, f'{split}_{layout}', sid + '.npz')
                fv = cache_lookup(split, vtag, sid + '.npz'); fr = cache_lookup(split, 'refs', sid + '.npz')
                if not os.path.exists(fa):
                    n_miss['atmap'] += 1; continue
                if fv is None:
                    n_miss['vc'] += 1; continue
                if fr is None:
                    n_miss['ref'] += 1; continue
                za, zv, zr = np.load(fa), np.load(fv), np.load(fr)
                if not bool(za['valid']) or 'ATcom' not in zr.files:
                    n_miss['ref'] += 1; continue
                pdir = os.path.dirname(torso)
                if pdir not in meshes:
                    V, _F, _ = load_surface(find_patient_files(pdir)['heart']); meshes[pdir] = V.astype(float) * 10.0   # cm -> mm
                V = meshes[pdir]
                gate = zr['qrs_goodcom'].astype(bool)
                if V.shape[0] != gate.shape[0]:
                    n_miss['ref'] += 1; continue
                com = measures(V, zr['ATcom'], gate, a.frac)
                if com is None:
                    n_miss['gate'] += 1; continue
                row = {'sid': sid, 'n_gate': int(gate.sum())}
                for key, at in (('ours', za['AT4']), ('allvalid', zr['AT2']), ('vconly', zv['AT1'])):
                    m = measures(V, at, gate, a.frac)
                    if m is None:
                        row[f'{key}_early'] = row[f'{key}_late'] = row[f'{key}_tat'] = np.nan; continue
                    row[f'{key}_early'] = float(np.linalg.norm(m['early'] - com['early']))
                    row[f'{key}_late'] = float(np.linalg.norm(m['late'] - com['late']))
                    row[f'{key}_tat'] = float(abs(m['tat'] - com['tat']))
                row['com_tat'] = com['tat']
                rows.append(row)
            with open(os.path.join(od, f'{split}_{layout}.csv'), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            res = {'n': len(rows), 'missing': n_miss, 'n_patients': len({pmap[r['sid']] for r in rows})}
            for end, _ in END:
                for key, _ in SRC:
                    v = np.array([r[f'{key}_{end}'] for r in rows], float); v = v[np.isfinite(v)]
                    res[f'{key}|{end}'] = dict(median=float(np.median(v)), q1=float(np.quantile(v, .25)), q3=float(np.quantile(v, .75)), mean=float(v.mean()), n=int(v.size))
                for alt in ('allvalid', 'vconly'):
                    ok = [r for r in rows if np.isfinite(r[f'ours_{end}']) and np.isfinite(r[f'{alt}_{end}'])]
                    res[f'ours-{alt}|{end}'] = paired([r[f'ours_{end}'] - r[f'{alt}_{end}'] for r in ok], pmap, [r['sid'] for r in ok])
            res['com_tat_median'] = float(np.median([r['com_tat'] for r in rows]))
            out[f'{layout}|{split}'] = res
            print(f'[{layout} {lab}] n={len(rows)} recordings, {res["n_patients"]} patients; missing {n_miss}; commercial TAT median {res["com_tat_median"]:.0f} ms')
            for end, nm in END:
                s = ' | '.join(f"{key} {res[f'{key}|{end}']['median']:.1f} [{res[f'{key}|{end}']['q1']:.1f}, {res[f'{key}|{end}']['q3']:.1f}]" for key, _ in SRC)
                t = ' | '.join(f"vs {alt}: pat med {res[f'ours-{alt}|{end}']['med_pat']:+.1f} ({res[f'ours-{alt}|{end}']['pat_better']}/{res[f'ours-{alt}|{end}']['n_pat']}) p={res[f'ours-{alt}|{end}']['p_pat']:.3g}; rec med {res[f'ours-{alt}|{end}']['med_rec']:+.1f} p={res[f'ours-{alt}|{end}']['p_rec']:.2g}" for alt in ('allvalid', 'vconly'))
                print(f'   {nm}: {s} || {t}')
        # ---- table
        L = [r'\begin{tabular}{llccccc}', r'\toprule',
             r' & & \multicolumn{3}{c}{median [IQR] over recordings} & \multicolumn{2}{c}{paired, patient level (Ours $-$ alt.)} \\',
             r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}',
             r'Cohort & endpoint & Ours ($N{=}16$) & all-valid & VC-only & vs.\ all-valid & vs.\ VC-only \\']
        for split, lab in SPLITS:
            L.append(r'\midrule')
            res = out[f'{layout}|{split}']
            for end, nm in END:
                c = [f"{res[f'{k}|{end}']['median']:.1f} [{res[f'{k}|{end}']['q1']:.1f}, {res[f'{k}|{end}']['q3']:.1f}]" for k, _ in SRC]
                t = [f"${res[f'ours-{alt}|{end}']['med_pat']:+.1f}$ ({res[f'ours-{alt}|{end}']['pat_better']}/{res[f'ours-{alt}|{end}']['n_pat']}){stars(res[f'ours-{alt}|{end}']['p_pat'])}" for alt in ('allvalid', 'vconly')]
                L.append(f'{lab} & {nm} & ' + ' & '.join(c + t) + r' \\')
        L += [r'\bottomrule', r'\end{tabular}']
        name = 'tab_site_error_v2.tex' if layout == 'template' else 'tab_site_error_km_v2.tex'
        open(assert_v2(os.path.join(V2_GENERATED, name)), 'w').write('\n'.join(L) + '\n'); print('wrote', name)
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'site_error_v2.json')), 'w'), indent=1)
    print('wrote site_error_v2.json')


if __name__ == '__main__':
    main()
