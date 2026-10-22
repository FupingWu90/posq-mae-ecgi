"""Hold-out replication split (run CARTOHO = final configuration retrained with 9 internal patients held out for a
planned invasive comparison; 2026-10-20): aggregates of the per-recording downstream evaluation of the held-out
patients under both deployment layouts, next to the internal test set of the final model (LOC3) for reference.
  python baselines_analyses/holdout_summary.py [--id CARTOHO] [--ref_id LOC3]
  -> generated/holdout_replication_v2.json, generated/tab_holdout_v2.tex
The all-valid-inverse-vs-commercial errors follow the paper's convention (results_v2.load_allvalid: refs/<sid>.npz of the
reference cache, gated by the commercial reference's good vertices), so the hold-out split needs
`inverse_evaluation/cache_references.py --split holdout` first.  Aggregates only.  Patient identity and content de-duplication follow the paper's identity map (v1 common._identity: union-find over
study numbers and shared content); keys are opaque and never printed or written; one recording per content key."""
import os, sys, csv, json, glob, argparse
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'baselines_analyses')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
from cache_references import CACHE_ROOT, CACHE_ROOT_V1
from paths import V2_GENERATED, V1_ROOT, assert_v2
from dataset import collect_split
from patient_level_stats import paired_stats
from patient_ids import patient_map

HOLDOUT_FILE = os.path.join(V2, 'training_sweeps', 'private', 'holdout_invasive.txt')
LAYOUTS = ['template', 'kmeans']
OURS, VC = 'src4_recon_valid', 'src1_vc'


def stars(p):
    if not np.isfinite(p):
        return ''
    return '$^{***}$' if p < 1e-3 else '$^{**}$' if p < 1e-2 else '$^{*}$' if p < 5e-2 else '$^{\\mathrm{ns}}$'


def _identity_maps():
    """content md5 per internal file and patient id per (cohort, folder): the paper's identity map (v1 common._identity,
    union-find over study numbers and shared content; read-only use of the frozen v1 tree)."""
    sys.path.insert(0, os.path.join(V1_ROOT, 'baselines_analyses'))
    from common import _identity
    return _identity()


def holdout_patient_map(run_id='CARTOHO'):
    """{sid: patient id} of the hold-out role, plus {sid: content md5}; the hold-out list is read from the run's args.json
    (--holdout_file; the private default is the fallback)."""
    hf = HOLDOUT_FILE
    runs = glob.glob(os.path.join(V2, 'runs', '*', f'{run_id}__*'))
    if runs and os.path.exists(os.path.join(runs[0], 'args.json')):
        hf = json.load(open(os.path.join(runs[0], 'args.json'))).get('holdout_file') or hf
    hold = set(l.strip() for l in open(hf) if l.strip())
    b, _ = collect_split(seed=42, holdout=hold)
    md5, pid = _identity_maps()
    pm, cm = {}, {}
    for sid, ecg, _t in b['holdout']:
        cohort, folder = sid.split('__', 2)[:2]
        pm[sid] = pid[(cohort, folder)]
        cm[sid] = md5.get(ecg, ecg)
    return pm, cm


def test_maps(split='test_internal'):
    b, _ = collect_split(seed=42)
    md5, _pid = _identity_maps()
    return patient_map(split), {sid: md5.get(ecg, ecg) for sid, ecg, _t in b[split]}


def dedup(sids, cm):
    """one sid per content key (first in sorted order): the paper counts unique recordings."""
    seen, out = set(), []
    for s in sorted(sids):
        k = cm.get(s, s)
        if k in seen:
            continue
        seen.add(k); out.append(s)
    return out


def load_allvalid(split, met):
    """paper convention: per-recording |AT2 - ATcom| (or RT) on the commercial-gated vertices, keyed by sid_safe."""
    out, files = {}, {}
    for root in (CACHE_ROOT, CACHE_ROOT_V1):
        for f in glob.glob(os.path.join(root, split, 'refs', '*.npz')):
            files.setdefault(os.path.basename(f), f)
    gate = 'qrs_goodcom' if met == 'AT' else 't_goodcom'
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    for name, f in files.items():
        if not _dedup_ok(name[:-4]):
            continue
        with np.load(f, allow_pickle=False) as z:
            if 'AT2' not in z.files or 'ATcom' not in z.files:
                continue
            g = z[gate].astype(bool)
            if g.sum() < 3:
                continue
            a, b = z[f'{met}2'].astype(float), z[f'{met}com'].astype(float)
            out[name[:-4]] = float(np.abs(a - b)[g].mean())
    return out


CLASSICAL = [('lmmse', 'LMMSE'), ('laplacian', 'Laplace interpolation'), ('tps', 'TPS interpolation')]


def load_classical(layout, split='holdout', vc=16):
    """{method: {sid: {'AT': at_com, 'RT': rt_com}}} from classical_template/percase_<layout>.csv (empty if absent)."""
    path = os.path.join(V2_GENERATED, 'classical_template', f'percase_{layout}.csv')
    out = {m: {} for m, _ in CLASSICAL}
    if not os.path.exists(path):
        return out
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    for r in csv.DictReader(open(path)):
        if r['split'] != split or int(r['vc']) != vc or r['method'] not in out or not _dedup_ok(r['sid']):
            continue
        d = {}
        for met, col in (('AT', 'at_com'), ('RT', 'rt_com')):
            try:
                v = float(r[col])
                if np.isfinite(v):
                    d[met] = v
            except (TypeError, ValueError):
                pass
        out[r['method']][r['sid']] = d
    return out


def load(path):
    """{(source, ref): {sid: row}}"""
    out = {}
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    for r in csv.DictReader(open(path)):
        if not _dedup_ok(r['sid']):
            continue
        out.setdefault((r['source'], r['ref']), {})[r['sid']] = r
    return out


def fnum(r, k):
    try:
        v = float(r[k]); return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError, KeyError):
        return np.nan


def summarise(path, pmap, split, layout, cmap=None):
    T = load(path)
    AV = {met: load_allvalid(split, met) for met in ('AT', 'RT')}
    CL = load_classical(layout, split) if split == 'holdout' else {m: {} for m, _ in CLASSICAL}
    ours_c, vc_c = T[(OURS, 'commercial')], T[(VC, 'commercial')]
    ours_a, vc_a = T[(OURS, 'src2_allvalid')], T[(VC, 'src2_allvalid')]
    sids = sorted(set(ours_c) & set(vc_c) & set(pmap))
    if cmap:
        sids = dedup(sids, cmap)
    pats = {pmap[s] for s in sids}
    res = dict(n_rec=len(sids), n_pat=len(pats))
    for met in ('AT', 'RT'):
        allv = {s: AV[met].get(s.replace('/', '_'), np.nan) for s in sids}
        ours = {s: fnum(ours_c[s], f'{met}_mae') for s in sids}
        vco = {s: fnum(vc_c[s], f'{met}_mae') for s in sids}
        res[met] = dict(mean_allvalid=float(np.nanmean(list(allv.values()))),
                        mean_ours=float(np.nanmean(list(ours.values()))),
                        mean_vconly=float(np.nanmean(list(vco.values()))),
                        mean_ours_vs_allvalid=float(np.nanmean([fnum(ours_a[s], f'{met}_mae') for s in sids if s in ours_a])),
                        mean_vconly_vs_allvalid=float(np.nanmean([fnum(vc_a[s], f'{met}_mae') for s in sids if s in vc_a])),
                        paired_ours_allvalid=paired_stats(ours, allv, None, pmap),
                        paired_ours_vconly=paired_stats(ours, vco, None, pmap))
        for m, _ in CLASSICAL:
            alt = {s: CL[m].get(s, {}).get(met, np.nan) for s in sids}
            fin = [v for v in alt.values() if np.isfinite(v)]
            res[met][f'mean_{m}'] = float(np.mean(fin)) if fin else None
            res[met][f'paired_ours_{m}'] = paired_stats(ours, alt, None, pmap) if fin else None
    res['bsp_cc_ours'] = float(np.nanmean([fnum(ours_c[s], 'bsp_cc') for s in sids]))
    res['hsp_cc_ours_vs_commercial'] = float(np.nanmean([fnum(ours_c[s], 'hsp_cc') for s in sids]))
    res['hsp_cc_ours_vs_allvalid'] = float(np.nanmean([fnum(ours_a[s], 'hsp_cc') for s in sids if s in ours_a]))
    return res


def cell(p):
    if p is None:
        return '--'
    return (f"${p['med_rec']:+.1f}$" + stars(p['p_rec']) + ' / ' + f"${p['med_pat']:+.1f}$" + stars(p['p_pat']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', default='CARTOHO'); ap.add_argument('--ref_id', default='LOC3')
    a = ap.parse_args()
    out = {'holdout_run': a.id, 'reference_run': a.ref_id, 'layouts': {}}
    (pm_h, cm_h), (pm_t, cm_t) = holdout_patient_map(a.id), test_maps('test_internal')
    for lay in LAYOUTS:
        ph = os.path.join(V2_GENERATED, 'percase_holdout', f'{a.id}_vc16_holdout_{lay}.csv')
        pt = os.path.join(V2_GENERATED, 'percase_candidate', f'{a.ref_id}_vc16_test_internal_{lay}.csv')
        d = {}
        if os.path.exists(pt):
            d['test_internal'] = summarise(pt, pm_t, 'test_internal', lay, cm_t)
        if os.path.exists(ph):
            d['holdout'] = summarise(ph, pm_h, 'holdout', lay, cm_h)
        out['layouts'][lay] = d
    assert_v2(V2_GENERATED)
    json.dump(out, open(os.path.join(V2_GENERATED, 'holdout_replication_v2.json'), 'w'), indent=1)
    # console (aggregates only)
    for lay, d in out['layouts'].items():
        for split, r in d.items():
            print(f"[{lay:8s}] {split:14s} n_rec={r['n_rec']} n_pat={r['n_pat']} BSP CC={r['bsp_cc_ours']:.3f} "
                  f"HSP CC vs com/allv={r['hsp_cc_ours_vs_commercial']:.3f}/{r['hsp_cc_ours_vs_allvalid']:.3f}")
            for met in ('AT', 'RT'):
                m = r[met]; pa, pv = m['paired_ours_allvalid'], m['paired_ours_vconly']
                print(f"    {met}: all-valid {m['mean_allvalid']:.2f} | Ours {m['mean_ours']:.2f} | VC-only {m['mean_vconly']:.2f} "
                      f"(vs all-valid ref: Ours {m['mean_ours_vs_allvalid']:.2f}, VC-only {m['mean_vconly_vs_allvalid']:.2f}) | "
                      f"Ours-allv med rec {pa['med_rec']:+.2f} p={pa['p_rec']:.2g}, pat {pa['med_pat']:+.2f} p={pa['p_pat']:.2g} "
                      f"({pa['pat_better']}/{pa['n_pat']} pts better) | Ours-VC pat {pv['med_pat']:+.2f} p={pv['p_pat']:.2g}")
                for mkey, mname in CLASSICAL:
                    q = m.get(f'paired_ours_{mkey}')
                    if q:
                        print(f"        vs {mname}: mean {m[f'mean_{mkey}']:.2f} | Ours-{mkey} med rec {q['med_rec']:+.2f} p={q['p_rec']:.2g}, pat {q['med_pat']:+.2f} p={q['p_pat']:.2g} ({q['pat_better']}/{q['n_pat']} better)")
    # LaTeX table body (same structure as Table 2 of the main text; the two layouts side by side)
    H = {lay: out['layouts'][lay].get('holdout') for lay in LAYOUTS}
    if all(H.values()):
        def pc(r):
            if r is None:
                return '-- & --'
            return (f"${r['med_rec']:+.1f}$" + stars(r['p_rec']) + f" & ${r['med_pat']:+.1f}$ ({r['pat_better']}/{r['n_pat']})" + stars(r['p_pat']))
        alts = [('allvalid', 'all-valid inverse ($252$ electrodes)'), ('vconly', 'VC-only inverse')] + [(m, n) for m, n in CLASSICAL]
        nrec = H['template']['AT']['paired_ours_allvalid']['n_rec']; npat = H['template']['n_pat']
        L = [r'\begin{tabular}{lcccccc}', r'\toprule',
             r' & \multicolumn{3}{c}{fixed template layout} & \multicolumn{3}{c}{patient-specific $k$-means layout} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'Alternative & MAE & Ours $-$ alt.\ (rec.) & Ours $-$ alt.\ (pat.) & MAE & Ours $-$ alt.\ (rec.) & Ours $-$ alt.\ (pat.) \\',
             r'\midrule']
        for met, name in (('AT', 'Activation time'), ('RT', 'Repolarisation time')):
            L.append(r'\multicolumn{7}{l}{\emph{%s, ms: Ours (PosQ-MAE retrained without the held-out patients, $16$ electrodes) %.1f template, %.1f $k$-means}} \\'
                     % (name, H['template'][met]['mean_ours'], H['kmeans'][met]['mean_ours']))
            for key, label in alts:
                cells = []
                for lay in LAYOUTS:
                    m = H[lay][met]
                    mean = m['mean_allvalid'] if key == 'allvalid' else m['mean_vconly'] if key == 'vconly' else m.get(f'mean_{key}')
                    pr = m['paired_ours_allvalid'] if key == 'allvalid' else m['paired_ours_vconly'] if key == 'vconly' else m.get(f'paired_ours_{key}')
                    cells.append(('--' if mean is None else f'{mean:.1f}') + ' & ' + pc(pr))
                L.append(f'{label} & ' + ' & '.join(cells) + r' \\')
            L.append(r'\midrule')
        L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
        body = '\n'.join(L) + '\n'
        print(f'table: {nrec} recordings / {npat} patients (AT pairs)')
    else:
        body = '% hold-out results missing\n'
    open(os.path.join(V2_GENERATED, 'tab_holdout_v2.tex'), 'w').write(body)
    print('wrote holdout_replication_v2.json, tab_holdout_v2.tex')


if __name__ == '__main__':
    main()
