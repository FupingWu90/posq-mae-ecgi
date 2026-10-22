"""Patient-level paired comparisons: the recordings of one patient are repeated measures, so the per-recording
tests of results_v2.py (n = 276 / 251 recordings) overstate the effective sample size.  Here, for every patient, the
per-recording error (level-1 mean over the reference-gated vertices, commercial reference) is averaged over the
patient's recordings for Ours and for the alternative, the paired difference is formed per patient, and the median
paired difference, the number of patients favouring Ours and a two-sided Wilcoxon signed-rank test over patients
(exact for the 11 internal patients) are reported; the recording-level values are kept for cross-reference.
Comparisons: Ours (PosQ-MAE) vs all-valid at every N, and vs VC-only, ST-MEM, MCMA, LMMSE, Laplace, TPS at N=16;
AT and RT; both deployment layouts.  Aggregates only (no identifiers printed).
  python baselines_analyses/patient_level_stats.py [--layouts template kmeans]
  -> generated/patient_level_v2.json, tab_patient_level_v2.tex, tab_patient_level_km_v2.tex
"""
import os, sys, json, argparse
import numpy as np
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('bsp_reconstruction', 'baselines_analyses', 'inverse_evaluation'):
    sys.path.insert(0, os.path.join(V2, d))
from paths import V2_GENERATED, assert_v2
from patient_ids import patient_map
import results_v2 as R

SPLITS = [('test_internal', 'internal'), ('test_external', 'external')]
ALTS = [('other', 'all-valid'), ('vconly', 'VC-only'), ('stmem', 'ST-MEM'), ('mcma', 'MCMA'), ('lmmse', 'LMMSE'),
        ('laplacian', 'Laplace'), ('tps', 'TPS')]


def paired_stats(ours, alt, met, pmap):
    """ours/alt: {sid: value or {met: value}}; returns dict with recording- and patient-level statistics."""
    get = lambda d, s: (d[s].get(met, np.nan) if isinstance(d[s], dict) else d[s])
    sids = [s for s in sorted(set(ours) & set(alt)) if s in pmap]
    x = np.array([get(ours, s) for s in sids], float); y = np.array([get(alt, s) for s in sids], float)
    ok = np.isfinite(x) & np.isfinite(y)
    sids = [s for s, o in zip(sids, ok) if o]; x, y = x[ok], y[ok]
    if len(x) < 5:
        return None
    d = x - y
    p_rec = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    pats = sorted({pmap[s] for s in sids})
    idx = {p: i for i, p in enumerate(pats)}
    sx = np.zeros(len(pats)); sy = np.zeros(len(pats)); cnt = np.zeros(len(pats))
    for s, xi, yi in zip(sids, x, y):
        i = idx[pmap[s]]; sx[i] += xi; sy[i] += yi; cnt[i] += 1
    px, py = sx / cnt, sy / cnt
    dp = px - py
    p_pat = wilcoxon(dp).pvalue if np.any(dp != 0) else 1.0
    return dict(n_rec=int(len(d)), med_rec=float(np.median(d)), p_rec=float(p_rec),
                n_pat=int(len(pats)), med_pat=float(np.median(dp)), mean_pat=float(np.mean(dp)), p_pat=float(p_pat),
                pat_better=int((dp < 0).sum()), pat_worse=int((dp > 0).sum()),
                mean_ours_pat=float(np.mean(px)), mean_alt_pat=float(np.mean(py)))


def cell(r):
    if r is None:
        return '--'
    return f"${r['med_pat']:+.1f}$ ({r['pat_better']}/{r['n_pat']}){R.stars(r['p_pat'])}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--layouts', nargs='+', default=['template', 'kmeans'])
    a = ap.parse_args()
    out = {}
    for layout in a.layouts:
        R.LAYOUT, R.REF = layout, 'commercial'
        P, _ = R.load_everything()
        res = {}
        for split, lab in SPLITS:
            pmap = patient_map(split)
            av = {met: R.load_allvalid(split, met) for met in ('AT', 'RT')}
            for vc in R.VCS:
                ours = P[('ours', vc, split)]
                for key, _ in ALTS:
                    if key != 'other' and vc != 16:
                        continue
                    for met in ('AT', 'RT'):
                        alt = av[met] if key == 'other' else P[(key, vc, split)]
                        res[f'{split}|{met}|{vc}|{key}'] = paired_stats(ours, alt, met, pmap)
        out[layout] = res
        # ---- table
        L = [r'\begin{tabular}{llcccc}', r'\toprule',
             r' & & \multicolumn{2}{c}{internal ($n{=}11$ patients)} & \multicolumn{2}{c}{external ($n{=}202$ patients)} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}', r'Ours vs. & $N$ & AT & RT & AT & RT \\', r'\midrule']
        for key, name in ALTS:
            for vc in (R.VCS if key == 'other' else [16]):
                cells = [cell(res[f'{split}|{met}|{vc}|{key}']) for split, _ in SPLITS for met in ('AT', 'RT')]
                L.append(f'{name} & {vc} & ' + ' & '.join(cells) + r' \\')
            if key == 'other':
                L.append(r'\midrule')
        L += [r'\bottomrule', r'\end{tabular}']
        name = 'tab_patient_level_v2.tex' if layout == 'template' else 'tab_patient_level_km_v2.tex'
        open(assert_v2(os.path.join(V2_GENERATED, name)), 'w').write('\n'.join(L) + '\n'); print('wrote', name)
        for split, lab in SPLITS:
            for key, nm in ALTS:
                r = {met: res[f'{split}|{met}|16|{key}'] for met in ('AT', 'RT')}
                print(f"  [{layout} {lab} N=16] vs {nm:9s}: " + ' | '.join(
                    f"{met} patient med {v['med_pat']:+.2f} ms (mean {v['mean_pat']:+.2f}; {v['pat_better']}/{v['n_pat']} better; p={v['p_pat']:.3g}) "
                    f"rec med {v['med_rec']:+.2f} p={v['p_rec']:.2g}" if v else f'{met} --' for met, v in r.items()))
            for vc in R.VCS:
                v = res[f'{split}|AT|{vc}|other']
                if v and vc != 16:
                    print(f"  [{layout} {lab} N={vc}] vs all-valid AT: patient med {v['med_pat']:+.2f} ({v['pat_better']}/{v['n_pat']}; p={v['p_pat']:.3g}); RT {res[f'{split}|RT|{vc}|other']['med_pat']:+.2f} (p={res[f'{split}|RT|{vc}|other']['p_pat']:.3g})")
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'patient_level_v2.json')), 'w'), indent=1)
    print('wrote patient_level_v2.json')


if __name__ == '__main__':
    main()
