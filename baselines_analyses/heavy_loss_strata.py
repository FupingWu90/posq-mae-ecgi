"""Heavy channel-loss strata (npj DM Results R7, 2026-10-20): downstream AT / RT error of every N=16 method as a
function of the number of ACTIVE (valid) electrodes of the recording.  Quartile strata of the internal test set
(+ external), a threshold sweep (paired Ours@16 - all-valid inverse below / above a cut), binned curves, and the
share of the whole cohorts below each cut (from the mask cache).  Both deployment layouts.  Aggregates only
(no identifiers printed).  Outputs: generated/heavy_loss_v2.json, tab_heavy_loss{,_km}_v2.tex,
figures/fig_heavy_loss{,_km}_v2.png.
  usage: python baselines_analyses/heavy_loss_strata.py [--layouts template kmeans]
"""
import os, sys, csv, json, glob, argparse
import numpy as np
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('bsp_reconstruction', 'baselines_analyses', 'baselines_analyses'):
    sys.path.insert(0, os.path.join(V2, d))
from paths import V2_GENERATED, V2_RUNS, FIG_DIR_V2, assert_v2
import results_v2 as R
from patient_ids import patient_map
from patient_level_stats import paired_stats
from dataset import collect_split, precompute_masks
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

VC = 16
METHODS = [('ours', 'Ours@16'), ('allvalid', 'all-valid inverse'), ('vconly', 'VC-only@16'),
           ('laplacian', 'Laplace@16'), ('tps', 'TPS@16'), ('lmmse', 'LMMSE@16'), ('stmem', 'ST-MEM@16'), ('mcma', 'MCMA@16')]
SPLITS = [('test_internal', 'internal'), ('test_external', 'external')]
BINS = [100, 140, 170, 200, 230, 253]
CUTS = list(range(120, 251, 10))


def n_active_map(split):
    out = {}
    for r in csv.DictReader(open(R.ours_csv(VC, split))):
        if r['source'] == 'src4_recon_valid' and r['ref'] == 'commercial':
            try:
                out[r['sid']] = int(float(r['n_active']))
            except (TypeError, ValueError):
                pass
    return out


def cohort_active_counts():
    """{split: sorted list of n_active over ALL recordings of that role} from the mask cache (final run's keep_idx)."""
    run = sorted(glob.glob(os.path.join(V2_RUNS, 'data-*', 'LOC3__*')))[0]
    keep_idx = np.load(os.path.join(run, 'keep_idx.npy'))
    b, _ = collect_split(seed=42)
    out = {}
    for role in ('train', 'val', 'test_internal', 'test_external'):
        samples = [r for r in b[role] if r[2]]
        masks = precompute_masks(samples, keep_idx, tag='exp', verbose=False)
        out[role] = sorted(int((~np.asarray(masks[fp], bool)).sum()) for _, fp, _ in samples if fp in masks)
    out['internal_all'] = sorted(out['train'] + out['val'] + out['test_internal'])
    return out


def series(P, av, split, key, met):
    if key == 'allvalid':
        return av[(split, met)]
    d = P.get((key, VC, split), {})
    return {s: v[met] for s, v in d.items() if met in v}


def strata(na, test_sids):
    """Quartile strata of n_active; when the quartiles are degenerate (external cohort: most recordings have all
    252 electrodes) fixed cuts are used instead.  Returns (edges, which) with which(x) in 0..3."""
    v = np.array([na[s] for s in test_sids]); q = np.percentile(v, [25, 50, 75])
    if len({int(x) for x in q}) < 3:
        q = np.array([180.0, 240.0, 251.0])
    def which(x):
        if x <= q[0]: return 0
        if x <= q[1]: return 1
        if x <= q[2]: return 2
        return 3
    return q, which


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--layouts', nargs='+', default=['template', 'kmeans'])
    a = ap.parse_args()
    cohort = cohort_active_counts()
    out = {'cohort_counts': {k: {'n': len(v), 'median': float(np.median(v)), 'q1': float(np.percentile(v, 25)),
                                 'q3': float(np.percentile(v, 75)), 'min': int(min(v)), 'max': int(max(v)),
                                 'frac_le': {str(c): float(np.mean(np.array(v) <= c)) for c in CUTS}}
                             for k, v in cohort.items()}}
    print('cohort active-electrode counts:', {k: (len(v), int(np.median(v)), int(min(v)), int(max(v))) for k, v in cohort.items()})
    for layout in a.layouts:
        R.LAYOUT, R.REF = layout, 'commercial'
        P, _ = R.load_everything()
        res = {}
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex='col')
        for ci, (split, lab) in enumerate(SPLITS):
            na = n_active_map(split); pmap = patient_map(split)
            av = {(split, m): R.load_allvalid(split, m) for m in ('AT', 'RT')}
            S = {(key, met): series(P, av, split, key, met) for key, _ in METHODS for met in ('AT', 'RT')}
            common = sorted(set(na) & set(S[('ours', 'AT')]) & set(S[('allvalid', 'AT')]) & set(pmap))
            q, which = strata(na, common)
            blk = {'n': len(common), 'quartile_edges': [float(x) for x in q],
                   'n_active_min': int(min(na[s] for s in common)), 'n_active_max': int(max(na[s] for s in common))}
            # ---- quartile strata
            for k in range(4):
                sids = [s for s in common if which(na[s]) == k]
                if not sids:
                    blk[f'Q{k + 1}'] = None; continue
                row = {'n_rec': len(sids), 'n_pat': len({pmap[s] for s in sids}),
                       'n_active_range': [int(min(na[s] for s in sids)), int(max(na[s] for s in sids))]}
                for met in ('AT', 'RT'):
                    for key, _ in METHODS:
                        vals = [S[(key, met)][s] for s in sids if s in S[(key, met)] and np.isfinite(S[(key, met)][s])]
                        row[f'{met}|{key}|mean'] = float(np.mean(vals)) if vals else None
                        row[f'{met}|{key}|n'] = len(vals)
                    sub = {s: S[('ours', met)][s] for s in sids if s in S[('ours', met)]}
                    for key in ('allvalid', 'vconly', 'laplacian'):
                        alt = {s: S[(key, met)][s] for s in sids if s in S[(key, met)]}
                        row[f'{met}|ours-{key}'] = paired_stats(sub, alt, met, pmap)
                blk[f'Q{k + 1}'] = row
            # ---- threshold sweep (paired Ours - all-valid, AT and RT, below / above the cut)
            sweep = {}
            for c in CUTS:
                for side, sel in (('le', [s for s in common if na[s] <= c]), ('gt', [s for s in common if na[s] > c])):
                    e = {'n_rec': len(sel), 'n_pat': len({pmap[s] for s in sel})}
                    for met in ('AT', 'RT'):
                        sub = {s: S[('ours', met)][s] for s in sel if s in S[('ours', met)]}
                        alt = {s: S[('allvalid', met)][s] for s in sel if s in S[('allvalid', met)]}
                        e[met] = paired_stats(sub, alt, met, pmap)
                    sweep[f'{c}|{side}'] = e
            blk['sweep'] = sweep
            # ---- binned curves
            curves = {}
            for met in ('AT', 'RT'):
                for key, _ in METHODS:
                    ys, ns = [], []
                    for lo, hi in zip(BINS[:-1], BINS[1:]):
                        sel = [s for s in common if lo <= na[s] < hi and s in S[(key, met)] and np.isfinite(S[(key, met)][s])]
                        ys.append(float(np.mean([S[(key, met)][s] for s in sel])) if sel else None); ns.append(len(sel))
                    curves[f'{met}|{key}'] = {'bins': BINS, 'mean': ys, 'n': ns}
            blk['curves'] = curves
            res[split] = blk
            # ---- figure
            for ri, met in enumerate(('AT', 'RT')):
                ax = axes[ri, ci]
                xc = [(lo + hi) / 2 for lo, hi in zip(BINS[:-1], BINS[1:])]
                for key, name in METHODS:
                    if key in ('stmem', 'mcma', 'lmmse'):
                        continue
                    y = curves[f'{met}|{key}']['mean']
                    ax.plot(xc, [np.nan if v is None else v for v in y], marker='o', label=name)
                if ri == 0:
                    for x, n in zip(xc, curves['AT|ours']['n']):
                        ax.annotate(f'n={n}', (x, ax.get_ylim()[1]), ha='center', va='top', fontsize=7)
                ax.set_ylabel(f'{met} MAE vs commercial (ms)'); ax.grid(alpha=0.3)
                if ri == 0:
                    ax.set_title(f'{lab} test cohort ({layout} layout)')
                if ri == 1:
                    ax.set_xlabel('active (valid) electrodes per recording')
                if ri == 0 and ci == 0:
                    ax.legend(fontsize=8)
        fig.tight_layout()
        sfx = '' if layout == 'template' else '_km'
        fig.savefig(assert_v2(os.path.join(FIG_DIR_V2, f'fig_heavy_loss{sfx}_v2.png')), dpi=150); plt.close(fig)
        out[layout] = res
        # ---- table (quartile strata, both cohorts)
        def c(x, p=1):
            return '--' if x is None else f'{x:.{p}f}'
        def pr(r):
            return '--' if r is None else f"${r['med_rec']:+.1f}${R.stars(r['p_rec'])} / ${r['med_pat']:+.1f}${R.stars(r['p_pat'])}"
        L = [r'\begin{tabular}{llcccccccc}', r'\toprule',
             r'Cohort & Quartile ($n_{\mathrm{active}}$) & rec./pat. & all-valid & Ours & VC-only & Laplace & Ours $-$ all-valid & Ours $-$ VC-only & Ours $-$ Laplace \\',
             r'\midrule']
        for met in ('AT', 'RT'):
            L.append(r'\multicolumn{10}{l}{\emph{%s MAE vs.\ commercial reference (ms); paired median difference, recording / patient level}} \\' % met)
            for split, lab in SPLITS:
                for k in range(4):
                    r = res[split][f'Q{k + 1}']
                    if r is None:
                        continue
                    L.append(f"{lab if k == 0 else ''} & Q{k + 1} ({r['n_active_range'][0]}--{r['n_active_range'][1]}) & {r['n_rec']}/{r['n_pat']} & "
                             f"{c(r[f'{met}|allvalid|mean'])} & {c(r[f'{met}|ours|mean'])} & {c(r[f'{met}|vconly|mean'])} & {c(r[f'{met}|laplacian|mean'])} & "
                             f"{pr(r[f'{met}|ours-allvalid'])} & {pr(r[f'{met}|ours-vconly'])} & {pr(r[f'{met}|ours-laplacian'])} \\\\")
            L.append(r'\midrule')
        L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
        open(assert_v2(os.path.join(V2_GENERATED, f'tab_heavy_loss{sfx}_v2.tex')), 'w').write('\n'.join(L) + '\n')
        # ---- console summary (aggregates)
        for split, lab in SPLITS:
            b = res[split]
            print(f'[{layout}] {lab}: n={b["n"]}, n_active {b["n_active_min"]}-{b["n_active_max"]}, quartile edges {[round(x) for x in b["quartile_edges"]]}')
            for k in range(4):
                r = b[f'Q{k + 1}']
                if r is None or r['AT|ours-allvalid'] is None:
                    print(f'   Q{k + 1}: empty or too small'); continue
                pa = r['AT|ours-allvalid']; pv = r['AT|ours-vconly']; prt = r['RT|ours-allvalid']
                print(f"   Q{k + 1} n_active {r['n_active_range']} rec/pat {r['n_rec']}/{r['n_pat']}: AT all-valid {r['AT|allvalid|mean']:.2f} ours {r['AT|ours|mean']:.2f} "
                      f"vconly {r['AT|vconly|mean']:.2f} laplace {r['AT|laplacian|mean']:.2f} | ours-allvalid med {pa['med_rec']:+.2f} p={pa['p_rec']:.3f} (pat {pa['med_pat']:+.2f} p={pa['p_pat']:.2f}) "
                      f"| ours-vconly {pv['med_rec']:+.2f} p={pv['p_rec']:.1e} | RT ours-allvalid {prt['med_rec']:+.2f} p={prt['p_rec']:.3f}")
            print('   sweep (AT, Ours - all-valid, recordings with n_active <= cut): ' + '; '.join(
                f"{c}: n={b['sweep'][f'{c}|le']['n_rec']} med {b['sweep'][f'{c}|le']['AT']['med_rec']:+.2f} p={b['sweep'][f'{c}|le']['AT']['p_rec']:.2g}"
                for c in CUTS if b['sweep'][f'{c}|le']['AT']))
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'heavy_loss_v2.json')), 'w'), indent=1)
    print('wrote heavy_loss_v2.json, tab_heavy_loss*_v2.tex, fig_heavy_loss*_v2.png')


if __name__ == '__main__':
    main()
