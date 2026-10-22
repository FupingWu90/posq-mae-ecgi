"""Summarise the emulated heavy channel-loss experiment (npj DM R7 / Fig. 5, 2026-10-20).
Reads generated/percase_loss/LOC3_vc16_<split>_<layout>_loss{25,50,75}s1.csv (sources: src1_vc = VC-only@16,
src4_recon_valid = Ours@16, src2_reduced = the reduced vest's own inverse; reference: commercial) plus the loss-0
baselines (percase_candidate LOC3 CSVs and the cached all-valid inverse).  Outputs heavy_loss_emulated_v2.json,
tab_extra_loss_v2.tex and fig_extra_loss_npjdm_v2.png (figures/ + npjdm package).  Aggregates only.
  usage: python baselines_analyses/extra_loss_summary.py [--layouts template kmeans]
"""
import os, sys, csv, json, glob, argparse, shutil
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('bsp_reconstruction', 'baselines_analyses', 'baselines_analyses'):
    sys.path.insert(0, os.path.join(V2, d))
from paths import V2_GENERATED, FIG_DIR_V2, assert_v2
import results_v2 as R
from patient_ids import patient_map
from patient_level_stats import paired_stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

LOSSES = [0, 25, 50, 75]
SPLITS = [('test_internal', 'internal'), ('test_external', 'external')]
PKG = os.environ.get('ECGI_PAPER_FIG_DIR')          # optional: paper package figures/ dir to copy fig5 into


def read(path):
    """{(source): {sid: row}} for ref == commercial; rows keep n_active, n_active_orig, n_template_sub, AT_mae, RT_mae, bsp_cc."""
    out = {}
    if not os.path.exists(path):
        return out
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    for r in csv.DictReader(open(path)):
        if r['ref'] != 'commercial' or not _dedup_ok(r['sid']):
            continue
        d = out.setdefault(r['source'], {})
        def _f(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return np.nan
        if not np.isfinite(_f(r['n_active'])):
            continue
        d[r['sid']] = {'AT': _f(r['AT_mae']), 'RT': _f(r['RT_mae']), 'n_active': _f(r['n_active']),   # fields parsed independently (RT may be blank: T-gate rule, 2026-10-20)
                       'n_active_orig': _f(r.get('n_active_orig') or r['n_active']), 'n_sub': _f(r.get('n_template_sub') or 0),
                       'bsp': _f(r['bsp_cc']) if r.get('bsp_cc') not in (None, '') else np.nan}
    return out


def series(d, met):
    return {s: v[met] for s, v in d.items() if np.isfinite(v[met])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--layouts', nargs='+', default=['template', 'kmeans'])
    a = ap.parse_args()
    out = {}
    for layout in a.layouts:
        R.LAYOUT, R.REF = layout, 'commercial'
        res = {}
        for split, lab in SPLITS:
            pmap = patient_map(split)
            base = read(R.ours_csv(16, split))
            av = {met: R.load_allvalid(split, met) for met in ('AT', 'RT')}
            for loss in LOSSES:
                if loss == 0:
                    D = {'ours': base.get('src4_recon_valid', {}), 'vconly': base.get('src1_vc', {}), 'reduced': None}
                else:
                    f = os.path.join(V2_GENERATED, 'percase_loss', f'LOC3_vc16_{split}_{layout}_loss{loss}s1.csv')
                    P = read(f)
                    if not P:
                        continue
                    D = {'ours': P.get('src4_recon_valid', {}), 'vconly': P.get('src1_vc', {}), 'reduced': P.get('src2_reduced', {})}
                e = {'n_rec': len(D['ours'])}
                if D['ours']:
                    e['n_active_mean'] = float(np.mean([v['n_active'] for v in D['ours'].values()]))
                    e['n_active_orig_mean'] = float(np.mean([v['n_active_orig'] for v in D['ours'].values()]))
                    e['n_template_sub_mean'] = float(np.mean([v['n_sub'] for v in D['ours'].values()]))
                    e['bsp_cc_ours'] = float(np.nanmean([v['bsp'] for v in D['ours'].values()]))
                for met in ('AT', 'RT'):
                    so = series(D['ours'], met); sv = series(D['vconly'], met)
                    sr = av[met] if D['reduced'] is None else series(D['reduced'], met)
                    common = sorted(set(so) & set(sr))
                    e[f'{met}|ours'] = float(np.mean([so[s] for s in common])) if common else None
                    e[f'{met}|reduced'] = float(np.mean([sr[s] for s in common])) if common else None
                    e[f'{met}|vconly'] = float(np.mean([sv[s] for s in common if s in sv])) if common else None
                    e[f'{met}|ours-reduced'] = paired_stats(so, sr, met, pmap)
                    e[f'{met}|ours-vconly'] = paired_stats(so, sv, met, pmap)
                res[f'{split}|{loss}'] = e
        out[layout] = res
        # ---- console
        for split, lab in SPLITS:
            print(f'[{layout}] {lab}:')
            for loss in LOSSES:
                e = res.get(f'{split}|{loss}')
                if not e or e.get('AT|ours') is None:
                    print(f'   loss {loss}%: (not available yet)'); continue
                pr = e['AT|ours-reduced']; prt = e['RT|ours-reduced']
                print(f"   extra loss {loss:2d}%: n={e['n_rec']} valid electrodes {e['n_active_mean']:.0f} (orig {e['n_active_orig_mean']:.0f}), template subs {e['n_template_sub_mean']:.1f} | "
                      f"AT reduced-vest {e['AT|reduced']:.2f} Ours@16 {e['AT|ours']:.2f} VC-only {e['AT|vconly']:.2f} | Ours-reduced med {pr['med_rec']:+.2f} p={pr['p_rec']:.2g} (pat {pr['med_pat']:+.2f} p={pr['p_pat']:.2g}) | "
                      f"RT reduced {e['RT|reduced']:.2f} Ours {e['RT|ours']:.2f} (med {prt['med_rec']:+.2f} p={prt['p_rec']:.2g}) | BSP CC {e['bsp_cc_ours']:.3f}")
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'heavy_loss_emulated_v2.json')), 'w'), indent=1)
    # ---- table (template layout, both cohorts; k-means internal rows appended when present)
    def c(x, p=1):
        return '--' if x is None else f'{x:.{p}f}'
    def pr_(r):
        return '--' if r is None else f"${r['med_rec']:+.1f}${R.stars(r['p_rec'])} / ${r['med_pat']:+.1f}${R.stars(r['p_pat'])}"
    L = [r'\begin{tabular}{llccccccc}', r'\toprule',
         r'Cohort & extra loss & valid electrodes & \multicolumn{3}{c}{AT MAE (ms)} & Ours $-$ reduced vest & \multicolumn{2}{c}{RT MAE (ms)} \\',
         r'\cmidrule(lr){4-6}\cmidrule(lr){8-9}',
         r' & & (mean) & reduced vest & Ours@16 & VC-only@16 & AT, rec.\ / pat. & reduced vest & Ours@16 \\', r'\midrule']
    for layout in a.layouts:
        for split, lab in SPLITS:
            if layout == 'kmeans' and split == 'test_external':
                continue
            for loss in LOSSES:
                e = out[layout].get(f'{split}|{loss}')
                if not e or e.get('AT|ours') is None:
                    continue
                row_lab = (lab + (' ($k$-means)' if layout == 'kmeans' else '')) if loss == 0 else ''
                loss_lab = 'none' if loss == 0 else str(loss) + '\\%'
                cells = [row_lab, loss_lab, c(e['n_active_mean'], 0), c(e['AT|reduced']), c(e['AT|ours']), c(e['AT|vconly']),
                         pr_(e['AT|ours-reduced']), c(e['RT|reduced']), c(e['RT|ours'])]
                L.append(' & '.join(cells) + ' \\\\')
            L.append(r'\midrule')
    L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_extra_loss_v2.tex')), 'w').write('\n'.join(L) + '\n')
    # ---- figure (template layout): AT and RT versus mean valid electrodes
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ci, (split, lab) in enumerate(SPLITS):
        pts = [(loss, out['template'][f'{split}|{loss}']) for loss in LOSSES if f'{split}|{loss}' in out['template'] and out['template'][f'{split}|{loss}'].get('AT|ours') is not None]
        for ri, met in enumerate(('AT', 'RT')):
            ax = axes[ri, ci]
            x = [e['n_active_mean'] for _, e in pts]
            for key, name, st in (('reduced', 'inverse from all remaining electrodes', 'o--'), ('ours', 'Ours (16 electrodes)', 'o-'), ('vconly', 'VC-only inverse (16 electrodes)', 'o:')):
                ax.plot(x, [e[f'{met}|{key}'] for _, e in pts], st, label=name)
            for loss, e in pts:
                ax.annotate('observed' if loss == 0 else f'+{loss}%', (e['n_active_mean'], e[f'{met}|ours']), textcoords='offset points', xytext=(0, 8 if loss == 0 else -14), ha='center', fontsize=8)
            ax.invert_xaxis(); ax.grid(alpha=0.3)
            ax.set_ylabel(f'{met} MAE vs commercial reference (ms)')
            if ri == 0:
                ax.set_title(f'{lab} test cohort')
            if ri == 1:
                ax.set_xlabel('valid electrodes per recording (mean)')
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    p = assert_v2(os.path.join(FIG_DIR_V2, 'fig_extra_loss_npjdm_v2.png')); fig.savefig(p, dpi=150); plt.close(fig)
    if PKG and os.path.isdir(PKG):
        shutil.copy(p, os.path.join(PKG, 'fig_extra_loss_npjdm_v2.png'))
    print('wrote heavy_loss_emulated_v2.json, tab_extra_loss_v2.tex, fig_extra_loss_npjdm_v2.png')


if __name__ == '__main__':
    main()
