"""Summarise the reviewer sensitivity reruns (baselines_analyses/sources_rerun_v2.py; generated/percase_rerun/) into the tables of
Supplementary Note 21 and a JSON of aggregates (2026-10-23).  Aggregates only are printed.

Tables (generated/):
  tab_sens_variants_v2.tex  N = 16, both cohorts: cohort-mean AT / RT MAE against the commercial reference of the
                            VC-only inverse, the full-vest inverse and the completion under every variant, with the
                            mean paired change of the completion (and of the raw inverses) relative to the paper's
                            evaluation (base) and a Wilcoxon test over recordings
  tab_sens_vconly_n_v2.tex  VC-only inverse at N = 4..64: AT / RT with Neumann rows at the active electrodes (paper)
                            and at all torso nodes, paired change
  tab_sens_timebase_v2.tex  base evaluation: per-recording time shift between sources (mean signed AT difference),
                            AT MAE before and after removing it, fraction of gated vertices on the QRS-window edges
  tab_sens_filter_v2.tex    0.5-Hz high-pass: change of the mean gated AT / RT of each source and of the reference
                            (absolute marker shift) and of the errors
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from paths import V2_GENERATED, assert_v2
D = os.path.join(V2_GENERATED, 'percase_rerun')
SRC = {'src1_vc': 'VC-only inverse', 'src2_allvalid': 'full-vest inverse', 'src4_recon_valid': 'PosQ-MAE', 'src3_recon_whole': 'full vest + completed missing'}
VARIANTS = [('base', 'paper evaluation'), ('neumann', 'zero-flux rows at all torso nodes'), ('oracle', 'oracle $\\lambda$ per recording (VC-only)'),
            ('nosub', 'no substitution of failed template electrodes'), ('reref_car', 'visible channels re-referenced to their mean'),
            ('reref_ch0', 'visible channels referenced to one electrode'), ('hp05', 'high-pass cut-off $0.5$\\,Hz, reference alike'),
            ('qrs120', 'QRS window $[R-50,\\,R+120]$\\,ms'), ('rpeaks_bsp', 'shared time base (R peaks from the body surface)'),
            ('visall', 'all valid electrodes visible, only missing ones completed')]
COH = [('test_internal', 'internal'), ('test_external', 'external')]


def load(tag, split, vc=16):
    p = os.path.join(D, f'{tag}_vc{vc}_{split}_template.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p); df = df[df.success == 1]
    return df


def stars(p):
    """significance superscript, to be placed INSIDE the math group of the number"""
    return '' if not np.isfinite(p) else '^{***}' if p < 1e-3 else '^{**}' if p < 1e-2 else '^{*}' if p < 0.05 else '^{\\mathrm{ns}}'


def paired(a, b, col):
    """mean paired change (a - b) over common recordings with Wilcoxon signed-rank p; a/b: DataFrames of one source vs commercial."""
    m = a.set_index('sid')[col].to_frame('x').join(b.set_index('sid')[col].to_frame('y'), how='inner').dropna()
    if len(m) < 5:
        return float('nan'), float('nan'), 0
    d = m.x - m.y
    return float(d.mean()), float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, int(len(d))


def main():
    out = {}
    base = {sp: load('base', sp) for sp, _ in COH}
    # ---------------- variants table
    L = [r'\begin{tabular}{llcccccc}', r'\toprule', r' & & \multicolumn{3}{c}{AT MAE (ms)} & \multicolumn{3}{c}{RT MAE (ms)} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}', r'Cohort & Evaluation & VC-only & full vest & PosQ-MAE & VC-only & full vest & PosQ-MAE \\']
    for sp, lab in COH:
        L.append(r'\midrule'); first = True
        for tag, name in VARIANTS:
            df = load(tag, sp)
            if df is None:
                continue
            cells = []
            for met in ('AT', 'RT'):
                for s in ('src1_vc', 'src2_allvalid', 'src4_recon_valid'):
                    sub = df[(df.source == s) & (df.ref == 'commercial')]
                    if tag == 'visall' and s == 'src4_recon_valid':
                        sub = df[(df.source == 'src3_recon_whole') & (df.ref == 'commercial')]
                    if len(sub) == 0:
                        cells.append('--'); continue
                    mean = float(np.nanmean(sub[f'{met}_mae']))
                    cell = f'{mean:.1f}'
                    if tag != 'base' and base[sp] is not None:
                        bsub = base[sp][(base[sp].source == s) & (base[sp].ref == 'commercial')]
                        md, p, n = paired(sub, bsub, f'{met}_mae')
                        if np.isfinite(md):
                            cell += f' (${md:+.2f}{stars(p)}$)'
                    cells.append(cell)
                    out[f'{lab}|{tag}|{s}|{met}'] = mean
            extra = ''
            if tag == 'nosub':
                extra = f" ($N_{{\\mathrm{{eff}}}}$ {float(df[df.source == 'src4_recon_valid'].n_vc.mean()):.1f})"
            if tag == 'visall':
                extra = f" ($N$ {float(df[df.source == 'src3_recon_whole'].n_vc.mean()):.0f})"
            L.append(f"{lab if first else ''} & {name}{extra} & " + ' & '.join(cells) + r' \\'); first = False
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_sens_variants_v2.tex')), 'w').write('\n'.join(L) + '\n')
    # ---------------- VC-only vs N
    L = [r'\begin{tabular}{lccccccc}', r'\toprule', r' & & \multicolumn{3}{c}{AT MAE (ms)} & \multicolumn{3}{c}{RT MAE (ms)} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}', r'Cohort & $N$ & active rows & all nodes & change & active rows & all nodes & change \\']
    for sp, lab in COH:
        L.append(r'\midrule'); first = True
        for vc in (4, 8, 16, 32, 64):
            b = load('base', sp, vc); n = load('neumann', sp, vc)
            if b is None or n is None:
                continue
            b1 = b[(b.source == 'src1_vc') & (b.ref == 'commercial')]; n1 = n[(n.source == 'src1_vc') & (n.ref == 'commercial')]
            cells = []
            for met in ('AT', 'RT'):
                md, p, _ = paired(n1, b1, f'{met}_mae')
                cells += [f"{float(np.nanmean(b1[f'{met}_mae'])):.1f}", f"{float(np.nanmean(n1[f'{met}_mae'])):.1f}", f"${md:+.2f}{stars(p)}$"]
                out[f'{lab}|vconly_N{vc}|{met}|base'] = float(np.nanmean(b1[f'{met}_mae'])); out[f'{lab}|vconly_N{vc}|{met}|neumann'] = float(np.nanmean(n1[f'{met}_mae']))
            L.append(f"{lab if first else ''} & {vc} & " + ' & '.join(cells) + r' \\'); first = False
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_sens_vconly_n_v2.tex')), 'w').write('\n'.join(L) + '\n')
    # ---------------- time base / window edges (base evaluation)
    L = [r'\begin{tabular}{llccccc}', r'\toprule',
         r'Cohort & Source & AT MAE & shift (mean signed) & AT MAE, shift removed & edge (start) & edge (end) \\']
    for sp, lab in COH:
        b = base[sp]
        if b is None:
            continue
        L.append(r'\midrule'); first = True
        for s in ('src1_vc', 'src2_allvalid', 'src4_recon_valid'):
            sub = b[(b.source == s) & (b.ref == 'commercial')]
            L.append(f"{lab if first else ''} & {SRC[s]} & {float(np.nanmean(sub.AT_mae)):.1f} & ${float(np.nanmean(sub.AT_bias)):+.1f}$ & {float(np.nanmean(sub.AT_mae_sc)):.1f} & "
                     f"{100 * float(np.nanmean(sub.edge_lo_frac)):.1f}\\% & {100 * float(np.nanmean(sub.edge_hi_frac)):.1f}\\% \\\\"); first = False
            out[f'{lab}|timebase|{s}'] = dict(mae=float(np.nanmean(sub.AT_mae)), bias=float(np.nanmean(sub.AT_bias)), mae_sc=float(np.nanmean(sub.AT_mae_sc)),
                                              edge_lo=float(np.nanmean(sub.edge_lo_frac)), edge_hi=float(np.nanmean(sub.edge_hi_frac)))
            nb = load('neumann', sp)
            if nb is not None and 'scale_ratio' in sub and 'scale_ratio' in nb:
                nsub = nb[(nb.source == s) & (nb.ref == 'commercial')]
                out[f'{lab}|boundary|{s}'] = dict(scale_base=float(np.nanmean(sub.scale_ratio)), scale_neumann=float(np.nanmean(nsub.scale_ratio)),
                                                  potcc_base=float(np.nanmean(sub.pot_cc)), potcc_neumann=float(np.nanmean(nsub.pot_cc)),
                                                  lam_base=float(np.nanmedian(sub.lam)), lam_neumann=float(np.nanmedian(nsub.lam)))
        sub = b[(b.source == 'src2_allvalid') & (b.ref == 'commercial')]
        L.append(f" & commercial reference & -- & -- & -- & {100 * float(np.nanmean(sub.ref_edge_lo_frac)):.1f}\\% & {100 * float(np.nanmean(sub.ref_edge_hi_frac)):.1f}\\% \\\\")
        out[f'{lab}|timebase|commercial'] = dict(edge_lo=float(np.nanmean(sub.ref_edge_lo_frac)), edge_hi=float(np.nanmean(sub.ref_edge_hi_frac)))
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_sens_timebase_v2.tex')), 'w').write('\n'.join(L) + '\n')
    # ---------------- filter: absolute marker shifts (needs the extended columns)
    L = [r'\begin{tabular}{llcccc}', r'\toprule', r'Cohort & Source & $\Delta$ mean AT (ms) & $\Delta$ mean RT (ms) & $\Delta$ AT MAE & $\Delta$ RT MAE \\']
    have = False
    for sp, lab in COH:
        b = base[sp]; h = load('hp05', sp)
        if b is None or h is None or 'RT_mean_src' not in h or 'RT_mean_src' not in b:
            continue
        have = True; L.append(r'\midrule'); first = True
        for s, name in (('src1_vc', 'VC-only inverse'), ('src2_allvalid', 'full-vest inverse'), ('src4_recon_valid', 'PosQ-MAE'), ('ref', 'commercial reference')):
            if s == 'ref':
                bs = b[(b.source == 'src2_allvalid') & (b.ref == 'commercial')].set_index('sid'); hs = h[(h.source == 'src2_allvalid') & (h.ref == 'commercial')].set_index('sid')
                j = bs[['AT_mean_ref', 'RT_mean_ref']].join(hs[['AT_mean_ref', 'RT_mean_ref']], lsuffix='_b', rsuffix='_h', how='inner')
                dat = (j.AT_mean_ref_h - j.AT_mean_ref_b).mean(); drt = (j.RT_mean_ref_h - j.RT_mean_ref_b).mean(); cells = [f'${dat:+.1f}$', f'${drt:+.1f}$', '--', '--']
            else:
                bs = b[(b.source == s) & (b.ref == 'commercial')].set_index('sid'); hs = h[(h.source == s) & (h.ref == 'commercial')].set_index('sid')
                j = bs[['AT_mean_src', 'RT_mean_src', 'AT_mae', 'RT_mae']].join(hs[['AT_mean_src', 'RT_mean_src', 'AT_mae', 'RT_mae']], lsuffix='_b', rsuffix='_h', how='inner')
                cells = [f'${(j.AT_mean_src_h - j.AT_mean_src_b).mean():+.1f}$', f'${(j.RT_mean_src_h - j.RT_mean_src_b).mean():+.1f}$',
                         f'${(j.AT_mae_h - j.AT_mae_b).mean():+.2f}$', f'${(j.RT_mae_h - j.RT_mae_b).mean():+.2f}$']
            L.append(f"{lab if first else ''} & {name} & " + ' & '.join(cells) + r' \\'); first = False
    L += [r'\bottomrule', r'\end{tabular}']
    if have:
        open(assert_v2(os.path.join(V2_GENERATED, 'tab_sens_filter_v2.tex')), 'w').write('\n'.join(L) + '\n')
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'review_sensitivity_v2.json')), 'w'), indent=1)
    for k, v in out.items():
        if not isinstance(v, dict):
            print(f'{k:45s} {v:7.2f}')
        else:
            print(f'{k:45s} ' + ' '.join(f'{a}={b:.3f}' for a, b in v.items()))
    print('wrote tab_sens_variants_v2.tex, tab_sens_vconly_n_v2.tex, tab_sens_timebase_v2.tex' + (', tab_sens_filter_v2.tex' if have else '') + ', review_sensitivity_v2.json')


if __name__ == '__main__':
    main()
