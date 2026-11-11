"""Summaries of the experiments requested by the user's review of 2026-10-25 (aggregates only):
  A. calibrated-lambda VC-only baseline per N (validation-set lambda*; percase_rerun/lamcal_*), beside the L-curve VC-only,
     the oracle lambda (N=16) and PosQ-MAE; patient-level superiority of PosQ-MAE over the calibrated baseline (N=16);
  B. clinical noise ladder with cross-completion (noisecross_white{20,10,5}): full vest, cross-completion, 16-electrode
     completion, VC-only vs the commercial reference; paired cross - full vest;
  C. simulated noise ladder vs ground truth (sim_gt/noise_{none,white20,white10,white5}_template.csv);
  D. shared time base from the 16 visible electrodes (rpeaks_vis) vs per-source detection (base) and all-channel (rpeaks_bsp);
  E. CARNZ (visible-set CAR + noise augmentation) vs LOC3 under base / re-referencing / noise variants and both layouts;
  F. VARN (variable-N training) vs the per-N models and under the all-visible / cross / no-substitution variants.
-> generated/review2_summary_v2.json and tab_review2_{lamcal,noisecross,simnoise,timebase,carnz,varn}_v2.tex
"""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('bsp_reconstruction', 'baselines_analyses', 'inverse_evaluation'):
    sys.path.insert(0, os.path.join(V2, d))
from paths import V2_GENERATED, assert_v2
from dedup import ok as dedup_ok
from patient_ids import patient_map
import results_v2 as R
RER = os.path.join(V2_GENERATED, 'percase_rerun'); CAND = os.path.join(V2_GENERATED, 'percase_candidate'); SIM = os.path.join(V2_GENERATED, 'sim_gt')
OURS = {4: 'P3N4', 8: 'P3N8', 16: 'LOC3', 32: 'P3N32', 64: 'P3N64'}
SPL = (('test_internal', 'internal'), ('test_external', 'external'))
OUT = {}


def load(path, source, ref='commercial', met='AT'):
    """sid -> value (success == 1, de-duplicated, numeric)."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path); d = d[(d.source == source) & (d.ref == ref)]
    if 'success' in d.columns:
        d = d[d.success == 1]
    d = d.loc[np.array([dedup_ok(s) for s in d.sid], dtype=bool)]
    if len(d) == 0:
        return None
    return pd.to_numeric(d.set_index('sid')[met + '_mae'], errors='coerce').dropna().to_dict()


def col(path, source, colname, ref='commercial'):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path); d = d[(d.source == source) & (d.ref == ref)]
    if 'success' in d.columns:
        d = d[d.success == 1]
    d = d.loc[np.array([dedup_ok(s) for s in d.sid], dtype=bool)]
    if len(d) == 0:
        return None
    return pd.to_numeric(d.set_index('sid')[colname], errors='coerce').dropna().to_dict()


def paired(a, b, split=None):
    """mean of a, mean of b over the common recordings, mean paired difference a-b, Wilcoxon P; with split also the
    patient-level mean difference with t-based 95% CI and one-sided superiority P (H0: mean >= 0)."""
    if a is None or b is None or not (set(a) & set(b)):
        return None
    k = sorted(set(a) & set(b)); x = np.array([a[s] for s in k]); y = np.array([b[s] for s in k]); d = x - y
    r = dict(n=len(k), a=float(x.mean()), b=float(y.mean()), d=float(d.mean()), d_med=float(np.median(d)),
             p=float(stats.wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else np.nan)
    if split:
        pm = patient_map(split); pats = np.array([str(pm.get(s, s.split('__')[1])) for s in k])
        up = sorted(set(pats)); dp = np.array([d[pats == p].mean() for p in up]); n = len(dp)
        se = dp.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
        r.update(n_pat=n, d_pat=float(dp.mean()), ci_lo=float(dp.mean() - tc * se), ci_hi=float(dp.mean() + tc * se),
                 p_sup=float(stats.t.cdf(dp.mean() / se, n - 1)), p_ni2=float(1 - stats.t.cdf((dp.mean() - 2.0) / se, n - 1)))
    return r


def st(p):
    return 'ns' if not np.isfinite(p) or p >= 0.05 else '*' if p >= 0.01 else '**' if p >= 0.001 else '***'


def sup(p):
    s = st(p); return r'\mathrm{ns}' if s == 'ns' else s


def cell(r, key='a', dkey='d', pkey='p', nd=1):
    if r is None:
        return '--'
    return '%.*f ($%+.2f^{%s}$)' % (nd, r[key], r[dkey], sup(r[pkey]))


# ---------------------------------------------------------------- A. calibrated lambda
def part_a():
    cal = json.load(open(os.path.join(V2_GENERATED, 'lambda_calibration_v2.json')))
    L = [r'\begin{tabular}{llcccccc}', r'\toprule', r' & & \multicolumn{3}{c}{internal} & \multicolumn{3}{c}{external} \\', r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
         r'$N$ & Source & AT MAE & RT MAE & scale ratio & AT MAE & RT MAE & scale ratio \\', r'\midrule']
    for vc in (4, 8, 16, 32, 64):
        rows = {}
        for split, lab in SPL:
            cand = os.path.join(CAND, f'{OURS[vc]}_vc{vc}_{split}_template.csv'); lc = os.path.join(RER, f'lamcal_vc{vc}_{split}_template.csv')
            base16 = os.path.join(RER, f'base_vc16_{split}_template.csv'); orc = os.path.join(RER, f'oracle_vc16_{split}_template.csv')
            src = {'lcurve': (cand, 'src1_vc'), 'cal': (lc, 'src1_vc'), 'posq': (cand, 'src4_recon_valid'), 'vest': (base16, 'src2_allvalid')}
            if vc == 16:
                src['oracle'] = (orc, 'src1_vc')
            for k, (pth, so) in src.items():
                rows[(lab, k)] = dict(AT=load(pth, so, met='AT'), RT=load(pth, so, met='RT'), sc=col(pth, so, 'scale_ratio'))
            # paired: PosQ vs calibrated VC-only (patient level), calibrated vs L-curve
            OUT[f'lamcal|{vc}|{lab}|posq_vs_cal|AT'] = paired(rows[(lab, 'posq')]['AT'], rows[(lab, 'cal')]['AT'], split)
            OUT[f'lamcal|{vc}|{lab}|posq_vs_cal|RT'] = paired(rows[(lab, 'posq')]['RT'], rows[(lab, 'cal')]['RT'], split)
            OUT[f'lamcal|{vc}|{lab}|cal_vs_lcurve|AT'] = paired(rows[(lab, 'cal')]['AT'], rows[(lab, 'lcurve')]['AT'])
            OUT[f'lamcal|{vc}|{lab}|cal_vs_lcurve|RT'] = paired(rows[(lab, 'cal')]['RT'], rows[(lab, 'lcurve')]['RT'])
            for k in src:
                OUT[f'lamcal|{vc}|{lab}|{k}'] = {m: (float(np.mean(list(rows[(lab, k)][m].values()))) if rows[(lab, k)][m] else None) for m in ('AT', 'RT', 'sc')}
        OUT[f'lamcal|{vc}|lambda'] = cal[str(vc)]['lam']
        names = [('lcurve', 'VC-only, L-curve $\\lambda$'), ('cal', 'VC-only, calibrated $\\lambda{=}%g$' % cal[str(vc)]['lam'])] + ([('oracle', 'VC-only, oracle $\\lambda$')] if vc == 16 else []) + [('posq', 'PosQ-MAE')]
        for i, (k, nm) in enumerate(names):
            c = []
            for split, lab in SPL:
                v = OUT[f'lamcal|{vc}|{lab}|{k}']
                c += ['%.1f' % v['AT'] if v['AT'] is not None else '--', '%.1f' % v['RT'] if v['RT'] is not None else '--', '%.2f' % v['sc'] if v['sc'] is not None else '--']
            L.append('%s & %s & %s \\\\' % (str(vc) if i == 0 else '', nm, ' & '.join(c)))
        L.append(r'\midrule')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_review2_lamcal_v2.tex')), 'w').write('\n'.join(L) + '\n')
    print('\n== A. calibrated lambda (validation set) ==')
    for vc in (4, 8, 16, 32, 64):
        print(f'N={vc:2d} lambda*={OUT[f"lamcal|{vc}|lambda"]:g}: ' + ' | '.join(
            f'{lab} L-curve {OUT[f"lamcal|{vc}|{lab}|lcurve"]["AT"]:.1f}/{OUT[f"lamcal|{vc}|{lab}|lcurve"]["RT"]:.1f} cal {OUT[f"lamcal|{vc}|{lab}|cal"]["AT"]:.1f}/{OUT[f"lamcal|{vc}|{lab}|cal"]["RT"]:.1f} (sc {OUT[f"lamcal|{vc}|{lab}|cal"]["sc"]:.2f}) PosQ {OUT[f"lamcal|{vc}|{lab}|posq"]["AT"]:.1f}/{OUT[f"lamcal|{vc}|{lab}|posq"]["RT"]:.1f}' for _, lab in SPL))
        for _, lab in SPL:
            r = OUT[f'lamcal|{vc}|{lab}|posq_vs_cal|AT']; r2 = OUT[f'lamcal|{vc}|{lab}|posq_vs_cal|RT']
            print(f'      {lab}: PosQ - cal AT {r["d"]:+.2f} (pat {r["d_pat"]:+.2f} [{r["ci_lo"]:+.2f},{r["ci_hi"]:+.2f}] sup P={r["p_sup"]:.4f}) | RT {r2["d"]:+.2f} (pat {r2["d_pat"]:+.2f}, sup P={r2["p_sup"]:.4f})')


# ---------------------------------------------------------------- B. clinical noise + cross
def part_b():
    L = [r'\begin{tabular}{llcccccccc}', r'\toprule', r' & & \multicolumn{4}{c}{AT MAE (ms)} & \multicolumn{4}{c}{RT MAE (ms)} \\', r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',
         r'Cohort & Added noise & full vest & cross-completion & PosQ-MAE $16$ & VC-only $16$ & full vest & cross-completion & PosQ-MAE $16$ & VC-only $16$ \\', r'\midrule']
    print('\n== B. clinical noise ladder with cross-completion ==')
    for split, lab in SPL:
        for i, (tag, nm) in enumerate((('base', 'none'), ('noisecross_white20', 'white, 20 dB'), ('noisecross_white10', 'white, 10 dB'), ('noisecross_white5', 'white, 5 dB'))):
            pth = os.path.join(RER, f'{tag}_vc16_{split}_template.csv'); mdn_base = os.path.join(RER, f'mdn_vc16_{split}_template.csv')
            S = {}
            for k, so in (('vest', 'src2_allvalid'), ('cross', 'src6_mdn'), ('posq', 'src4_recon_valid'), ('vc', 'src1_vc')):
                p2 = mdn_base if (tag == 'base' and k == 'cross') else pth
                S[k] = {m: load(p2, so, met=m) for m in ('AT', 'RT')}
            c = []
            for m in ('AT', 'RT'):
                pv = paired(S['vest'][m], S['vest'][m]); pc = paired(S['cross'][m], S['vest'][m]); pq = paired(S['posq'][m], S['vest'][m]); pvc = paired(S['vc'][m], S['vest'][m])
                OUT[f'noisecross|{lab}|{nm}|{m}'] = dict(vest=pv['a'] if pv else None, cross=pc, posq=pq, vc=pvc)
                c += ['%.1f' % pv['a'], cell(pc), cell(pq), cell(pvc)]
            L.append('%s & %s & %s \\\\' % (lab if i == 0 else '', nm, ' & '.join(c)))
            print(f'{lab:8s} {nm:12s}: ' + ' | '.join(f'{m}: vest {OUT[f"noisecross|{lab}|{nm}|{m}"]["vest"]:.1f} cross {OUT[f"noisecross|{lab}|{nm}|{m}"]["cross"]["a"]:.1f} ({OUT[f"noisecross|{lab}|{nm}|{m}"]["cross"]["d"]:+.2f} {st(OUT[f"noisecross|{lab}|{nm}|{m}"]["cross"]["p"])}) posq {OUT[f"noisecross|{lab}|{nm}|{m}"]["posq"]["a"]:.1f} ({OUT[f"noisecross|{lab}|{nm}|{m}"]["posq"]["d"]:+.2f}) vc {OUT[f"noisecross|{lab}|{nm}|{m}"]["vc"]["a"]:.1f}' for m in ('AT', 'RT')))
        L.append(r'\midrule')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_review2_noisecross_v2.tex')), 'w').write('\n'.join(L) + '\n')


# ---------------------------------------------------------------- C. simulated noise ladder (ground truth)
def part_c():
    print('\n== C. simulated noise ladder vs ground truth ==')
    L = [r'\begin{tabular}{lcccccccc}', r'\toprule', r' & \multicolumn{4}{c}{AT MAE vs.\ ground truth (ms)} & \multicolumn{4}{c}{RT MAE vs.\ ground truth (ms)} \\', r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
         r'Added noise & full vest & cross-completion & PosQ-MAE $16$ & VC-only $16$ & full vest & cross-completion & PosQ-MAE $16$ & VC-only $16$ \\', r'\midrule']
    for tag, nm in (('none', 'none'), ('white20', 'white, 20 dB'), ('white10', 'white, 10 dB'), ('white5', 'white, 5 dB')):
        pth = os.path.join(SIM, f'noise_{tag}_template.csv')
        if not os.path.exists(pth):
            print('  missing', pth); continue
        d = pd.read_csv(pth); d = d[d.success == 1]
        S = {k: {m: pd.to_numeric(d[d.source == so].set_index('sid')[m + '_mae'], errors='coerce').dropna().to_dict() for m in ('AT', 'RT')}
             for k, so in (('vest', 'src2_allvalid'), ('cross', 'src6_mdn'), ('posq', 'src4_recon_valid'), ('vc', 'src1_vc'))}
        c = []
        for m in ('AT', 'RT'):
            pv = paired(S['vest'][m], S['vest'][m]); pc = paired(S['cross'][m], S['vest'][m]); pq = paired(S['posq'][m], S['vest'][m]); pvc = paired(S['vc'][m], S['vest'][m])
            OUT[f'simnoise|{nm}|{m}'] = dict(n=pv['n'], vest=pv['a'], cross=pc, posq=pq, vc=pvc)
            c += ['%.1f' % pv['a'], cell(pc), cell(pq), cell(pvc)]
        bsp = d[d.source == 'src4_recon_valid']
        OUT[f'simnoise|{nm}|bsp_cc_clean'] = float(pd.to_numeric(bsp.bsp_cc_clean, errors='coerce').mean()); OUT[f'simnoise|{nm}|bsp_cc'] = float(pd.to_numeric(bsp.bsp_cc, errors='coerce').mean())
        L.append('%s & %s \\\\' % (nm, ' & '.join(c)))
        print(f'{nm:12s} (n={OUT[f"simnoise|{nm}|AT"]["n"]}): ' + ' | '.join(f'{m}: vest {OUT[f"simnoise|{nm}|{m}"]["vest"]:.1f} cross {OUT[f"simnoise|{nm}|{m}"]["cross"]["a"]:.1f} ({OUT[f"simnoise|{nm}|{m}"]["cross"]["d"]:+.2f} {st(OUT[f"simnoise|{nm}|{m}"]["cross"]["p"])}) posq {OUT[f"simnoise|{nm}|{m}"]["posq"]["a"]:.1f} ({OUT[f"simnoise|{nm}|{m}"]["posq"]["d"]:+.2f} {st(OUT[f"simnoise|{nm}|{m}"]["posq"]["p"])}) vc {OUT[f"simnoise|{nm}|{m}"]["vc"]["a"]:.1f}' for m in ('AT', 'RT')) + f' | bspCC clean {OUT[f"simnoise|{nm}|bsp_cc_clean"]:.3f} noisy {OUT[f"simnoise|{nm}|bsp_cc"]:.3f}')
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_review2_simnoise_v2.tex')), 'w').write('\n'.join(L) + '\n')


# ---------------------------------------------------------------- D. time base
def part_d():
    print('\n== D. shared time base from the 16 visible electrodes ==')
    L = [r'\begin{tabular}{llcccccc}', r'\toprule', r' & & \multicolumn{3}{c}{AT MAE (ms)} & \multicolumn{3}{c}{RT MAE (ms)} \\', r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
         r'Cohort & Beat detection & VC-only & full vest & PosQ-MAE & VC-only & full vest & PosQ-MAE \\', r'\midrule']
    for split, lab in SPL:
        base = os.path.join(RER, f'base_vc16_{split}_template.csv')
        for i, (tag, nm) in enumerate((('base', 'per source (paper)'), ('rpeaks_vis', 'shared, $16$ visible electrodes'), ('rpeaks_bsp', 'shared, all active channels'))):
            pth = os.path.join(RER, f'{tag}_vc16_{split}_template.csv'); c = []
            for m in ('AT', 'RT'):
                for so, k in (('src1_vc', 'vc'), ('src2_allvalid', 'vest'), ('src4_recon_valid', 'posq')):
                    r = paired(load(pth, so, met=m), load(base, so, met=m)); OUT[f'timebase|{lab}|{tag}|{k}|{m}'] = r
                    c.append('%.1f' % r['a'] if tag == 'base' else cell(r))
            L.append('%s & %s & %s \\\\' % (lab if i == 0 else '', nm, ' & '.join(c)))
            print(f'{lab:8s} {nm:32s}: ' + ' | '.join(f'{m}: vc {OUT[f"timebase|{lab}|{tag}|vc|{m}"]["a"]:.1f} ({OUT[f"timebase|{lab}|{tag}|vc|{m}"]["d"]:+.2f}) vest {OUT[f"timebase|{lab}|{tag}|vest|{m}"]["a"]:.1f} ({OUT[f"timebase|{lab}|{tag}|vest|{m}"]["d"]:+.2f}) posq {OUT[f"timebase|{lab}|{tag}|posq|{m}"]["a"]:.1f} ({OUT[f"timebase|{lab}|{tag}|posq|{m}"]["d"]:+.2f})' for m in ('AT', 'RT')))
        L.append(r'\midrule')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_review2_timebase_v2.tex')), 'w').write('\n'.join(L) + '\n')


# ---------------------------------------------------------------- E. CARNZ
def part_e():
    print('\n== E. CARNZ (visible-set CAR + noise augmentation) vs LOC3 ==')
    L = [r'\begin{tabular}{llcccccc}', r'\toprule', r' & & \multicolumn{3}{c}{AT MAE (ms)} & \multicolumn{3}{c}{RT MAE (ms)} \\', r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
         r'Cohort & Condition & full vest & PosQ-MAE & PosQ-MAE-CAR+noise & full vest & PosQ-MAE & PosQ-MAE-CAR+noise \\', r'\midrule']
    conds = (('base', 'none (template layout)', 'base'), ('reref_car', 'common-average reference', 'reref_car'), ('reref_ch0', 'single-electrode reference', 'reref_ch0'),
             ('noise_emg10', 'EMG-band noise, $10$ dB', 'noise_emg10'), ('noise_white10', 'white noise, $10$ dB', 'noise_white10'), ('noise_white5', 'white noise, $5$ dB', 'noise_white5'))
    for split, lab in SPL:
        for i, (tag_l, nm, tag_c) in enumerate(conds):
            pl = os.path.join(RER, f'{tag_l}_vc16_{split}_template.csv'); pc = os.path.join(RER, f'carnz_{tag_c}_vc16_{split}_template.csv'); c = []
            for m in ('AT', 'RT'):
                # 2026-11-11 audit fix: re-referencing acts on the 16 inputs only, so the full-vest column of the re-referencing
                # rows is the unperturbed full-vest inverse (the earlier fallback read a field whose visible rows were re-referenced)
                pv = os.path.join(RER, f'base_vc16_{split}_template.csv') if tag_l.startswith('reref') else pl
                vest = load(pv, 'src2_allvalid', met=m) or load(pc, 'src2_allvalid', met=m); loc = load(pl, 'src4_recon_valid', met=m); car = load(pc, 'src4_recon_valid', met=m)
                rv = paired(car, vest, split); rl = paired(car, loc, split); OUT[f'carnz|{lab}|{tag_l}|{m}'] = dict(vest=float(np.mean(list(vest.values()))), loc=float(np.mean(list(loc.values()))), car_vs_vest=rv, car_vs_loc=rl)
                c += ['%.1f' % OUT[f'carnz|{lab}|{tag_l}|{m}']['vest'], '%.1f' % OUT[f'carnz|{lab}|{tag_l}|{m}']['loc'], cell(rl)]
            L.append('%s & %s & %s \\\\' % (lab if i == 0 else '', nm, ' & '.join(c)))
            print(f'{lab:8s} {nm:28s}: ' + ' | '.join(f'{m}: vest {OUT[f"carnz|{lab}|{tag_l}|{m}"]["vest"]:.1f} LOC3 {OUT[f"carnz|{lab}|{tag_l}|{m}"]["loc"]:.1f} CARNZ {OUT[f"carnz|{lab}|{tag_l}|{m}"]["car_vs_loc"]["a"]:.1f} (vs LOC3 {OUT[f"carnz|{lab}|{tag_l}|{m}"]["car_vs_loc"]["d"]:+.2f} {st(OUT[f"carnz|{lab}|{tag_l}|{m}"]["car_vs_loc"]["p"])})' for m in ('AT', 'RT')))
        # k-means layout from eval_downstream
        for lay in ('template', 'kmeans'):
            for m in ('AT', 'RT'):
                a = load(os.path.join(CAND, f'CARNZ_vc16_{split}_{lay}.csv'), 'src4_recon_valid', met=m); b = load(os.path.join(CAND, f'LOC3_vc16_{split}_{lay}.csv'), 'src4_recon_valid', met=m)
                OUT[f'carnz_eval|{lab}|{lay}|{m}'] = paired(a, b, split)
            r = OUT[f'carnz_eval|{lab}|{lay}|AT']; r2 = OUT[f'carnz_eval|{lab}|{lay}|RT']
            print(f'   eval_downstream {lab} {lay}: AT CARNZ {r["a"]:.2f} vs LOC3 {r["b"]:.2f} ({r["d"]:+.2f}, P={r["p"]:.3g}) | RT {r2["a"]:.2f} vs {r2["b"]:.2f} ({r2["d"]:+.2f}, P={r2["p"]:.3g})')
        L.append(r'\midrule')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_review2_carnz_v2.tex')), 'w').write('\n'.join(L) + '\n')


# ---------------------------------------------------------------- F. VARN
def part_f():
    print('\n== F. VARN (variable-N training, 4-64) vs the per-N models ==')
    L = [r'\begin{tabular}{llcccccc}', r'\toprule', r' & & \multicolumn{3}{c}{AT MAE (ms)} & \multicolumn{3}{c}{RT MAE (ms)} \\', r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
         r'Cohort & Condition & per-$N$ model & variable-$N$ model & $\Delta$ & per-$N$ model & variable-$N$ model & $\Delta$ \\', r'\midrule']
    for split, lab in SPL:
        for i, vc in enumerate((4, 8, 16, 32, 64)):
            c = []
            for m in ('AT', 'RT'):
                a = load(os.path.join(CAND, f'VARN_vc{vc}_{split}_template.csv'), 'src4_recon_valid', met=m); b = load(os.path.join(CAND, f'{OURS[vc]}_vc{vc}_{split}_template.csv'), 'src4_recon_valid', met=m)
                r = paired(a, b, split); OUT[f'varn|{lab}|N{vc}|{m}'] = r; c += ['%.1f' % r['b'], '%.1f' % r['a'], '$%+.2f^{%s}$' % (r['d'], sup(r['p']))]
            L.append('%s & $N{=}%d$, template & %s \\\\' % (lab if i == 0 else '', vc, ' & '.join(c)))
            print(f'{lab:8s} N={vc:2d}: AT per-N {OUT[f"varn|{lab}|N{vc}|AT"]["b"]:.2f} VARN {OUT[f"varn|{lab}|N{vc}|AT"]["a"]:.2f} ({OUT[f"varn|{lab}|N{vc}|AT"]["d"]:+.2f} {st(OUT[f"varn|{lab}|N{vc}|AT"]["p"])}) | RT {OUT[f"varn|{lab}|N{vc}|RT"]["b"]:.2f} VARN {OUT[f"varn|{lab}|N{vc}|RT"]["a"]:.2f} ({OUT[f"varn|{lab}|N{vc}|RT"]["d"]:+.2f} {st(OUT[f"varn|{lab}|N{vc}|RT"]["p"])})')
        for m in ('AT', 'RT'):
            a = load(os.path.join(CAND, f'VARN_vc16_{split}_kmeans.csv'), 'src4_recon_valid', met=m); b = load(os.path.join(CAND, f'LOC3_vc16_{split}_kmeans.csv'), 'src4_recon_valid', met=m)
            OUT[f'varn|{lab}|N16km|{m}'] = paired(a, b, split)
        c = []
        for m in ('AT', 'RT'):
            r = OUT[f'varn|{lab}|N16km|{m}']; c += ['%.1f' % r['b'], '%.1f' % r['a'], '$%+.2f^{%s}$' % (r['d'], sup(r['p']))]
        L.append(' & $N{=}16$, $k$-means & %s \\\\' % ' & '.join(c))
        print(f'{lab:8s} N=16 k-means: AT {OUT[f"varn|{lab}|N16km|AT"]["b"]:.2f} -> {OUT[f"varn|{lab}|N16km|AT"]["a"]:.2f} ({OUT[f"varn|{lab}|N16km|AT"]["d"]:+.2f}) | RT {OUT[f"varn|{lab}|N16km|RT"]["b"]:.2f} -> {OUT[f"varn|{lab}|N16km|RT"]["a"]:.2f} ({OUT[f"varn|{lab}|N16km|RT"]["d"]:+.2f})')
        for tag_v, tag_l, nm, so in (('varn_visall', 'visall', 'all valid electrodes visible (missing only)', 'src3_recon_whole'), ('varn_mdn', 'mdn', 'cross-completion of two halves', 'src6_mdn'), ('varn_nosub', 'nosub', 'no electrode substitution', 'src4_recon_valid')):
            c = []
            for m in ('AT', 'RT'):
                a = load(os.path.join(RER, f'{tag_v}_vc16_{split}_template.csv'), so, met=m); b = load(os.path.join(RER, f'{tag_l}_vc16_{split}_template.csv'), so, met=m)
                r = paired(a, b, split); OUT[f'varn|{lab}|{tag_l}|{m}'] = r; c += ['%.1f' % r['b'], '%.1f' % r['a'], '$%+.2f^{%s}$' % (r['d'], sup(r['p']))]
            L.append(' & %s & %s \\\\' % (nm, ' & '.join(c)))
            print(f'{lab:8s} {nm:44s}: AT {OUT[f"varn|{lab}|{tag_l}|AT"]["b"]:.2f} -> {OUT[f"varn|{lab}|{tag_l}|AT"]["a"]:.2f} ({OUT[f"varn|{lab}|{tag_l}|AT"]["d"]:+.2f} {st(OUT[f"varn|{lab}|{tag_l}|AT"]["p"])}) | RT {OUT[f"varn|{lab}|{tag_l}|RT"]["b"]:.2f} -> {OUT[f"varn|{lab}|{tag_l}|RT"]["a"]:.2f} ({OUT[f"varn|{lab}|{tag_l}|RT"]["d"]:+.2f} {st(OUT[f"varn|{lab}|{tag_l}|RT"]["p"])})')
        L.append(r'\midrule')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_review2_varn_v2.tex')), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    for fn in (part_a, part_b, part_c, part_d, part_e, part_f):
        try:
            fn()
        except Exception as e:
            import traceback; print(f'[{fn.__name__}] FAILED: {type(e).__name__}: {e}'); traceback.print_exc(limit=2)
    json.dump(OUT, open(assert_v2(os.path.join(V2_GENERATED, 'review2_summary_v2.json')), 'w'), indent=1, default=float)
    print('\nwrote review2_summary_v2.json and tab_review2_*_v2.tex')
