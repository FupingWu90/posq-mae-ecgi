"""Formal non-inferiority / superiority statistics for the N = 16 comparisons and GRRAS-style agreement statistics
with confidence intervals (reviewer points 5a, 5b, 5d; 2026-10-23).

Part A (per-recording errors against the commercial reference, fixed template layout, N = 16):
  for every alternative and both test cohorts, the paired difference d = E_Ours - E_alt per recording is averaged
  per patient and summarised at the patient level (mean, SD, t-based 95% CI, two-sided t test), with
    * non-inferiority of the completion to the full-vest (all-valid) inverse: one-sided test of H0: mean d >= +delta
      (delta = 2 ms for AT, designated as primary; non-inferiority is declared when the upper bound of the two-sided 95% CI lies
      below delta, i.e. one-sided P < 0.025; the one-sided P is reported);
    * superiority over every other alternative: one-sided test of H0: mean d >= 0;
    * a linear mixed model with a patient random intercept on the recording-level differences (Wald CI; normal
      approximation; Kenward-Roger corrections were not applied);
    * a patient-cluster bootstrap CI of the recording-weighted mean (2,000 resamples);
    * the recording-level median and Wilcoxon p of the main text, for cross-reference.
Part B (clinical endpoints, generated/clinical_endpoints/<split>_template.csv): bias, 95% limits of agreement and
  ICC(2,1) with patient-cluster bootstrap 95% CIs for completion - full vest, raw 16-electrode inverse - full vest and
  the floor (full vest - commercial), five endpoints, three cohorts.
Outputs: generated/noninferiority_v2.json, tab_noninferiority_v2.tex, tab_agreement_ci_v2.tex,
         figures/fig_forest_n16_v2.png.  Aggregates only are printed.
"""
import os, sys, json
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
for p in ('baselines_analyses', 'baselines_analyses', 'bsp_reconstruction', 'inverse_evaluation'):
    sys.path.insert(0, os.path.join(V2, p))
from paths import V2_GENERATED, FIG_DIR_V2, assert_v2
import results_v2 as R
from patient_ids import patient_map
from patient_level_stats import paired_stats
from clinical_endpoints import icc21
from dedup import ok as dedup_ok

ALTS = [('allvalid', 'full-vest inverse'), ('vconly', 'VC-only inverse'), ('vconly_cal', 'VC-only inverse, calibrated $\\lambda$'), ('laplacian', 'Laplace interpolation'),
        ('tps', 'TPS interpolation'), ('lmmse', 'LMMSE'), ('stmem', 'ST-MEM adapted to the vest'), ('mcma', 'MCMA')]
DELTA = {'AT': 2.0}                # non-inferiority margin (ms) vs the full-vest inverse, activation time only; repolarisation is reported as estimates (user review 2026-10-25: the 4-ms RT margin had no basis)
B = 2000


def patient_level(d, pats):
    """d: per-recording paired differences; pats: patient id per recording -> dict of patient-level statistics."""
    pats = np.asarray(pats); up = sorted(set(pats))
    dp = np.array([d[pats == p].mean() for p in up]); n = len(up)
    m, sd = float(dp.mean()), float(dp.std(ddof=1)); se = sd / np.sqrt(n); tcrit = stats.t.ppf(0.975, n - 1)
    out = dict(n_pat=n, mean_pat=m, sd_pat=sd, se_pat=float(se), ci_lo=float(m - tcrit * se), ci_hi=float(m + tcrit * se),
               p_two_sided=float(stats.ttest_1samp(dp, 0.0).pvalue), ub95_one_sided=float(m + stats.t.ppf(0.95, n - 1) * se),
               p_superiority=float(stats.t.cdf(m / se, n - 1)))          # H0: mean >= 0 (one-sided)
    return out, dp


def ni_p(m, se, n, delta):
    """one-sided p for H0: mean >= delta (non-inferiority within delta)."""
    return float(stats.t.cdf((m - delta) / se, n - 1))


def cluster_boot(d, pats, seed=0):
    pats = np.asarray(pats); up = sorted(set(pats)); groups = [d[pats == p] for p in up]
    psum = np.array([g.sum() for g in groups]); pcnt = np.array([len(g) for g in groups], float)
    idx = np.random.default_rng(seed).integers(0, len(up), size=(B, len(up)))
    boots = psum[idx].sum(axis=1) / pcnt[idx].sum(axis=1)
    return float(psum.sum() / pcnt.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mixed(d, pats):
    import statsmodels.api as sm
    import pandas as pd
    df = pd.DataFrame({'d': d, 'pat': pats})
    try:
        mdl = sm.MixedLM(df['d'], np.ones((len(df), 1)), groups=df['pat']).fit(reml=True)
        est = float(mdl.params.iloc[0]); se = float(mdl.bse.iloc[0])
        return dict(est=est, se=se, ci_lo=est - 1.96 * se, ci_hi=est + 1.96 * se, p=float(mdl.pvalues.iloc[0]),
                    var_pat=float(mdl.cov_re.iloc[0, 0]), var_res=float(mdl.scale))
    except Exception as e:   # singular fits
        return dict(est=float('nan'), se=float('nan'), ci_lo=float('nan'), ci_hi=float('nan'), p=float('nan'), error=str(e)[:60])


def part_a():
    R.LAYOUT, R.REF = 'template', 'commercial'
    P, _ = R.load_everything()
    res = {}
    for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
        pmap = patient_map(split); ours = P[('ours', 16, split)]
        av = {met: R.load_allvalid(split, met) for met in ('AT', 'RT')}
        for met in ('AT', 'RT'):
            for key, label in ALTS:
                alt = av[met] if key == 'allvalid' else {s: v[met] for s, v in P.get((key, 16, split), {}).items() if met in v}
                sids = [s for s in sorted(set(ours) & set(alt)) if s in pmap and met in ours[s] and np.isfinite(alt[s])]
                d = np.array([ours[s][met] - alt[s] for s in sids]); pats = [pmap[s] for s in sids]
                pl, dp = patient_level(d, pats)
                r = dict(label=label, n_rec=len(d), mean_rec=float(d.mean()), median_rec=float(np.median(d)),
                         p_wilcoxon_rec=float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, **pl)
                r['p_noninferiority'] = ni_p(pl['mean_pat'], pl['se_pat'], pl['n_pat'], DELTA[met]) if (key == 'allvalid' and met in DELTA) else None
                r['delta'] = DELTA[met] if (key == 'allvalid' and met in DELTA) else None
                r['boot_mean'], r['boot_lo'], r['boot_hi'] = cluster_boot(d, pats)
                r['mixed'] = mixed(d, pats)
                res[f'{lab}|{met}|{key}'] = r
                print(f'{lab:8s} {met} vs {key:9s}: n_rec {len(d):3d} n_pat {pl["n_pat"]:3d} patient mean {pl["mean_pat"]:+6.2f} '
                      f'[{pl["ci_lo"]:+6.2f}, {pl["ci_hi"]:+6.2f}] p2s {pl["p_two_sided"]:.3f} | UB95 {pl["ub95_one_sided"]:+6.2f} '
                      f'| mixed {r["mixed"]["est"]:+6.2f} [{r["mixed"]["ci_lo"]:+6.2f}, {r["mixed"]["ci_hi"]:+6.2f}] '
                      f'| boot [{r["boot_lo"]:+6.2f}, {r["boot_hi"]:+6.2f}]'
                      + (f' | NI(delta={DELTA[met]}) p={r["p_noninferiority"]:.4f}' if (key == 'allvalid' and met in DELTA) else f' | sup p={pl["p_superiority"]:.4f}'))
    return res


def part_nosub():
    """Non-inferiority test of the completion WITHOUT nearest-neighbour substitution of failed template electrodes (the
    physically sparse array; percase_rerun/nosub_vc16_<split>_template.csv, sources src4_recon_valid vs src2_allvalid,
    commercial reference) -- user review 2026-10-25.  -> generated/noninferiority_nosub_v2.json / tab_noninferiority_nosub_v2.tex"""
    import pandas as pd
    out = {}; L = [r'\begin{tabular}{llrrrr}', r'\toprule', r'Cohort & Endpoint (PosQ-MAE vs.\ full vest) & patient-level mean [95\% CI] & one-sided $P$ & mixed model [95\% CI] & rec.\ median ($P$) \\', r'\midrule']
    for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
        pmap = patient_map(split)
        df = pd.read_csv(os.path.join(V2_GENERATED, 'percase_rerun', f'nosub_vc16_{split}_template.csv'))
        df = df[(df.ref == 'commercial') & (df.success == 1)]; df = df[[dedup_ok(s_) for s_ in df.sid]]
        for met in ('AT', 'RT'):
            o = pd.to_numeric(df[df.source == 'src4_recon_valid'].set_index('sid')[met + '_mae'], errors='coerce')
            v = pd.Series(R.load_allvalid(split, met), dtype=float)     # full-vest inverse vs commercial (reference cache, as in part_a)
            j = o.to_frame('o').join(v.to_frame('v'), how='inner').dropna(); j = j[[s_ in pmap for s_ in j.index]]
            assert len(j) > 0, (split, met)
            d = (j['o'] - j['v']).values; pats = [pmap[s_] for s_ in j.index]
            pl, dp = patient_level(d, pats)
            r = dict(n_rec=int(len(d)), mean_rec=float(d.mean()), median_rec=float(np.median(d)), p_wilcoxon_rec=float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                     mean_ours=float(j['o'].mean()), mean_vest=float(j['v'].mean()), **pl)
            r['delta'] = DELTA.get(met); r['p_noninferiority'] = ni_p(pl['mean_pat'], pl['se_pat'], pl['n_pat'], DELTA[met]) if met in DELTA else None
            r['mixed'] = mixed(d, pats); out[f'{lab}|{met}'] = r
            pc = (f"NI ($\\delta{{=}}{r['delta']:g}$): ${r['p_noninferiority']:.3f}$" if r['p_noninferiority'] >= 0.001 else f"NI ($\\delta{{=}}{r['delta']:g}$): $<0.001$") if met in DELTA else 'estimate only'
            m = r['mixed']; mmv = f"${m['est']:+.1f}$ [${m['ci_lo']:+.1f}$, ${m['ci_hi']:+.1f}$]" if np.isfinite(m['est']) else '--'
            L.append(f"{lab if met == 'AT' else ''} & {met} ({r['mean_ours']:.1f} vs.\\ {r['mean_vest']:.1f}\\,ms) & ${pl['mean_pat']:+.1f}$ [${pl['ci_lo']:+.1f}$, ${pl['ci_hi']:+.1f}$] & {pc} & {mmv} & ${r['median_rec']:+.1f}${R.stars(r['p_wilcoxon_rec'])} \\\\")
            print(f'nosub {lab} {met}: ours {r["mean_ours"]:.2f} vest {r["mean_vest"]:.2f} | patient mean {pl["mean_pat"]:+.2f} [{pl["ci_lo"]:+.2f}, {pl["ci_hi"]:+.2f}] n_pat {pl["n_pat"]} ' + (f'NI p={r["p_noninferiority"]:.4f}' if met in DELTA else '') + f' | rec median {r["median_rec"]:+.2f} p={r["p_wilcoxon_rec"]:.3g}')
        L.append(r'\midrule')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_noninferiority_nosub_v2.tex')), 'w').write('\n'.join(L) + '\n')
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'noninferiority_nosub_v2.json')), 'w'), indent=1, default=float)
    print('wrote tab_noninferiority_nosub_v2.tex, noninferiority_nosub_v2.json')
    return out


def forest(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    dpi = int(os.environ.get('ECGI_FIG_DPI', '150'))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey='row')
    for i, met in enumerate(('AT', 'RT')):
        for j, lab in enumerate(('internal', 'external')):
            ax = axes[i, j]; ys = np.arange(len(ALTS))[::-1]
            for y, (key, label) in zip(ys, ALTS):
                r = res[f'{lab}|{met}|{key}']
                ax.errorbar(r['mean_pat'], y, xerr=[[r['mean_pat'] - r['ci_lo']], [r['ci_hi'] - r['mean_pat']]], fmt='o', color='#1f2937',
                            ecolor='#1f2937', elinewidth=1.4, capsize=3, markersize=5)
            ax.axvline(0, color='#6b7280', lw=1, ls='--')
            if met in DELTA:
                ax.axvspan(-DELTA[met], DELTA[met], color='#9ca3af', alpha=0.18, lw=0)
            ax.set_yticks(ys); ax.set_yticklabels([l for _, l in ALTS] if j == 0 else [])
            ax.set_title(f'{"(a)" if (i, j) == (0, 0) else "(b)" if (i, j) == (0, 1) else "(c)" if (i, j) == (1, 0) else "(d)"} {met}, {lab} cohort', loc='left', fontsize=10)
            ax.set_xlabel('PosQ-MAE $-$ alternative, patient-level mean paired difference (ms)' if i == 1 else '')
            ax.grid(axis='x', color='#e5e7eb', lw=0.8); ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR_V2, 'fig_forest_n16_v2.png'); fig.savefig(out, dpi=dpi); plt.close(fig); print('wrote', out)


def table_a(res):
    def ci(r):
        return f"${r['mean_pat']:+.1f}$ [${r['ci_lo']:+.1f}$, ${r['ci_hi']:+.1f}$]"
    def mm(r):
        m = r['mixed']; return f"${m['est']:+.1f}$ [${m['ci_lo']:+.1f}$, ${m['ci_hi']:+.1f}$]" if np.isfinite(m['est']) else '--'
    def pcol(r, key):
        if key == 'allvalid':
            if r['delta'] is None:
                return 'estimate only'
            return f"NI ($\\delta{{=}}{r['delta']:g}$): ${r['p_noninferiority']:.3f}$" if r['p_noninferiority'] >= 0.001 else f"NI ($\\delta{{=}}{r['delta']:g}$): $<0.001$"
        return f"sup.: ${r['p_superiority']:.3f}$" if r['p_superiority'] >= 0.001 else 'sup.: $<0.001$'
    L = [r'\begin{tabular}{llrrrr}', r'\toprule',
         r'Endpoint, cohort & Alternative & patient-level mean [95\% CI] & mixed model [95\% CI] & one-sided $P$ & rec.\ median ($P$) \\']
    for met in ('AT', 'RT'):
        for lab in ('internal', 'external'):
            L.append(r'\midrule'); first = True
            for key, label in ALTS:
                r = res[f'{lab}|{met}|{key}']
                L.append(f"{(met + ', ' + lab) if first else ''} & {label} & {ci(r)} & {mm(r)} & {pcol(r, key)} & ${r['median_rec']:+.1f}${R.stars(r['p_wilcoxon_rec'])} \\\\")
                first = False
    L += [r'\bottomrule', r'\end{tabular}']
    out = assert_v2(os.path.join(V2_GENERATED, 'tab_noninferiority_v2.tex')); open(out, 'w').write('\n'.join(L) + '\n'); print('wrote', out)


def part_b():
    import pandas as pd
    EP = [('lvat95', 'LV activation time (95th pct)'), ('tat', 'total activation time'), ('lvdi', 'LV dyssynchrony index (SD)'),
          ('veu', 'ventricular electrical uncoupling'), ('ari_mean', 'mean ARI')]
    PAIRS = [('ours', 'allvalid', 'completion $-$ full vest'), ('vconly', 'allvalid', 'raw $16$-electrode inverse $-$ full vest'),
             ('allvalid', 'commercial', 'full vest $-$ commercial (floor)')]
    res = {}
    rng = np.random.default_rng(0)
    for split, lab in (('test_internal', 'internal'), ('test_external', 'external'), ('holdout', 'replication')):
        df = pd.read_csv(os.path.join(V2_GENERATED, 'clinical_endpoints', f'{split}_template.csv'))
        pats = df['patient'].values; up = sorted(set(pats)); gi = [np.where(pats == p)[0] for p in up]
        boot_idx = rng.integers(0, len(up), size=(B, len(up)))
        for ep, _ in EP:
            for src, ref, _ in PAIRS:
                x = df[f'{src}_{ep}'].values.astype(float); y = df[f'{ref}_{ep}'].values.astype(float)
                ok = np.isfinite(x) & np.isfinite(y); d = x - y
                def est(rows):
                    r = rows[ok[rows]]
                    if r.size < 3:
                        return np.nan, np.nan, np.nan, np.nan
                    dd = d[r]; b = dd.mean(); s = dd.std(ddof=1)
                    return b, b - 1.96 * s, b + 1.96 * s, icc21(x[r], y[r])
                bias, lo, hi, icc = est(np.arange(len(d)))
                bs = np.array([est(np.concatenate([gi[k] for k in row])) for row in boot_idx])
                q = lambda col: (float(np.nanpercentile(bs[:, col], 2.5)), float(np.nanpercentile(bs[:, col], 97.5)))
                res[f'{lab}|{ep}|{src}-{ref}'] = dict(n=int(ok.sum()), bias=float(bias), bias_ci=q(0), loa_lo=float(lo), loa_lo_ci=q(1), loa_hi=float(hi), loa_hi_ci=q(2),
                                                     icc=float(icc), icc_ci=q(3))
    # table
    L = [r'\begin{tabular}{llrrrr}', r'\toprule', r'Cohort, endpoint & Pair & bias [95\% CI] & LoA$_{\mathrm{lo}}$ [95\% CI] & LoA$_{\mathrm{hi}}$ [95\% CI] & ICC(2,1) [95\% CI] \\']
    for split, lab in (('test_internal', 'internal'), ('test_external', 'external'), ('holdout', 'replication')):
        for ep, name in EP:
            L.append(r'\midrule'); first = True
            for src, ref, pl in PAIRS:
                r = res.get(f'{lab}|{ep}|{src}-{ref}')
                if r is None or not np.isfinite(r['bias']):
                    continue
                f = lambda v, c: f"${v:+.1f}$ [${c[0]:+.1f}$, ${c[1]:+.1f}$]"
                L.append(f"{(lab + ', ' + name) if first else ''} & {pl} & {f(r['bias'], r['bias_ci'])} & {f(r['loa_lo'], r['loa_lo_ci'])} & {f(r['loa_hi'], r['loa_hi_ci'])} & "
                         f"{r['icc']:.2f} [{r['icc_ci'][0]:.2f}, {r['icc_ci'][1]:.2f}] \\\\")
                first = False
    L += [r'\bottomrule', r'\end{tabular}']
    out = assert_v2(os.path.join(V2_GENERATED, 'tab_agreement_ci_v2.tex')); open(out, 'w').write('\n'.join(L) + '\n'); print('wrote', out)
    for k, r in res.items():
        if k.endswith('ours-allvalid') or k.endswith('allvalid-commercial'):
            print(f'{k:40s} bias {r["bias"]:+5.1f} [{r["bias_ci"][0]:+5.1f},{r["bias_ci"][1]:+5.1f}] LoA [{r["loa_lo"]:+5.1f},{r["loa_hi"]:+5.1f}] ICC {r["icc"]:.2f} [{r["icc_ci"][0]:.2f},{r["icc_ci"][1]:.2f}]')
    return res


if __name__ == '__main__':
    part_nosub()
    A = part_a(); table_a(A); forest(A); Bres = part_b()
    json.dump({'part_a': A, 'part_b': Bres, 'delta_ms': DELTA, 'bootstrap': B}, open(assert_v2(os.path.join(V2_GENERATED, 'noninferiority_v2.json')), 'w'), indent=1)
    print('wrote noninferiority_v2.json')
