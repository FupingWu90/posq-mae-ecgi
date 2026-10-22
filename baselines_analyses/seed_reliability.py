"""Reliability flag from training-seed disagreement (collaborators' item B4, 2026-10-21).  Three training seeds of the final
configuration (LOC3 = seed 42, P3S43, P3S44) complete every test recording; where their activation maps disagree, the
completion is uncertain.  Per recording: D = mean over the commercial-gated vertices of the SD of the three seeds'
activation times (ms; RT analogously on the T-gated vertices); E = error of the seed-42 map against the commercial
reference and against the full-vest inverse; also the error of the three-seed mean map (ensemble).  Statistics per
cohort: Spearman rho(D, E); flagging the 20% of recordings with the largest D (and an absolute 5-ms rule): error of the
flagged vs the unflagged recordings, and the AUC of D for identifying the recordings whose error is in the top decile;
the same for the LV activation time of baselines_analyses/clinical_endpoints.py (|LVAT95 completion - full vest|).  Fixed template
layout.  Aggregates only.
  python baselines_analyses/seed_reliability.py [--ids LOC3 P3S43 P3S44] [--splits test_internal test_external] [--top 0.2]
  -> generated/seed_reliability/<split>.csv, seed_reliability_v2.json, tab_seed_reliability_v2.tex"""
import os, sys, csv, json, argparse, warnings
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('bsp_reconstruction', 'baselines_analyses', 'inverse_evaluation'):
    sys.path.insert(0, os.path.join(V2, d))
from paths import V2_GENERATED, assert_v2
from cache_references import cache_lookup
from dataset import collect_split
from dedup import ok as dedup_ok
from patient_ids import patient_map
from tgate_rule import T_MIN
OUT = os.path.join(V2_GENERATED, 'seed_reliability')


def auc(score, label):
    """rank-based AUC of `score` for the binary `label`."""
    s = np.asarray(score, float); y = np.asarray(label, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s); return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='+', default=['LOC3', 'P3S43', 'P3S44']); ap.add_argument('--splits', nargs='+', default=['test_internal', 'test_external'])
    ap.add_argument('--layout', default='template'); ap.add_argument('--top', type=float, default=0.2); ap.add_argument('--abs_ms', type=float, default=5.0); ap.add_argument('--n', type=int, default=10**9)
    a = ap.parse_args(); warnings.filterwarnings('ignore'); os.makedirs(OUT, exist_ok=True)
    b, _ = collect_split(seed=42); res = {'ids': a.ids, 'layout': a.layout, 'top_fraction': a.top, 'abs_ms': a.abs_ms}
    for split in a.splits:
        pm = patient_map(split); rows = []; n_miss = 0
        ce = {}
        cep = os.path.join(V2_GENERATED, 'clinical_endpoints', f'{split}_{a.layout}.csv')
        if os.path.exists(cep):
            for r in csv.DictReader(open(cep)):
                try:
                    ce[r['sid']] = abs(float(r['ours_lvat95']) - float(r['allvalid_lvat95']))
                except (TypeError, ValueError):
                    pass
        for sid, _e, torso in [r for r in b[split] if r[2] and dedup_ok(r[0])][:a.n]:
            sid_safe = sid.replace('/', '_')
            paths = [os.path.join(V2_GENERATED, 'atmaps', i, f'{split}_{a.layout}', sid_safe + '.npz') for i in a.ids]
            fr = cache_lookup(split, 'refs', sid_safe + '.npz')
            if fr is None or not all(os.path.exists(p) for p in paths):
                n_miss += 1; continue
            ats, rts = [], []
            for p in paths:
                with np.load(p, allow_pickle=False) as z:
                    if not bool(z['valid']):
                        ats = None; break
                    ats.append(z['AT4'].astype(float)); rts.append(z['RT4'].astype(float))
            if ats is None:
                n_miss += 1; continue
            with np.load(fr, allow_pickle=False) as zr:
                if 'ATcom' not in zr.files:
                    n_miss += 1; continue
                atc, at2, rtc, rt2 = zr['ATcom'].astype(float), zr['AT2'].astype(float), zr['RTcom'].astype(float), zr['RT2'].astype(float)
                qg, tg, qg2, tg2 = zr['qrs_goodcom'].astype(bool), zr['t_goodcom'].astype(bool), zr['qrs_good2'].astype(bool), zr['t_good2'].astype(bool)
            A = np.stack(ats); R = np.stack(rts)
            row = {'sid': sid, 'patient': pm.get(sid, ''), 'n_q': int(qg.sum()), 'n_t': int(tg.sum()),
                   'D_at': float(A.std(0, ddof=1)[qg].mean()), 'E_at_com': float(np.abs(A[0] - atc)[qg].mean()), 'E_at_fv': float(np.abs(A[0] - at2)[qg2].mean()) if qg2.sum() else np.nan,
                   'E_at_fv_com': float(np.abs(at2 - atc)[qg].mean()), 'E_at_ens_com': float(np.abs(A.mean(0) - atc)[qg].mean()),
                   'E_at_s43_com': float(np.abs(A[1] - atc)[qg].mean()), 'E_at_s44_com': float(np.abs(A[2] - atc)[qg].mean()),
                   'D_rt': float(R.std(0, ddof=1)[tg].mean()) if tg.sum() >= T_MIN else np.nan,
                   'E_rt_com': float(np.abs(R[0] - rtc)[tg].mean()) if tg.sum() >= T_MIN else np.nan, 'E_rt_ens_com': float(np.abs(R.mean(0) - rtc)[tg].mean()) if tg.sum() >= T_MIN else np.nan,
                   'dLVAT95_fv': ce.get(sid, np.nan)}
            rows.append(row)
        if not rows:
            print(f'[{split}] no recordings (missing {n_miss})'); continue
        with open(os.path.join(OUT, f'{split}_{a.layout}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        g = lambda k: np.array([r[k] for r in rows], float)
        D, Ec, Ef, Efc, Ens, Dr, Er, Ensr, dL = (g(k) for k in ('D_at', 'E_at_com', 'E_at_fv', 'E_at_fv_com', 'E_at_ens_com', 'D_rt', 'E_rt_com', 'E_rt_ens_com', 'dLVAT95_fv'))
        S = {'n': len(rows), 'missing': n_miss, 'D_at_median': float(np.median(D)), 'D_at_iqr': [float(np.quantile(D, .25)), float(np.quantile(D, .75))],
             'E_at_com_mean': float(Ec.mean()), 'E_at_ens_com_mean': float(Ens.mean()), 'E_at_seeds_com_mean': [float(Ec.mean()), float(g('E_at_s43_com').mean()), float(g('E_at_s44_com').mean())],
             'rho_D_Ecom': float(spearmanr(D, Ec)[0]), 'rho_D_Efv': float(spearmanr(D[np.isfinite(Ef)], Ef[np.isfinite(Ef)])[0]), 'rho_D_Efvcom': float(spearmanr(D, Efc)[0])}
        m = np.isfinite(dL)
        if m.sum() > 10:
            S['rho_D_dLVAT95'] = float(spearmanr(D[m], dL[m])[0])
        for name, thr in (('top', float(np.quantile(D, 1 - a.top))), ('abs', a.abs_ms)):
            fl = D > thr; S[f'flag_{name}'] = {'threshold_ms': thr, 'n_flagged': int(fl.sum()), 'frac_flagged': float(fl.mean()),
                                               'E_at_com_flagged_median': float(np.median(Ec[fl])) if fl.any() else np.nan, 'E_at_com_unflagged_median': float(np.median(Ec[~fl])) if (~fl).any() else np.nan,
                                               'E_at_fv_flagged_median': float(np.nanmedian(Ef[fl])) if fl.any() else np.nan, 'E_at_fv_unflagged_median': float(np.nanmedian(Ef[~fl])) if (~fl).any() else np.nan,
                                               'E_at_com_unflagged_mean': float(Ec[~fl].mean()) if (~fl).any() else np.nan,
                                               'p_mannwhitney_Ecom': float(mannwhitneyu(Ec[fl], Ec[~fl]).pvalue) if fl.any() and (~fl).any() else np.nan}
            if m.sum() > 10 and fl.any() and (~fl).any():
                S[f'flag_{name}']['dLVAT95_flagged_median'] = float(np.nanmedian(dL[fl])); S[f'flag_{name}']['dLVAT95_unflagged_median'] = float(np.nanmedian(dL[~fl]))
        top_err = Ec >= np.quantile(Ec, 0.9); S['auc_D_for_top_decile_Ecom'] = auc(D, top_err)
        if np.isfinite(Ef).sum() > 10:
            Efm = np.where(np.isfinite(Ef), Ef, np.nan); thr = np.nanquantile(Efm, 0.9); S['auc_D_for_top_decile_Efv'] = auc(D[np.isfinite(Efm)], Efm[np.isfinite(Efm)] >= thr)
        if m.sum() > 10:
            S['auc_D_for_dLVAT95_gt10'] = auc(D[m], dL[m] > 10)
        mr = np.isfinite(Dr) & np.isfinite(Er)
        if mr.sum() > 10:
            S['rt'] = {'n': int(mr.sum()), 'D_rt_median': float(np.median(Dr[mr])), 'rho_D_Ecom': float(spearmanr(Dr[mr], Er[mr])[0]), 'E_rt_com_mean': float(Er[mr].mean()), 'E_rt_ens_com_mean': float(np.nanmean(Ensr[mr]))}
        res[split] = S
        print(f'[{split}] n={len(rows)} (missing {n_miss}); D_at median {S["D_at_median"]:.1f} ms; rho(D,E vs commercial) {S["rho_D_Ecom"]:.2f}, (D, E vs full vest) {S["rho_D_Efv"]:.2f}, (D, full-vest error) {S["rho_D_Efvcom"]:.2f}; '
              f'ensemble AT {S["E_at_ens_com_mean"]:.2f} vs seed42 {S["E_at_com_mean"]:.2f}; top-20% flag: flagged/unflagged median E_com {S["flag_top"]["E_at_com_flagged_median"]:.1f}/{S["flag_top"]["E_at_com_unflagged_median"]:.1f}, '
              f'E_fv {S["flag_top"]["E_at_fv_flagged_median"]:.1f}/{S["flag_top"]["E_at_fv_unflagged_median"]:.1f}; AUC top-decile E_com {S["auc_D_for_top_decile_Ecom"]:.2f}, E_fv {S.get("auc_D_for_top_decile_Efv", float("nan")):.2f}' + (f'; rho(D, dLVAT95) {S["rho_D_dLVAT95"]:.2f}' if 'rho_D_dLVAT95' in S else ''))
    json.dump(res, open(assert_v2(os.path.join(V2_GENERATED, 'seed_reliability_v2.json')), 'w'), indent=1)
    L = [r'\begin{tabular}{lrrrrrrr}', r'\toprule', r'Cohort & $n$ & $D$ median [IQR] (ms) & $\rho(D, E_{\mathrm{com}})$ & $\rho(D, E_{\mathrm{vest}})$ & flagged $20\%$: $E_{\mathrm{com}}$ flagged / rest & AUC (top decile $E_{\mathrm{com}}$) & $\rho(D, \Delta$LVAT$)$ \\', r'\midrule']
    for split in a.splits:
        S = res.get(split)
        if not S:
            continue
        ft = S['flag_top']
        L.append(f"{'internal' if 'internal' in split else 'external'} & {S['n']} & {S['D_at_median']:.1f} [{S['D_at_iqr'][0]:.1f}, {S['D_at_iqr'][1]:.1f}] & {S['rho_D_Ecom']:.2f} & {S['rho_D_Efv']:.2f} & "
                 f"{ft['E_at_com_flagged_median']:.1f} / {ft['E_at_com_unflagged_median']:.1f} & {S['auc_D_for_top_decile_Ecom']:.2f} & {S.get('rho_D_dLVAT95', float('nan')):.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_seed_reliability_v2.tex')), 'w').write('\n'.join(L) + '\n'); print('wrote seed_reliability_v2.json, tab_seed_reliability_v2.tex')


if __name__ == '__main__':
    main()
