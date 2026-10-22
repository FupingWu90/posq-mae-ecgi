"""Half-torso deployment of the final model (supplement S8, tab:region-deploy; v2 port of the deployment-side part
of the frozen v1 region_analysis.py, both deployment protocols).  Rows = cohort means (BSP CC on held-out electrodes,
HSP CC, AT / RT / ARI MAE vs the commercial reference) of per-recording CSVs; Delta = median paired difference to
the reference row (full-torso deployment of the same protocol) with Wilcoxon stars; VC-only rows = the same
restricted electrodes without completion (src1_vc).
  CSV sources: full torso  generated/percase_candidate/<ID>_vc16_<split>_<layout>.csv
               half torso  generated/percase_region/<ID>_vc16_<split>_<cond>.csv  (training_sweeps/run_deploy_robustness.sh)
  python baselines_analyses/region_deploy_v2.py [--id LOC3]  -> generated/tab_region_deploy_v2.tex, region_deploy_v2.json
A second block for region-trained models (P3REGF / P3REGB) is added automatically when their region CSVs exist
(same file pattern with their IDs).  Prints aggregates only."""
import os, sys, json, argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from paths import V2_GENERATED, assert_v2

SPLITS = ('test_internal', 'test_external')
PC = os.path.join(V2_GENERATED, 'percase_candidate'); PR = os.path.join(V2_GENERATED, 'percase_region')
KEYS = ('bsp_cc', 'hsp_cc', 'AT_mae', 'RT_mae', 'ARI_mae', 'n_region_fallback', 'n_template_sub')
# (row label, protocol, condition key) ; condition None = full-torso deployment of that protocol
ROWS = [('full torso (paper)', 'template', None), ('anterior template', 'template', 'tpl_front'), ('posterior template', 'template', 'tpl_back'),
        ('full torso ($k$-means)', 'kmeans', None), ('anterior half, $k$-means', 'kmeans', 'km_front'), ('posterior half, $k$-means', 'kmeans', 'km_back')]


REG_ROWS = [('anterior-trained, anterior template', 'P3REGF', 'template', 'tpl_front'),
            ('anterior-trained, anterior half, $k$-means', 'P3REGF', 'kmeans', 'km_front'),
            ('posterior-trained, posterior template', 'P3REGB', 'template', 'tpl_back'),
            ('posterior-trained, posterior half, $k$-means', 'P3REGB', 'kmeans', 'km_back')]


def stars(p):
    return '$^{***}$' if p < 1e-3 else '$^{**}$' if p < 1e-2 else '$^{*}$' if p < 5e-2 else '$^{\\mathrm{ns}}$'


def load(path, src):
    if not os.path.exists(path):
        return None
    X = pd.read_csv(path)
    X = X[(X.source == src) & (X.ref == 'commercial')].set_index('sid')
    return X[[k for k in KEYS if k in X.columns]].apply(pd.to_numeric, errors='coerce')


def csv_for(pid, split, layout, cond):
    return os.path.join(PC, f'{pid}_vc16_{split}_{layout}.csv') if cond is None else os.path.join(PR, f'{pid}_vc16_{split}_{cond}.csv')


def paired(d, base, k):
    common = d.index.intersection(base.index)
    x = (d.loc[common, k] - base.loc[common, k]).dropna()
    p = float(wilcoxon(x).pvalue) if len(x) >= 10 and (x != 0).any() else 1.0
    return float(x.median()), p, int(len(x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--id', default='LOC3'); ap.add_argument('--engine_cc', action='store_true')
    a = ap.parse_args()
    f2 = lambda v, n=2: ('%.*f' % (n, v)) if np.isfinite(v) else '--'
    out = {}
    L = [r'\begin{tabular}{llccccccc}', r'\toprule', r'Cohort & Deployment & BSP CC & HSP CC & AT & $\Delta$AT & RT & $\Delta$RT & ARI \\']
    for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
        L.append(r'\midrule')
        base = {}
        first = True
        for label, layout, cond in ROWS:
            d = load(csv_for(a.id, split, layout, cond), 'src4_recon_valid')
            if d is None:
                L.append((lab if first else '') + f' & {label} & ' + ' & '.join(['--'] * 7) + r' \\'); first = False
                continue
            if cond is None:
                base[layout] = d
            m = {k: float(d[k].mean()) for k in ('bsp_cc', 'hsp_cc', 'AT_mae', 'RT_mae', 'ARI_mae')}
            rec = {'mean': m, 'n': int(len(d))}
            if cond is not None and layout in base:
                for k in ('AT_mae', 'RT_mae', 'ARI_mae', 'hsp_cc', 'bsp_cc'):
                    rec[f'd_{k}'] = paired(d, base[layout], k)
                rec['n_fallback_recs'] = int((d['n_region_fallback'] > 0).sum()) if 'n_region_fallback' in d else None
                rec['mean_template_sub'] = float(d['n_template_sub'].mean()) if 'n_template_sub' in d else None
                dAT, pAT, _ = rec['d_AT_mae']; dRT, pRT, _ = rec['d_RT_mae']
                cells = [f2(m['bsp_cc'], 3), f2(m['hsp_cc'], 3), f2(m['AT_mae']), f'${dAT:+.1f}$' + stars(pAT), f2(m['RT_mae']), f'${dRT:+.1f}$' + stars(pRT), f2(m['ARI_mae'])]
            else:
                cells = [f2(m['bsp_cc'], 3), f2(m['hsp_cc'], 3), f2(m['AT_mae']), '--', f2(m['RT_mae']), '--', f2(m['ARI_mae'])]
            L.append((lab if first else '') + f' & {label} & ' + ' & '.join(cells) + r' \\'); first = False
            out[f'{split}|{label}'] = rec
            if cond is not None:
                v = load(csv_for(a.id, split, layout, cond), 'src1_vc')
                if v is not None:
                    mv = {k: float(v[k].mean()) for k in ('hsp_cc', 'AT_mae', 'RT_mae', 'ARI_mae')}
                    out[f'{split}|VC-only {label}'] = {'mean': mv, 'n': int(len(v))}
                    L.append(f' & VC-only, {label} & -- & {f2(mv["hsp_cc"], 3)} & {f2(mv["AT_mae"])} & & {f2(mv["RT_mae"])} & & {f2(mv["ARI_mae"])} \\\\')
    # ---- lower block: the final configuration retrained with the training selection confined to one half ----
    # (P3REGF anterior / P3REGB posterior), deployed on the same half; Delta vs the full-torso-trained model (a.id) on the
    # SAME electrodes (same template / same restricted k-means); the differences to a.id's full-torso deployment go to the JSON.
    have = any(os.path.exists(csv_for(pid, s, lay, cond)) for _, pid, lay, cond in REG_ROWS for s in SPLITS)
    if have:
        L.append(r'\midrule')
        L.append(r'\multicolumn{9}{l}{\emph{Final configuration retrained with the training selection confined to the same half '
                 r'($\Delta$ vs.\ the full-torso-trained model on the same electrodes)}} \\')
        for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
            L.append(r'\midrule')
            first = True
            for label, pid, layout, cond in REG_ROWS:
                d = load(csv_for(pid, split, layout, cond), 'src4_recon_valid')
                same = load(csv_for(a.id, split, layout, cond), 'src4_recon_valid')
                full = load(csv_for(a.id, split, layout, None), 'src4_recon_valid')
                if d is None or same is None or full is None:
                    L.append((lab if first else '') + f' & {label} & ' + ' & '.join(['--'] * 7) + r' \\'); first = False
                    continue
                m = {k: float(d[k].mean()) for k in ('bsp_cc', 'hsp_cc', 'AT_mae', 'RT_mae', 'ARI_mae')}
                rec = {'mean': m, 'n': int(len(d)), 'pid': pid, 'cond': cond}
                for k in ('AT_mae', 'RT_mae', 'ARI_mae', 'hsp_cc', 'bsp_cc'):
                    rec[f'd_same_{k}'] = paired(d, same, k)        # vs full-torso-trained, same electrodes
                    rec[f'd_full_{k}'] = paired(d, full, k)        # vs full-torso-trained, full-torso deployment
                dAT, pAT, _ = rec['d_same_AT_mae']; dRT, pRT, _ = rec['d_same_RT_mae']
                cells = [f2(m['bsp_cc'], 3), f2(m['hsp_cc'], 3), f2(m['AT_mae']), f'${dAT:+.1f}$' + stars(pAT), f2(m['RT_mae']), f'${dRT:+.1f}$' + stars(pRT), f2(m['ARI_mae'])]
                L.append((lab if first else '') + f' & {label} & ' + ' & '.join(cells) + r' \\'); first = False
                out[f'{split}|{label}'] = rec
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_region_deploy_v2.tex')), 'w').write('\n'.join(L) + '\n')
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'region_deploy_v2.json')), 'w'), indent=1)
    for k, v in out.items():
        m = v['mean']
        line = f"{k:45s} n={v['n']:3d} " + ' '.join(f"{kk} {vv:.3f}" if 'cc' in kk else f"{kk[:-4]} {vv:.2f}" for kk, vv in m.items())
        if 'd_AT_mae' in v:
            line += ' | paired: ' + ', '.join(f"{kk[2:-4] if kk.endswith('mae') else kk[2:]} {vv[0]:+.2f} (p={vv[1]:.1e})" for kk, vv in v.items() if kk.startswith('d_'))
            line += f" | fallback recs {v.get('n_fallback_recs')} | mean template subst. {v.get('mean_template_sub')}"
        elif 'd_same_AT_mae' in v:
            line += ' | vs same electrodes: ' + ', '.join(f"{kk[7:-4] if kk.endswith('mae') else kk[7:]} {vv[0]:+.2f} (p={vv[1]:.1e})" for kk, vv in v.items() if kk.startswith('d_same_'))
            line += ' | vs full torso: ' + ', '.join(f"{kk[7:-4] if kk.endswith('mae') else kk[7:]} {vv[0]:+.2f} (p={vv[1]:.1e})" for kk, vv in v.items() if kk.startswith('d_full_'))
        print(line)
    print('wrote tab_region_deploy_v2.tex')


if __name__ == '__main__':
    main()
