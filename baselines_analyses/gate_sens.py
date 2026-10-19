"""Gating sensitivity at N = 16 (supplement S5, tab:gate-sens) for the v2 model under both deployment layouts.
Columns: AT MAE on the reference-gated set (QRS gate) and on all vertices; RT MAE on the T-gated set, on the
QRS-gated set (RT_QRS) and on all vertices -- all vs the commercial reference, cohort means of the per-recording
spatial means (CSV columns AT_mae / AT_mae_allv / RT_mae / RT_mae_all / RT_mae_allv).  Rows: Ours (PosQ-MAE),
ST-MEM, MCMA, the VC-only inverse (src1 rows), the all-valid inverse as a source (refs cache: AT2/RT2 vs
ATcom/RTcom with the same map_errors convention as the CSVs), and the median paired differences Ours - VC-only
and Ours - all-valid with Wilcoxon stars.
  python baselines_analyses/gate_sens.py   -> generated/tab_gate_sens_v2.tex (template), tab_gate_sens_km_v2.tex (k-means),
                                 gate_sens_v2.json.  Aggregates only."""
import os, sys, json, glob
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from paths import V1_GENERATED, V2_GENERATED, assert_v2
from cache_references import CACHE_ROOT, CACHE_ROOT_V1
from metrics import map_errors

VC = 16
SPLITS = ('test_internal', 'test_external')
COLS = [('AT_mae', 'AT gated'), ('AT_mae_allv', 'AT all'), ('RT_mae', 'RT gated'), ('RT_mae_all', 'RT QRS'), ('RT_mae_allv', 'RT all')]
STD = 'cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800'


def stars(p):
    if p is None or not np.isfinite(p):
        return ''
    return '$^{***}$' if p < 1e-3 else '$^{**}$' if p < 1e-2 else '$^{*}$' if p < 5e-2 else '$^{\\mathrm{ns}}$'


def percase(path, source):
    X = pd.read_csv(path)
    X = X[(X.source == source) & (X.ref == 'commercial')].set_index('sid')
    return X[[c for c, _ in COLS]].apply(pd.to_numeric, errors='coerce')


def allvalid(split):
    files = {}
    for root in (CACHE_ROOT, CACHE_ROOT_V1):
        for f in glob.glob(os.path.join(root, split, 'refs', '*.npz')):
            files.setdefault(os.path.basename(f)[:-4], f)
    rows = {}
    for sid, f in files.items():
        with np.load(f, allow_pickle=False) as z:
            if not all(k in z.files for k in ('AT2', 'RT2', 'ATcom', 'RTcom', 'qrs_goodcom', 't_goodcom')):
                continue
            at2, atc, rt2, rtc = (z[k].astype(float) for k in ('AT2', 'ATcom', 'RT2', 'RTcom'))
            q, t = z['qrs_goodcom'].astype(bool), z['t_goodcom'].astype(bool)
            rows[sid] = {'AT_mae': map_errors(at2, atc, q)[0], 'AT_mae_allv': map_errors(at2, atc, None)[0],
                         'RT_mae': map_errors(rt2, rtc, t)[0], 'RT_mae_all': map_errors(rt2, rtc, q)[0],
                         'RT_mae_allv': map_errors(rt2, rtc, None)[0]}
    return pd.DataFrame(rows).T


def sources(layout, split):
    pc = os.path.join(V2_GENERATED, 'percase_candidate', f'LOC3_vc{VC}_{split}_{layout}.csv')
    S = {'ours': percase(pc, 'src4_recon_valid'), 'vconly': percase(pc, 'src1_vc')}
    if layout == 'template':
        S['stmem'] = percase(os.path.join(V2_GENERATED, 'percase_template', f'stmem_vc{VC}_{split}.csv'), 'src4_recon_valid')
        S['mcma'] = percase(os.path.join(V2_GENERATED, 'percase_template', f'mcma_vc{VC}_{split}.csv'), 'src4_recon_valid')
    else:
        S['stmem'] = percase(os.path.join(V1_GENERATED, 'vc_percase', f'peindex_vc{VC}_{split}.csv'), 'src4_recon_valid')
        S['mcma'] = percase(os.path.join(V1_GENERATED, 'vc_percase', f'mcma_vc{VC}_{split}.csv'), 'src4_recon_valid')
    S['allvalid'] = allvalid(split)
    # the all-valid frame may hold recordings the CSVs do not (and vice versa); the index of the ours CSV is the cohort
    S['allvalid'] = S['allvalid'].reindex(S['ours'].index.unique())
    return S


def main():
    out = {}
    f2 = lambda v: ('%.2f' % v) if np.isfinite(v) else '--'
    for layout, name in (('template', 'tab_gate_sens_v2.tex'), ('kmeans', 'tab_gate_sens_km_v2.tex')):
        L = [r'\begin{tabular}{llccccc}', r'\toprule',
             r' & & \multicolumn{2}{c}{AT MAE} & \multicolumn{3}{c}{RT MAE} \\', r'\cmidrule(lr){3-4}\cmidrule(lr){5-7}',
             r'Cohort & Source & gated & all & gated & RT$_{\text{QRS}}$ & all \\']
        for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
            S = sources(layout, split)
            L.append(r'\midrule')
            res = {}
            for key, rlab in (('ours', 'Ours'), ('stmem', 'ST-MEM'), ('mcma', 'MCMA'), ('vconly', 'VC-only'), ('allvalid', 'all-valid')):
                m = {c: float(S[key][c].mean()) for c, _ in COLS}
                res[key] = {'mean': m, 'n': int(S[key][COLS[0][0]].notna().sum())}
                L.append((lab if key == 'ours' else '') + f' & {rlab} & ' + ' & '.join(f2(m[c]) for c, _ in COLS) + r' \\')
            for key, rlab in (('vconly', 'VC-only'), ('allvalid', 'all-valid')):
                cells, dd = [], {}
                common = S['ours'].index.intersection(S[key].index)
                for c, _ in COLS:
                    d = (S['ours'].loc[common, c] - S[key].loc[common, c]).dropna()
                    p = float(wilcoxon(d).pvalue) if len(d) >= 10 and (d != 0).any() else None
                    dd[c] = {'med': float(d.median()), 'p': p, 'n': int(len(d))}
                    cells.append(f'${d.median():+.1f}$' + stars(p))
                res[f'diff_{key}'] = dd
                L.append(f' & $\\Delta$ Ours$-${rlab} & ' + ' & '.join(cells) + r' \\')
            # headline paired comparisons vs the learned baselines (JSON / printout only): sign and significance
            for key in ('stmem', 'mcma'):
                dd = {}
                common = S['ours'].index.intersection(S[key].index)
                for c, _ in COLS:
                    d = (S['ours'].loc[common, c] - S[key].loc[common, c]).dropna()
                    p = float(wilcoxon(d).pvalue) if len(d) >= 10 and (d != 0).any() else None
                    dd[c] = {'med': float(d.median()), 'p': p, 'n': int(len(d))}
                res[f'diff_{key}'] = dd
            out[f'{layout}|{split}'] = res
            print(f'--- {layout} / {split}: ' + ' | '.join(f"{k} AT {v['mean']['AT_mae']:.2f}/{v['mean']['AT_mae_allv']:.2f} RT {v['mean']['RT_mae']:.2f}/{v['mean']['RT_mae_all']:.2f}/{v['mean']['RT_mae_allv']:.2f} (n={v['n']})" for k, v in res.items() if 'mean' in v))
            for key in ('vconly', 'allvalid', 'stmem', 'mcma'):
                print('    delta ours-' + key + ': ' + ', '.join(f"{c} {v['med']:+.2f} p={v['p']:.2g}" for c, v in res[f'diff_{key}'].items()))
        L += [r'\bottomrule', r'\end{tabular}']
        open(assert_v2(os.path.join(V2_GENERATED, name)), 'w').write('\n'.join(L) + '\n'); print('wrote', name)
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'gate_sens_v2.json')), 'w'), indent=1)


if __name__ == '__main__':
    main()
