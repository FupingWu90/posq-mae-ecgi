"""All compared methods on the TEST cohorts under BOTH deployment layouts -- fixed template (v2
re-evaluation of the frozen models / ported classical baselines) and per-recording k-means (frozen v1
artefacts) -- doubling grid, commercial reference.  Cohort means (AT / RT MAE in ms, HSP CC, BSP CC)
and the paired per-recording layout difference (template minus k-means: median, Wilcoxon p).
No model is retrained; aggregates only.
  python baselines_analyses/baselines_two_layouts.py
  -> baselines_analyses/generated/baselines_two_layouts.json ; results/baselines_two_layouts.md
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'baselines_analyses'))
from paths import V1_GENERATED, V2_GENERATED, V1_RUNS, assert_v2
from compare_classical_layouts import load_template as cl_template, load_kmeans as cl_kmeans

VCS = [4, 8, 16, 32, 64]
SPLITS = ['test_internal', 'test_external']
TPL = os.path.join(V2_GENERATED, 'percase_template')
# method key -> (label, v2 template CSV pattern, v1 k-means CSV pattern, source column)
LEARNED = {
    'ours-v1': ('Ours (v1 final model)', 'ours-v1_vc{vc}_{split}.csv', 'vc{vc}_{split}.csv', 'src4_recon_valid'),
    'stmem': ('ST-MEM adapted to the vest (index embedding)', 'stmem_vc{vc}_{split}.csv', 'peindex_vc{vc}_{split}.csv', 'src4_recon_valid'),
    'mcma': ('MCMA', 'mcma_vc{vc}_{split}.csv', 'mcma_vc{vc}_{split}.csv', 'src4_recon_valid'),
    'vconly': ('VC-only inverse (no completion)', 'ours-v1_vc{vc}_{split}.csv', 'vc{vc}_{split}.csv', 'src1_vc'),
}
CLASSICAL = {'lmmse': 'LMMSE', 'laplacian': 'Laplace', 'tps': 'TPS'}
METRICS = ['AT_mae', 'RT_mae', 'hsp_cc', 'bsp_cc']
STD = 'cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800'
V1_RUN = {'ours-v1': 'fixedn_ecgi_wide_vc{vc}_' + STD, 'stmem': 'fixedn_ecgi_wide_vc{vc}_' + STD + '_pe-index',
          'mcma': 'mcma_unet_vc{vc}_' + STD}


def engine_bsp_cc(key, vc, split, deploy='cluster-kmean'):
    """Cohort BSP CC of a frozen v1 run from its test_compare.json (engine evaluation, one layout)."""
    f = os.path.join(V1_RUNS, V1_RUN[key].format(vc=vc), 'test_compare.json')
    if not os.path.exists(f):
        return None
    return json.load(open(f))['results'].get(split, {}).get(deploy, {}).get('cc')
ORDER = ['ours-v1', 'stmem', 'mcma', 'lmmse', 'laplacian', 'tps', 'vconly']


def learned_percase(path, source):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[(df.source == source) & (df.ref == 'commercial')].set_index('sid')
    return {m: pd.to_numeric(df[m], errors='coerce') for m in METRICS if m in df.columns}


def classical_percase(entries, method, vc):
    rows = {s: e for (s, m, v), e in entries.items() if m == method and v == vc}
    if not rows:
        return None
    df = pd.DataFrame.from_dict(rows, orient='index')
    ren = {'at_com': 'AT_mae', 'rt_com': 'RT_mae', 'hsp_cc': 'hsp_cc', 'bsp_cc': 'bsp_cc'}
    return {ren[k]: pd.to_numeric(df[k], errors='coerce') for k in ren if k in df.columns}


def paired(T, K):
    cell = {}
    for m in METRICS:
        if T is None or K is None or m not in T or m not in K:
            continue
        a, b = T[m], K[m]
        common = a.index.intersection(b.index); a, b = a.loc[common], b.loc[common]
        ok = a.notna() & b.notna(); d = (a - b)[ok]
        if ok.sum() < 5:
            continue
        p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
        cell[m] = dict(template=float(a[ok].mean()), kmeans=float(b[ok].mean()),
                       median_diff=float(d.median()), p=p, n=int(ok.sum()))
    return cell


def main():
    out, md = {}, ['# Compared methods under both deployment layouts (test cohorts, commercial reference, doubling grid)', '',
                   'tmpl = fixed template layout (v2 re-evaluation, no retraining); km = per-recording k-means (frozen v1 results).',
                   'Paired = per-recording difference tmpl minus km (median, Wilcoxon p; n recordings). Lower AT/RT, higher CC is better.', '']
    for split in SPLITS:
        T_cl, K_cl = cl_template(split), cl_kmeans(split)
        md += [f'## {split}', '',
               '| N | method | AT tmpl | AT km | AT paired (med, p) | RT tmpl | RT km | RT paired (med, p) | HSP CC tmpl | HSP CC km | p | BSP CC tmpl | BSP CC km | n |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for vc in VCS:
            for key in ORDER:
                if key in LEARNED:
                    lab, pt, pk, src = LEARNED[key]
                    T = learned_percase(os.path.join(TPL, pt.format(vc=vc, split=split)), src)
                    K = learned_percase(os.path.join(V1_GENERATED, 'vc_percase', pk.format(vc=vc, split=split)), src)
                else:
                    lab = CLASSICAL[key]
                    T, K = classical_percase(T_cl, key, vc), classical_percase(K_cl, key, vc)
                cell = paired(T, K)
                if key in V1_RUN and 'bsp_cc' not in cell and T is not None and 'bsp_cc' in T:
                    # v1 per-recording k-means CSVs carry no BSP CC: template = per-recording mean of the v2 CSV,
                    # k-means = engine cohort value of the frozen run (same definition, no paired test)
                    t = T['bsp_cc']; t = t[t.notna()]
                    cell['bsp_cc'] = dict(template=float(t.mean()), kmeans=engine_bsp_cc(key, vc, split),
                                          median_diff=float('nan'), p=float('nan'), n=int(t.size))
                out[f'{split}|{key}|{vc}'] = cell
                g = lambda m, f: cell.get(m, {}).get(f, float('nan'))
                ff = lambda x, n=2: ('%.*f' % (n, x)) if isinstance(x, (int, float)) and np.isfinite(x) else '--'
                n = cell.get('AT_mae', {}).get('n', 0)
                md.append(f"| {vc} | {lab} | {ff(g('AT_mae','template'))} | {ff(g('AT_mae','kmeans'))} | {ff(g('AT_mae','median_diff'))}, {ff(g('AT_mae','p'),3)} | "
                          f"{ff(g('RT_mae','template'))} | {ff(g('RT_mae','kmeans'))} | {ff(g('RT_mae','median_diff'))}, {ff(g('RT_mae','p'),3)} | "
                          f"{ff(g('hsp_cc','template'),3)} | {ff(g('hsp_cc','kmeans'),3)} | {ff(g('hsp_cc','p'),3)} | "
                          f"{ff(g('bsp_cc','template'),3)} | {ff(g('bsp_cc','kmeans'),3)} | {n} |")
        md.append('')
    jp = assert_v2(os.path.join(V2_GENERATED, 'baselines_two_layouts.json'))
    json.dump(out, open(jp, 'w'), indent=1)
    mp = assert_v2(os.path.join(V2, 'results', 'baselines_two_layouts.md'))
    open(mp, 'w').write('\n'.join(md))
    print('\n'.join(md)); print('wrote', jp, 'and', mp)


if __name__ == '__main__':
    main()
