"""Off-grid A/B test of the physical position code (reviewer point 9, 2026-10-23).

--build    Fixed index split of the 252 kept channels into two halves A / B (random permutation, seed 0)
           -> bsp_reconstruction/templates/subset_AB.npz (subA, subB: (252,) bool), and fixed template layouts restricted to each
           half for N in (4, 8, 16, 32, 64) -> bsp_reconstruction/templates/template_vc<N>_sub{A,B}.npz (k-means medoids of the
           mean normalised TRAINING geometry inside the half; same recipe as bsp_reconstruction/build_templates.py).
           Training with `--select_region subA` draws every visible electrode from half A, so the positions of half B
           are never seen as inputs; evaluation with `--region subB` (template or k-means) then measures the
           completion from positions the model never saw as inputs ('off-grid').  Prints aggregates only.
--summary  Paired comparison of the per-recording CSVs in generated/percase_offgrid/ (training_sweeps/run_offgrid_ab.sh):
           control model (trained on all positions) vs the half-A model, both deployed on half B, plus the half-A
           model on its own half -> generated/offgrid_ab_v2.json and tab_offgrid_ab_v2.tex.
"""
import os, sys, json, argparse
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from paths import assert_v2, V2_GENERATED
TPL = os.path.join(V2, 'bsp_reconstruction', 'templates')
NS = (4, 8, 16, 32, 64)


def build():
    import dataset as D
    import channel_select as CS
    b, _ = D.collect_split(42)
    keep = np.where(D.geometry_union(b['train']))[0].astype(np.int64)
    assert len(keep) == 252, len(keep)
    tors = sorted({tp for _, _, tp in b['train'] if tp})
    C = np.stack([D._coords_for(tp)[keep] for tp in tors])
    ok = np.abs(C).sum(-1) > 1e-6
    mean_xyz = ((C * ok[..., None]).sum(0) / np.maximum(ok.sum(0), 1)[:, None]).astype(np.float32)
    nn = np.linalg.norm(mean_xyz[:, None] - mean_xyz[None], axis=-1); np.fill_diagonal(nn, 9)
    spacing = float(np.median(nn.min(1)))
    perm = np.random.default_rng(0).permutation(252)
    subA = np.zeros(252, bool); subA[perm[:126]] = True; subB = ~subA
    np.savez(assert_v2(os.path.join(TPL, 'subset_AB.npz')), subA=subA, subB=subB, seed=0)
    CS._SUBSET_MASKS.clear()
    dAB = nn[np.ix_(subB, subA)].min(1) / spacing          # distance of every B electrode to its nearest A electrode
    meta = {'seed': 0, 'n_A': int(subA.sum()), 'n_B': int(subB.sum()), 'spacing_normalised': spacing,
            'B_to_nearest_A_spacing': {'median': float(np.median(dAB)), 'mean': float(dAB.mean()), 'max': float(dAB.max()),
                                       'frac_gt_1.5': float((dAB > 1.5).mean())},
            'anterior_A': int((-mean_xyz[subA, 1] > 0).sum()), 'anterior_B': int((-mean_xyz[subB, 1] > 0).sum()), 'layouts': {}}
    main16 = CS.load_template(16)[0]
    for sub, name in ((subA, 'subA'), (subB, 'subB')):
        for n in NS:
            vis = CS.select_visible_channels(mean_xyz, np.ones(252, bool), n, 'cluster', 'kmean', np.random.default_rng(42), region=name)
            idx = np.where(vis)[0].astype(np.int64)
            assert len(idx) == n and sub[idx].all()
            np.savez(assert_v2(os.path.join(TPL, f'template_vc{n}_{name}.npz')), idx=idx, mean_xyz=mean_xyz, keep_idx=keep)
            other = subA if name == 'subB' else subB
            d_other = nn[np.ix_(idx, other)].min(1) / spacing
            d_main = np.linalg.norm(mean_xyz[idx][:, None] - mean_xyz[main16][None], axis=-1).min(1) / spacing
            meta['layouts'][f'{name}_N{n}'] = {'idx252': idx.tolist(), 'n_anterior': int((-mean_xyz[idx, 1] > 0).sum()),
                                              'nearest_other_half_spacing_median': float(np.median(d_other)),
                                              'nearest_main16_spacing_median': float(np.median(d_main)),
                                              'shared_with_main16': int(len(set(idx.tolist()) & set(main16.tolist())))}
            print(f'{name} N={n:2d}: {n} electrodes, {meta["layouts"][f"{name}_N{n}"]["n_anterior"]} anterior; nearest other-half electrode '
                  f'{np.median(d_other):.2f} spacing (median); shared with main template {meta["layouts"][f"{name}_N{n}"]["shared_with_main16"]}')
    json.dump(meta, open(os.path.join(TPL, 'subset_AB.json'), 'w'), indent=1)
    print(f"A {meta['n_A']} / B {meta['n_B']} electrodes; B-to-nearest-A distance median {meta['B_to_nearest_A_spacing']['median']:.2f}, "
          f"mean {meta['B_to_nearest_A_spacing']['mean']:.2f}, max {meta['B_to_nearest_A_spacing']['max']:.2f} spacings; "
          f"anterior A/B {meta['anterior_A']}/{meta['anterior_B']}")


def summary():
    import pandas as pd
    from scipy.stats import wilcoxon
    D = os.path.join(V2_GENERATED, 'percase_offgrid')
    out = {}; L = [r'\begin{tabular}{llcccccc}', r'\toprule',
                   r' & & \multicolumn{3}{c}{internal} & \multicolumn{3}{c}{external} \\', r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
                   r'Layout & Model & BSP CC & AT MAE & RT MAE & BSP CC & AT MAE & RT MAE \\']
    conds = [('tplB', 'template, half B'), ('kmB', 'k-means, half B'), ('tplA', 'template, half A'), ('kmA', 'k-means, half A')]
    models = [('LOC3', 'all positions (control)'), ('OFFGRIDA', 'half A only')]
    for cond, clab in conds:
        L.append(r'\midrule'); first = True
        for mid, mlab in models:
            cells = []
            for sp in ('test_internal', 'test_external'):
                p = os.path.join(D, f'{mid}_vc16_{sp}_{cond}.csv')
                if not os.path.exists(p):
                    cells += ['--'] * 3; continue
                df = pd.read_csv(p); df = df[df.success == 1] if 'success' in df else df
                s4 = df[(df.source == 'src4_recon_valid') & (df.ref == 'commercial')]
                vals = [float(np.nanmean(s4.bsp_cc)), float(np.nanmean(s4.AT_mae)), float(np.nanmean(s4.RT_mae))]
                out[f'{cond}|{mid}|{sp}'] = dict(bsp_cc=vals[0], AT=vals[1], RT=vals[2], n=int(len(s4)))
                cell = [f'{vals[0]:.3f}', f'{vals[1]:.1f}', f'{vals[2]:.1f}']
                if mid != 'LOC3':
                    pc = os.path.join(D, f'LOC3_vc16_{sp}_{cond}.csv')
                    if os.path.exists(pc):
                        dc = pd.read_csv(pc); dc = dc[dc.success == 1] if 'success' in dc else dc
                        c4 = dc[(dc.source == 'src4_recon_valid') & (dc.ref == 'commercial')].set_index('sid')
                        j = s4.set_index('sid')[['bsp_cc', 'AT_mae', 'RT_mae']].join(c4[['bsp_cc', 'AT_mae', 'RT_mae']], lsuffix='_a', rsuffix='_c', how='inner').dropna()
                        for k, col in enumerate(('bsp_cc', 'AT_mae', 'RT_mae')):
                            d = j[col + '_a'] - j[col + '_c']
                            pv = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                            st = '^{***}' if pv < 1e-3 else '^{**}' if pv < 1e-2 else '^{*}' if pv < 0.05 else '^{\\mathrm{ns}}'
                            cell[k] += f' (${d.mean():+.3f}{st}$)' if col == 'bsp_cc' else f' (${d.mean():+.2f}{st}$)'
                            out[f'{cond}|{mid}|{sp}'][f'd_{col}'] = float(d.mean()); out[f'{cond}|{mid}|{sp}'][f'p_{col}'] = float(pv)
                cells += cell
            L.append(f"{clab if first else ''} & {mlab} & " + ' & '.join(cells) + r' \\'); first = False
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_offgrid_ab_v2.tex')), 'w').write('\n'.join(L) + '\n')
    json.dump(out, open(assert_v2(os.path.join(V2_GENERATED, 'offgrid_ab_v2.json')), 'w'), indent=1)
    for k, v in out.items():
        print(f'{k:32s} ' + ' '.join(f'{a}={b:.3f}' if isinstance(b, float) else f'{a}={b}' for a, b in v.items()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--build', action='store_true'); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if a.build:
        build()
    if a.summary:
        summary()
