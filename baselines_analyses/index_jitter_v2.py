"""Index-slot position jitter vs the per-recording advantage of the physical position code (main text,
"Physical positions versus channel indices"; v2 re-computation of the v1 ad-hoc analysis, 2026-10-13).

For every test recording with a registered geometry: the normalised position of each registered electrode is
compared with the population-mean position of the same channel index (bsp_reconstruction/templates/template_vc16.npz,
'mean_xyz' = mean over the training geometries, the quantity an index embedding can at best encode), in units
of the recording's own electrode spacing (median nearest-neighbour distance).  Per recording we take the mean
jitter over (a) all registered active electrodes, (b) the visible electrodes of the layout and (c) the held-out
(completed) electrodes, and correlate it (Spearman) with the paired per-recording difference
positions - index (LOC3 - P3IDX42 by default; same electrodes, same recordings) in AT MAE, RT MAE, HSP CC and
BSP CC taken from the per-recording CSVs of baselines_analyses/generated/percase_candidate/.
  python baselines_analyses/index_jitter_v2.py [--ppe LOC3] [--index P3IDX42] [--layouts template kmeans]
Out: baselines_analyses/generated/index_jitter_v2.json.  Prints aggregates only (no recording IDs)."""
import os, sys, json, glob, argparse, warnings
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import numpy as np, pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from paths import V2_GENERATED, V2_RUNS, assert_v2
from build_sources import load_model, build_context
from channel_select import select_visible_channels, load_template, _coord_valid
from dataset import collect_split

warnings.filterwarnings('ignore')
PC = os.path.join(V2_GENERATED, 'percase_candidate')
OUT = os.path.join(V2_GENERATED, 'index_jitter_v2.json')
VC = 16
METRICS = ('AT_mae', 'RT_mae', 'hsp_cc', 'bsp_cc')


def run_dir(pid):
    c = sorted(glob.glob(os.path.join(V2_RUNS, 'data-cohortA+cohortB_vs_external', f'{pid}__*')))
    return c[0] if c else None


def percase(pid, split, layout):
    f = os.path.join(PC, f'{pid}_vc{VC}_{split}_{layout}.csv')
    d = pd.read_csv(f)
    d = d[(d.source == 'src4_recon_valid') & (d.ref == 'commercial')].set_index('sid')
    return d[list(METRICS)].apply(pd.to_numeric, errors='coerce')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppe', default='LOC3'); ap.add_argument('--index', default='P3IDX42')
    ap.add_argument('--layouts', nargs='*', default=['template', 'kmeans'])
    a = ap.parse_args()
    _, keep_idx, args0 = load_model(run_dir(a.ppe), torch.device('cpu'))
    _, mean_xyz = load_template(VC)
    buckets, _ = collect_split(seed=42)
    res = {'config': {'ppe': a.ppe, 'index': a.index, 'vc': VC, 'layouts': a.layouts}}
    for split in ('test_internal', 'test_external'):
        recs = [r for r in buckets[split] if r[2]]
        rows, pooled = {}, []
        for sid, ecg, torso in recs:
            ctx = build_context(ecg, torso, keep_idx, args0, vc=VC, mode='template', cluster_method='greedy', seed=42)
            if ctx is None:
                continue
            coords, active = ctx['coords'], ctx['active252'].astype(bool)
            reg = _coord_valid(coords)
            spacing = float(np.median(cKDTree(coords[reg]).query(coords[reg], k=2)[0][:, 1]))
            jit = np.linalg.norm(coords - mean_xyz, axis=1) / spacing
            ok = reg & active
            pooled.extend(jit[ok].tolist())
            vis = {'template': ctx['visible252'].astype(bool)}
            if 'kmeans' in a.layouts:
                vis['kmeans'] = select_visible_channels(coords, active, VC, 'cluster', 'kmean',
                                                        np.random.default_rng(42)).astype(bool)
            r = {'jit_all': float(jit[ok].mean()), 'spacing': spacing}
            for lay, v in vis.items():
                r[f'jit_vis_{lay}'] = float(jit[v & reg].mean())
                r[f'jit_held_{lay}'] = float(jit[ok & ~v].mean())
            rows[sid] = r
        J = pd.DataFrame(rows).T
        pooled = np.asarray(pooled)
        S = {'n_recordings': int(len(J)),
             'jitter_pooled_median_spacings': float(np.median(pooled)),
             'jitter_pooled_mean_spacings': float(pooled.mean()),
             'jitter_pooled_p90_spacings': float(np.percentile(pooled, 90)),
             'jitter_per_recording_mean_spacings': float(J['jit_all'].mean()),
             'jitter_per_recording_range': [float(J['jit_all'].min()), float(J['jit_all'].max())]}
        print(f'--- {split}: n={len(J)}; per-electrode jitter of the channel index vs its population-mean position: '
              f'median {S["jitter_pooled_median_spacings"]:.2f}, mean {S["jitter_pooled_mean_spacings"]:.2f}, '
              f'p90 {S["jitter_pooled_p90_spacings"]:.2f} spacings; per-recording mean {S["jitter_per_recording_mean_spacings"]:.2f} '
              f'(range {S["jitter_per_recording_range"][0]:.2f}-{S["jitter_per_recording_range"][1]:.2f})')
        for lay in a.layouts:
            P, I = percase(a.ppe, split, lay), percase(a.index, split, lay)
            c = P.index.intersection(I.index).intersection(J.index)
            D = (P.loc[c] - I.loc[c])
            S[lay] = {'n_paired': int(len(c))}
            for jcol in ('jit_all', f'jit_vis_{lay}', f'jit_held_{lay}'):
                x = J.loc[c, jcol].astype(float)
                for m in METRICS:
                    d = D[m].astype(float); ok = x.notna() & d.notna()
                    rho, p = spearmanr(x[ok], d[ok])
                    S[lay][f'{jcol}|{m}'] = {'rho': float(rho), 'p': float(p), 'n': int(ok.sum())}
                print(f'  {lay:8s} {jcol:18s} ' + ' | '.join(
                    f'{m}: rho={S[lay][f"{jcol}|{m}"]["rho"]:+.3f} p={S[lay][f"{jcol}|{m}"]["p"]:.2g}' for m in METRICS))
        res[split] = S
    json.dump(res, open(assert_v2(OUT), 'w'), indent=1)
    print('wrote', os.path.relpath(OUT, V2))


if __name__ == '__main__':
    main()
