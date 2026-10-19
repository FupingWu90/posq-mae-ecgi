"""Deployment emulation 'patient-specific spread placement decided from the GEOMETRY ALONE'.

The per-recording k-means protocol of the paper clusters the VALID electrodes of the recording, i.e. it
uses the knowledge of which vest channels work.  In a real few-electrode deployment the operator would
simply place N electrodes at mutually distant positions on the patient's torso.  This script emulates
that: the N positions are chosen on ALL registered electrode positions of the patient (k-means medoids, or
farthest-point sampling), without looking at the signals; a planned electrode that has no usable signal in
the retrospective recording is replaced by the nearest valid one (as for the fixed template layout).

v1 final model, N=16, both test cohorts; the v2 copy of the evaluation chain is used (parity-tested).
Output: results/diagnostics_20261004/geomonly_<method>_vc16_<split>.csv  (column n_template_sub = substitutions)
  python baselines_analyses/diag_geometry_only_selection.py --split test_internal --method kmean
"""
import os, sys, argparse, warnings
import numpy as np
import torch
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import build_sources as BS
import eval_downstream as E
from channel_select import select_visible_channels, _coord_valid
from paths import V1_RUNS

RUN = os.path.join(V1_RUNS, 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800')
OUT = os.path.join(V2, 'results', 'diagnostics_20261004')


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True, choices=['test_internal', 'test_external'])
    p.add_argument('--method', default='kmean', choices=['kmean', 'greedy'])
    p.add_argument('--vc', type=int, default=16); p.add_argument('--n', type=int, default=1000000)
    a = p.parse_args()
    orig = BS.build_context

    def build_context_geom(ecg, tor, keep_idx, args, vc=16, mode='cluster', cluster_method='greedy', seed=42, region='all'):
        ctx = orig(ecg, tor, keep_idx, args, vc=vc, mode=mode, cluster_method=cluster_method, seed=seed, region=region)
        if ctx is None:
            return None
        coords, active = ctx['coords'], ctx['active252']
        reg = _coord_valid(coords)
        plan = select_visible_channels(coords, reg, vc, 'cluster', a.method, np.random.default_rng(seed))   # geometry only
        avail = active & reg
        vis = np.zeros(len(active), bool); nsub = 0
        for c in np.where(plan)[0]:
            if avail[c] and not vis[c]:
                vis[c] = True
                continue
            d = np.linalg.norm(coords - coords[c], axis=1); d[~avail | vis] = np.inf
            j = int(np.argmin(d))
            if np.isfinite(d[j]):
                vis[j] = True; nsub += 1
        g = float(torch.from_numpy(ctx['prep252'][vis]).std(dim=1).median())
        ctx = dict(ctx)
        ctx.update(visible252=vis, g=(g if g > 1e-8 else 1.0), n_vc=int(vis.sum()), n_recon=int((active & ~vis).sum()),
                   n_template_sub=nsub, src1_vc=BS._to256(ctx['prep252'], vis, keep_idx))
        return ctx

    E.build_context = build_context_geom
    E.vc_tag = lambda *x, **k: f'vc{a.vc}_geomonly-{a.method}_NOCACHE'
    E.evaluate(RUN, split=a.split, n=a.n, vc=a.vc, mode='cluster', cluster_method='kmean', seed=42,
               out_csv=os.path.join(OUT, f'geomonly_{a.method}_vc{a.vc}_{a.split}.csv'), verbose=False)


if __name__ == '__main__':
    main()
