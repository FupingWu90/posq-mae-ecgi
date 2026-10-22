"""Build the fixed deployment layouts (v2 'template' selection mode).

For each electrode count N: k-means medoids (k = N, seed 42) of the MEAN normalised electrode positions
of the clinical TRAINING geometries (no validation/test geometry is used).  Output:
bsp_reconstruction/templates/template_vc<N>.npz  (idx: indices into the 252 kept channels; mean_xyz: (252,3)).
"""
import os, sys, json
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
import dataset as D
from channel_select import select_visible_channels
from paths import assert_v2

OUT = os.path.join(V2, 'bsp_reconstruction', 'templates')
os.makedirs(OUT, exist_ok=True)
b, _ = D.collect_split(42)
keep = np.where(D.geometry_union(b['train']))[0].astype(np.int64)
assert len(keep) == 252, len(keep)
tors = sorted({tp for _, _, tp in b['train'] if tp})
C = np.stack([D._coords_for(tp)[keep] for tp in tors])            # (n_geom, 252, 3); 0 = unregistered
ok = np.abs(C).sum(-1) > 1e-6
mean_xyz = ((C * ok[..., None]).sum(0) / np.maximum(ok.sum(0), 1)[:, None]).astype(np.float32)
nn = np.linalg.norm(mean_xyz[:, None] - mean_xyz[None], axis=-1); np.fill_diagonal(nn, 9)
spacing = float(np.median(nn.min(1)))
meta = {'n_train_geometries': len(tors), 'spacing_normalised': spacing, 'seed': 42, 'layouts': {}}
for n in (4, 8, 16, 32, 64):
    vis = select_visible_channels(mean_xyz, np.ones(len(keep), bool), n, 'cluster', 'kmean', np.random.default_rng(42))
    idx = np.where(vis)[0].astype(np.int64)
    assert len(idx) == n
    np.savez(assert_v2(os.path.join(OUT, f'template_vc{n}.npz')), idx=idx, mean_xyz=mean_xyz, keep_idx=keep)
    front = int((-mean_xyz[idx, 1] > 0).sum())
    meta['layouts'][n] = {'idx252': idx.tolist(), 'n_anterior': front, 'n_posterior': n - front}
    print(f'N={n:2d}: {n} electrodes ({front} anterior / {n - front} posterior)')
json.dump(meta, open(os.path.join(OUT, 'templates.json'), 'w'), indent=1)
print('spacing (normalised units): %.3f | geometries: %d' % (spacing, len(tors)))
