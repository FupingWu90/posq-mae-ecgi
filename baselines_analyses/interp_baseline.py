"""Non-learned interpolation baselines for the Number-of-electrodes analysis.

From the SAME N visible electrodes as the model (k-means medoids, seed 42),
reconstruct the full BSP with classical interpolation, feed the SAME MFS
inverse, and score the SAME downstream AT/RT metrics against the cached
references. Two classical methods:

  laplacian : Dirichlet problem on the torso mesh graph Laplacian
              (inverse-edge-length weights) -- the classic Oostendorp-style
              missing-lead interpolation. Unknown vertices solve
              L_uu phi_u = -L_uk phi_k with the N electrodes as boundary.
  tps       : 3-D thin-plate-spline RBF on the normalised electrode
              coordinates (scipy RBFInterpolator; falls back to a linear RBF
              if the TPS system is singular, e.g. N=4 near-coplanar).

The assembled field keeps the measured signal on the visible electrodes and
the interpolated signal on all other mesh-registered ACTIVE electrodes (same
active-mask philosophy as source 4). Metrics per recording x N x method:
AT/RT MAE (mean over reference-gated vertices) vs commercial and vs the
all-valid inverse, from the cached reference AT/RT maps (no potential-level
metrics -> no heavy fields loaded).

Usage:  python interp_baseline.py --split test_internal --shard 0 --nshards 6
Output: generated/interp/<split>_shard<k>.csv   (aggregate with vc_interp_agg.py)
"""
import argparse
import csv
import os
import sys
import warnings

import numpy as np

from common import GEN, PROJ, run_dir

sys.path.insert(0, os.path.join(PROJ, 'inverse_evaluation'))
sys.path.insert(0, os.path.join(PROJ, 'bsp_reconstruction'))
from build_sources import build_context, _to256, CROP                    # noqa: E402
from channel_select import select_visible_channels                        # noqa: E402
from inverse import assemble_solver, solve_from_field                     # noqa: E402
from at_rt import extract_sample                                          # noqa: E402
from metrics import map_mae_cc                                            # noqa: E402
from mesh_io import load_surface, find_patient_files                      # noqa: E402
from cache_references import CACHE_ROOT                                   # noqa: E402
from dataset import collect_split                                         # noqa: E402

warnings.filterwarnings('ignore')

VCS = [4, 8, 16, 24, 32, 48, 64]
OUTDIR = os.path.join(GEN, 'interp')
os.makedirs(OUTDIR, exist_ok=True)


# --------------------------------------------------------------- laplacian --
def mesh_laplacian(torso_V, torso_F):
    """Graph Laplacian with inverse-edge-length weights (dense, n_t x n_t)."""
    n = torso_V.shape[0]
    W = np.zeros((n, n))
    for tri in torso_F:
        for a in range(3):
            i, j = int(tri[a]), int(tri[(a + 1) % 3])
            w = 1.0 / (np.linalg.norm(torso_V[i] - torso_V[j]) + 1e-9)
            W[i, j] = W[j, i] = max(W[i, j], w)
    L = np.diag(W.sum(1)) - W
    return L


def laplacian_interp(L, known_idx, known_vals, n_vert):
    """Solve the Dirichlet problem; returns (n_vert, T) with knowns in place."""
    u = np.setdiff1d(np.arange(n_vert), known_idx)
    phi = np.zeros((n_vert, known_vals.shape[1]), np.float64)
    phi[known_idx] = known_vals
    A = L[np.ix_(u, u)]
    B = -L[np.ix_(u, known_idx)] @ known_vals
    phi[u] = np.linalg.solve(A + 1e-9 * np.eye(len(u)), B)
    return phi


# --------------------------------------------------------------------- tps --
def tps_interp(pts_known, vals_known, pts_eval):
    from scipy.interpolate import RBFInterpolator
    for kw in (dict(kernel='thin_plate_spline'), dict(kernel='thin_plate_spline',
                                                      smoothing=1e-6),
               dict(kernel='linear')):
        try:
            f = RBFInterpolator(pts_known, vals_known, **kw)
            return f(pts_eval)
        except Exception:
            continue
    return None


# ------------------------------------------------------------------- main ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--n', type=int, default=10**9)
    p.add_argument('--vcs', type=int, nargs='*', default=None,
                   help='restrict to these N (e.g. 72 when extending the sweep); '
                        'output goes to <split>_shard<N..>_<k>.csv')
    a = p.parse_args()
    global VCS
    vcs_tag = ''
    if a.vcs:
        VCS = list(a.vcs)
        vcs_tag = '-'.join(str(v) for v in VCS) + '_'

    import json
    rd = run_dir(model='fixedn_ecgi_wide', vc=16, ws='0', wt='3', rw=100)
    args = json.load(open(os.path.join(rd, 'args.json')))
    keep_idx = np.load(os.path.join(rd, 'keep_idx.npy')).astype(np.int64)
    pos252 = np.full(256, -1, np.int64)
    pos252[keep_idx] = np.arange(len(keep_idx))

    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets[a.split] if r[2]][:a.n][a.shard::a.nshards]
    refs_dir = os.path.join(CACHE_ROOT, a.split, 'refs')

    out_csv = os.path.join(OUTDIR, f'{a.split}_shard{vcs_tag}{a.shard}.csv')
    rows = []
    solver_cache, lap_cache = {}, {}
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        try:
            gdir = os.path.dirname(torsopath)
            if gdir not in solver_cache:
                solver_cache[gdir] = assemble_solver(gdir)
                tV, tF, tci = load_surface(find_patient_files(gdir)['torso'])
                lap_cache[gdir] = (mesh_laplacian(tV, tF), tci)
            solver, torso_ci, hV, hF = solver_cache[gdir]
            L, chidx = lap_cache[gdir]
            n_vert = L.shape[0]

            ref = np.load(os.path.join(refs_dir, sid.replace('/', '_') + '.npz'),
                          allow_pickle=False)
            at2, atc = ref['AT2'].astype(float), ref['ATcom'].astype(float)
            rt2, rtc = ref['RT2'].astype(float), ref['RTcom'].astype(float)
            q2, t2 = ref['qrs_good2'].astype(bool), ref['t_good2'].astype(bool)
            qc, tc_ = ref['qrs_goodcom'].astype(bool), ref['t_goodcom'].astype(bool)

            ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16,
                                mode='cluster', cluster_method='kmean', seed=42)
            if ctx is None:
                continue
            prep, coords, active = ctx['prep252'], ctx['coords'], ctx['active252']
            coord_ok = np.abs(coords).sum(1) > 1e-6
            # channel(252-idx) of each mesh vertex, -1 if not a kept channel
            v2c = pos252[chidx]
            c2v = np.full(len(keep_idx), -1, np.int64)
            ok = v2c >= 0
            c2v[v2c[ok]] = np.where(ok)[0]
            on_mesh = c2v >= 0

            for vc in VCS:
                visible = select_visible_channels(
                    coords, active, vc, 'cluster', 'kmean',
                    np.random.default_rng(42))
                fields = {}
                # -- mesh-Laplacian
                kn_c = np.where(visible & on_mesh)[0]
                if len(kn_c) >= 4:
                    phiV = laplacian_interp(L, c2v[kn_c],
                                            prep[kn_c].astype(np.float64), n_vert)
                    f252 = prep.copy().astype(np.float32)
                    fill = active & on_mesh & ~visible
                    f252[fill] = phiV[c2v[fill]].astype(np.float32)
                    fields['laplacian'] = (f252, active & (visible | on_mesh))
                # -- thin-plate-spline RBF
                kn = visible & coord_ok
                ev = active & coord_ok & ~visible
                if kn.sum() >= 4 and ev.sum() > 0:
                    vals = tps_interp(coords[kn].astype(np.float64),
                                      prep[kn].astype(np.float64),
                                      coords[ev].astype(np.float64))
                    if vals is not None:
                        f252 = prep.copy().astype(np.float32)
                        f252[ev] = vals.astype(np.float32)
                        fields['tps'] = (f252, active & (visible | coord_ok))

                for meth, (f252, amask) in fields.items():
                    V256, a256 = _to256(f252, amask, keep_idx)
                    phi = solve_from_field(solver, torso_ci, V256, a256)[0]
                    if phi is None or not np.all(np.isfinite(phi)):
                        continue
                    r = extract_sample(phi)
                    if r is None:
                        continue
                    AT, RT = r['AT'].astype(float), r['RT'].astype(float)
                    ARI = RT - AT
                    ari_c = rtc - atc
                    ari_2 = rt2 - at2
                    row = dict(sid=sid, vc=vc, method=meth,
                               at_com=map_mae_cc(AT, atc, qc)[0],
                               at_src2=map_mae_cc(AT, at2, q2)[0],
                               rt_com=map_mae_cc(RT, rtc, tc_)[0],
                               rt_src2=map_mae_cc(RT, rt2, t2)[0],
                               ari_com=map_mae_cc(ARI, ari_c, tc_)[0],
                               ari_src2=map_mae_cc(ARI, ari_2, t2)[0])
                    rows.append(row)
            if (i + 1) % 10 == 0:
                print(f'[{a.split} s{a.shard}] {i+1}/{len(recs)}', flush=True)
        except Exception as e:
            print(f'[{a.split} s{a.shard}] {sid[:40]} FAILED: '
                  f'{type(e).__name__}: {e}', flush=True)

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sid', 'vc', 'method', 'at_com',
                                          'at_src2', 'rt_com', 'rt_src2',
                                          'ari_com', 'ari_src2'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {out_csv} ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
    main()
