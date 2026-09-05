"""Per-vertex (position-based) downstream error, mapped into the patient-
independent epicardial grid so it can be averaged ACROSS patients.

For every test recording it computes PER-VERTEX quantities for the VC-only
baseline (src1) and the model reconstruction (src4) against both references
(src2 all-valid, commercial), bins them onto the common (theta,z) grid from
heart_coords, and saves the compact per-recording maps.

  HSP  : per-vertex temporal CC and MAE of the epicardial potential
  AT/RT/ARI : per-vertex |error| (ms), restricted to the REFERENCE-gated
              vertices (source-independent -> fair across sources)

Output is split by what the result depends on, so nothing is recomputed or
duplicated across models:

  src1 (VC-only) is MODEL-INDEPENDENT -- it depends only on the VC selection --
      -> inverse_evaluation/spatial_out/<split>/<vc_tag>/<sid>.npz   (shared)
  src4 (model reconstruction) depends on the trained model
      -> <run_dir>/spatial_maps/<split>/<vc_tag>/<sid>.npz   (per-model)

Each file carries (36,20) maps + vertex counts, the per-vertex (theta,z), and the
patient id so aggregation can go recording -> patient -> cohort.

  python spatial_maps.py --run_dir <trained run> [--n N]
"""
import os, sys, argparse, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction')
sys.path.insert(0, _CODE)

import torch
from build_sources import load_model, build_context, reconstruct_sources, CROP
from inverse import assemble_solver, solve_from_field, load_commercial
from at_rt import extract_sample, good_vertex_mask
from cache_references import CACHE_ROOT, vc_tag
from dataset import collect_split
from mesh_io import load_surface, find_patient_files
from heart_coords import vertex_coords, coords_for_geometry, bin_map, N_THETA, N_Z

# src1 (VC-only) is model-independent -> shared downstream dir;
# src4 (model recon) -> the model's own run dir (set at runtime).
SHARED_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spatial_out')
# per-PATIENT epicardial (theta,z) coords: model- and VC-independent -> written
# once per geometry and shared by every model/config.
GEOM_DIR = os.path.join(SHARED_ROOT, 'geom')
SOURCES = ['src1_vc', 'src4_recon_valid']
SRC_SHARED = 'src1_vc'                 # goes to SHARED_ROOT
SRC_MODEL = 'src4_recon_valid'         # goes to <run_dir>/spatial_maps
REFS = ['src2_allvalid', 'commercial']


def _atrt(phi):
    if phi is None or not np.all(np.isfinite(phi)):
        return None
    try:
        return extract_sample(phi)
    except Exception:
        return None


def _load_npz(p):
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _f(x):
    return None if x is None else np.asarray(x, np.float64)


def per_vertex_hsp(a, b):
    """per-vertex temporal CC and MAE between two (V,T) fields."""
    d = a - b
    mae_v = np.abs(d).mean(1)
    am = a - a.mean(1, keepdims=True); bm = b - b.mean(1, keepdims=True)
    den = np.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1))
    cc_v = np.divide((am * bm).sum(1), den, out=np.full(a.shape[0], np.nan), where=den > 1e-12)
    return cc_v, mae_v


def patient_of(sid):
    parts = sid.split('__')
    return parts[1] if len(parts) > 1 else parts[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--n', type=int, default=100000)
    p.add_argument('--vc', type=int, default=16)
    p.add_argument('--mode', default='cluster')
    p.add_argument('--cluster_method', default='kmean')   # deploy selection
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--splits', nargs='+', default=['test_internal', 'test_external'])
    a = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, keep_idx, args = load_model(a.run_dir, device)
    buckets, _ = collect_split(seed=a.seed)
    tag = vc_tag(a.vc, a.mode, a.cluster_method, a.seed)

    for split in a.splits:
        # src1 -> shared (model-independent); src4 -> this model's run dir
        dir_shared = os.path.join(SHARED_ROOT, split, tag)
        dir_model = os.path.join(a.run_dir, 'spatial_maps', split, tag)
        os.makedirs(dir_shared, exist_ok=True); os.makedirs(dir_model, exist_ok=True)
        recs = [r for r in buckets[split] if r[2]][:a.n]
        refs_dir = os.path.join(CACHE_ROOT, split, 'refs')
        vc_dir = os.path.join(CACHE_ROOT, split, tag)
        solver_cache, coord_cache = {}, {}
        done = 0
        for i, (sid, ecgpath, torsopath) in enumerate(recs):
            sid_safe = sid.replace('/', '_')
            p_shared = os.path.join(dir_shared, sid_safe + '.npz')
            p_model = os.path.join(dir_model, sid_safe + '.npz')
            need_shared, need_model = not os.path.exists(p_shared), not os.path.exists(p_model)
            if not (need_shared or need_model):
                done += 1; continue
            try:
                gdir = os.path.dirname(torsopath)
                if gdir not in solver_cache:
                    solver_cache[gdir] = assemble_solver(gdir)
                solver, tci, hV, hF = solver_cache[gdir]
                n_h = hV.shape[0]
                if gdir not in coord_cache:
                    th_, z_, _ = coords_for_geometry(gdir)   # valve-oriented apex->base
                    coord_cache[gdir] = (th_, z_)
                    # persist the shared per-geometry coords once (used by the
                    # 3D template rendering; identical for every model/config)
                    gkey = gdir.strip('/').replace('/', '_')
                    gp = os.path.join(GEOM_DIR, gkey + '.npz')
                    if not os.path.exists(gp):
                        os.makedirs(GEOM_DIR, exist_ok=True)
                        th_, z_ = coord_cache[gdir]
                        np.savez_compressed(gp, theta=th_.astype(np.float32),
                                            z=z_.astype(np.float32), n_h=n_h,
                                            geom_dir=gdir)
                theta, zc = coord_cache[gdir]

                ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=a.vc,
                                    mode=a.mode, cluster_method=a.cluster_method, seed=a.seed)
                if ctx is None:
                    continue
                tw = (ctx['crop_start'], ctx['crop_start'] + CROP)

                ref = _load_npz(os.path.join(refs_dir, sid_safe + '.npz'))
                c1 = _load_npz(os.path.join(vc_dir, sid_safe + '.npz'))
                phicom = load_commercial(os.path.splitext(ecgpath)[0] + '.potential', n_h, time_window=tw)
                phi = {'src2_allvalid': _f(ref.get('phi2')) if ref else None, 'commercial': phicom}
                if phi['src2_allvalid'] is None:
                    phi['src2_allvalid'] = solve_from_field(solver, tci, *ctx['src2_allvalid'])[0]
                if need_shared:
                    phi['src1_vc'] = _f(c1.get('phi1')) if c1 else \
                        solve_from_field(solver, tci, *ctx['src1_vc'])[0]
                if need_model:
                    recon = reconstruct_sources(model, ctx, device)
                    phi['src4_recon_valid'] = solve_from_field(solver, tci, *recon['src4_recon_valid'])[0]

                # AT/RT + reference gates
                at, gd = {}, {}
                for k, v in phi.items():
                    r = _atrt(v)
                    at[k] = r
                    if k in REFS and r is not None:
                        gd[k] = good_vertex_mask(r['sigavg'], *r['windows'])

                # one output dict per source (src1 -> shared, src4 -> model dir).
                # (theta,z) live in the shared per-geometry cache, not duplicated here.
                common = {'patient': patient_of(sid), 'n_h': n_h,
                          'geom_key': gdir.strip('/').replace('/', '_')}
                outs = {s: dict(common) for s in SOURCES}
                for s in SOURCES:
                    out = outs[s]
                    ps = phi.get(s)
                    if ps is None or not np.all(np.isfinite(ps)):
                        continue
                    for r in REFS:
                        pr = phi.get(r)
                        if pr is None or not np.all(np.isfinite(pr)):
                            continue
                        cc_v, mae_v = per_vertex_hsp(ps, pr)
                        for nm, vals, msk in [('hspcc', cc_v, None), ('hspmae', mae_v, None)]:
                            m, c = bin_map(theta, zc, vals, msk)
                            out[f'{s}__{r}__{nm}'] = m.astype(np.float32)
                            out[f'{s}__{r}__{nm}__cnt'] = c.astype(np.int32)
                            # RAW per-vertex values too: lets us re-bin at another
                            # resolution, re-gate, or use a different cross-patient
                            # correspondence (template registration / AHA) later
                            # without re-running the inverse.
                            out[f'{s}__{r}__{nm}__v'] = vals.astype(np.float16)
                        a_s, a_r, g = at.get(s), at.get(r), gd.get(r)
                        if a_s is None or a_r is None or g is None:
                            continue
                        dAT = np.abs(a_s['AT'].astype(float) - a_r['AT'].astype(float))
                        dRT = np.abs(a_s['RT'].astype(float) - a_r['RT'].astype(float))
                        dARI = np.abs((a_s['RT'] - a_s['AT']).astype(float)
                                      - (a_r['RT'] - a_r['AT']).astype(float))
                        for nm, vals, msk in [('AT', dAT, g['qrs']), ('RT', dRT, g['t']),
                                              ('ARI', dARI, g['t'])]:
                            # (a) REFERENCE-GATED map (reliable-beat vertices only)
                            m, c = bin_map(theta, zc, vals, msk)
                            out[f'{s}__{r}__{nm}'] = m.astype(np.float32)
                            out[f'{s}__{r}__{nm}__cnt'] = c.astype(np.int32)
                            # raw per-vertex values are un-gated (full length n_h)
                            out[f'{s}__{r}__{nm}__v'] = vals.astype(np.float16)
                            # (b) UN-GATED map over ALL vertices. The T-gate leaves
                            # RT/ARI very sparse (~40/720 bins on many recordings),
                            # so the dense un-gated map is kept alongside it.
                            m2, c2 = bin_map(theta, zc, vals, None)
                            out[f'{s}__{r}__{nm}_allv'] = m2.astype(np.float32)
                            out[f'{s}__{r}__{nm}_allv__cnt'] = c2.astype(np.int32)
                        # the reference-gated vertex masks (source-independent):
                        # needed to re-apply/relax the gating on the raw per-vertex values
                        out[f'{r}__qrs_good'] = g['qrs'].astype(bool)
                        out[f'{r}__t_good'] = g['t'].astype(bool)
                if need_shared and len(outs[SRC_SHARED]) > 3:
                    np.savez_compressed(p_shared, **outs[SRC_SHARED])
                if need_model and len(outs[SRC_MODEL]) > 3:
                    np.savez_compressed(p_model, **outs[SRC_MODEL])
                done += 1
                if done % 25 == 0:
                    print(f"  [{split}] {done}/{len(recs)}", flush=True)
            except Exception as e:
                print(f"  [{split}] {sid[:40]} FAILED: {type(e).__name__}: {e}", flush=True)
        print(f"[{split}] wrote {done} recordings"
              f"  src1 -> {dir_shared}  src4 -> {dir_model}", flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
