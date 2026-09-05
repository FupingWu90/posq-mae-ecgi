"""Dense-field decoding validation: does super-sampling the reconstructed
boundary field improve the downstream MFS inverse?

The PPE-MAE is queryable at ARBITRARY torso positions because channel identity
enters the network ONLY through the physical position embedding PPE(coords)
(channel slots are anonymous: the patch conv, mask token, decoder head and
attention are all shared / permutation-equivariant across channel rows).  We
exploit this to decode the reconstructed body-surface field not just at the 252
vest electrodes but at Q in {512, 1024} approximately-uniform points on the
torso mesh (midpoint-subdivided twice, farthest-point sampled), then feed the
densified boundary field through the SAME MFS inverse (Dirichlet+Neumann rows
at the Q points, same alpha_in/alpha_out/CAR/SVD-L-curve code path) and compare
epicardial AT/RT against the commercial reference.

Conditions (all from the SAME 16 k-means-selected electrodes, wide vc16 run):
  model252   standard path: src4_recon_valid -> solve_from_field  (baseline)
  model512   model field decoded at 512 dense points  -> dense inverse
  model1024  model field decoded at 1024 dense points -> dense inverse
  tps512     thin-plate-spline fit on the same 16 electrodes -> 512 points
  tps1024    idem at 1024 points

Lambda fairness: the L-curve floor is re-calibrated per (method x density) on
~12 validation recordings (grid {1e-3,3e-3,1e-2,3e-2}), minimising AT+RT MAE
vs commercial — exactly the lam_recal.py protocol.  model252 keeps the
pipeline-standard floor (calibration.json, 3e-3).

  --sanity              slot-anonymity numeric check: query the decoder at the
                        TRUE positions of the held-out electrodes and compare
                        with the standard per-slot reconstruction
                        -> generated/dense_field/sanity.json
  --calibrate           per-condition lam_floor -> generated/dense_field_lam.json
  --eval --split S [--n 80 --shard k --nshards m]
                        per-recording rows (idempotent cache)
                        -> generated/dense_field/cache/<split>/<sid>.json
                        -> generated/dense_field/<split>.csv
  --summary             cohort means + paired Wilcoxon vs model252
                        -> generated/dense_field.json  (+ printed table)
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import torch

from common import GEN, PROJ, run_dir

sys.path.insert(0, os.path.join(PROJ, 'inverse_evaluation'))
sys.path.insert(0, os.path.join(PROJ, 'bsp_reconstruction'))
from build_sources import (build_context, load_model, reconstruct_sources,  # noqa: E402
                           CROP)
from inverse import (ALPHA_IN, ALPHA_OUT, LAM_CEIL, LAM_N,                  # noqa: E402
                     assemble_solver, solve_from_field)
from mfs import MFSSolver                                                   # noqa: E402
from mesh_io import load_surface, vertex_normals, find_patient_files        # noqa: E402
from torso_loader import load_torso_coords                                  # noqa: E402
from at_rt import extract_sample                                            # noqa: E402
from metrics import map_mae_cc                                              # noqa: E402
from cache_references import CACHE_ROOT                                     # noqa: E402
from dataset import collect_split                                           # noqa: E402

warnings.filterwarnings('ignore')

QS = [512, 1024]
FLOORS = [1e-3, 3e-3, 1e-2, 3e-2]
CONDS = ['model252', 'model512', 'model1024', 'tps512', 'tps1024']
VER = 1                                        # bump to invalidate row caches

OUTD = os.path.join(GEN, 'dense_field')
CACHED = os.path.join(OUTD, 'cache')
LAMJ = os.path.join(GEN, 'dense_field_lam.json')
SUMJ = os.path.join(GEN, 'dense_field.json')
SANJ = os.path.join(OUTD, 'sanity.json')
os.makedirs(OUTD, exist_ok=True)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def _setup():
    rd = run_dir(model='fixedn_ecgi_wide', vc=16, ws='0', wt='3', rw=100)
    args = json.load(open(os.path.join(rd, 'args.json')))
    keep_idx = np.load(os.path.join(rd, 'keep_idx.npy')).astype(np.int64)
    assert args.get('pp_norm', 'centroid') == 'centroid', 'expects centroid norm'
    return rd, args, keep_idx


def _ctx(rec, keep_idx, args):
    """Deployment context: k-means 16-electrode selection, seed 42 (as every
    other manuscript baseline)."""
    sid, ecgpath, torsopath = rec
    return build_context(ecgpath, torsopath, keep_idx, args, vc=16,
                         mode='cluster', cluster_method='kmean', seed=42)


def _ref(split, sid):
    p = os.path.join(CACHE_ROOT, split, 'refs', sid.replace('/', '_') + '.npz')
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


class LRU(OrderedDict):
    """Tiny LRU cache for the big per-geometry solver objects."""
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        v = fn()
        self[key] = v
        while len(self) > self.maxsize:
            self.popitem(last=False)
        return v


# --------------------------------------------------------------------------- #
# Dense torso points: midpoint subdivision x2 + farthest-point sampling
# --------------------------------------------------------------------------- #

def _subdivide(V, F):
    """One round of midpoint subdivision (geometry unchanged: new vertices lie
    on the original triangle planes; orientation preserved)."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    edge_mid = {}
    new_pts = []
    nv = len(V)

    def mid(a, b):
        nonlocal nv
        key = (a, b) if a < b else (b, a)
        m = edge_mid.get(key)
        if m is None:
            new_pts.append((V[a] + V[b]) * 0.5)
            m = nv
            nv += 1
            edge_mid[key] = m
        return m

    F2 = []
    for a, b, c in F:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        F2 += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    V2 = np.vstack([V, np.asarray(new_pts)]) if new_pts else V
    return V2, np.asarray(F2, np.int64)


def _fps(cand, seed_pts, k):
    """Greedy farthest-point sampling of k points from `cand`, initialised with
    the min-distance field to `seed_pts` (deterministic)."""
    dmin = np.min(np.linalg.norm(cand[:, None, :] - seed_pts[None, :, :],
                                 axis=2), axis=1)
    picks = np.empty(k, np.int64)
    for i in range(k):
        j = int(np.argmax(dmin))
        picks[i] = j
        dmin = np.minimum(dmin, np.linalg.norm(cand - cand[j], axis=1))
    return picks


def dense_geometry(torsopath, vis_ch):
    """Q_max dense torso points for one recording.

    Row order: [the n_e visible-electrode vertices (in ascending channel
    order)] + [FPS picks seeded by those electrodes].  Because FPS is greedy,
    the first Q rows are exactly the Q-point dense set (prefix property), so
    one array serves both Q=512 and Q=1024.

    Returns dict: dense_V (Qmax,3) raw cm, dense_N (Qmax,3) outward unit
    normals (from the subdivided mesh), heart_V, n_elec, outward_frac.
    """
    gdir = os.path.dirname(torsopath)
    files = find_patient_files(gdir)
    tV, tF, tci = load_surface(files['torso'])
    hV, _hF, _ = load_surface(files['heart'])
    if tci is None:
        raise ValueError(f'{files["torso"]}: missing ChannelIndices')
    V2, F2 = _subdivide(*_subdivide(tV, tF))
    if len(V2) < max(QS):
        V2, F2 = _subdivide(V2, F2)
    N2 = vertex_normals(V2, F2).astype(np.float64)
    ch2v = {int(c): i for i, c in enumerate(tci)}
    vidx = np.asarray([ch2v[int(c)] for c in vis_ch], np.int64)
    picks = _fps(V2, V2[vidx], max(QS) - len(vidx))
    idx = np.concatenate([vidx, picks])
    dV, dN = V2[idx], N2[idx]
    # Orientation bookkeeping only: Neumann rows have zero RHS, so a sign flip
    # of any normal leaves the least-squares problem unchanged.
    outward = float(np.mean(np.einsum('ij,ij->i', dN, dV - tV.mean(0)) > 0))
    return dict(dense_V=dV, dense_N=dN, heart_V=hV.astype(np.float64),
                n_elec=len(vidx), outward_frac=outward)


def _norm_stats(torsopath):
    """The per-patient electrode centroid + mean electrode-to-centroid distance
    used by torso_loader.normalize_coords(pp_norm='centroid') — recomputed so
    the SAME statistics normalise arbitrary mesh points."""
    raw = load_torso_coords(torsopath)                 # (256,3), NaN = no pos
    valid = ~np.isnan(raw[:, 0])
    v = raw[valid].astype(np.float64)
    centroid = v.mean(axis=0)
    scale = float(np.sqrt(((v - centroid) ** 2).sum(axis=1)).mean())
    return centroid, (scale if scale > 1e-8 else 1.0)


# --------------------------------------------------------------------------- #
# Dense decoding: query the decoder at arbitrary normalised positions
# --------------------------------------------------------------------------- #

@torch.no_grad()
def decode_at_points(model, ctx, q_norm, device, chunk=None, seed=0):
    """Decode the model field at arbitrary normalised torso positions.

    Each forward pass keeps the 16 visible electrodes at their true slots
    (true coords + prep/g signal) and overwrites the coords of the ~236
    non-visible slots with query positions (zero signal, masked status — the
    encoder never sees these tokens; the decoder fills each with the shared
    mask token + PPE(query coords)).  A short final pass is padded with other
    query points (all coords in a pass stay distinct real torso positions);
    padded predictions are discarded.

    q_norm : (Nq, 3) float32   Returns (Nq, CROP) float32, PHYSICAL units.
    """
    coords0 = ctx['coords']
    vis = ctx['visible252']
    prep = ctx['prep252']
    g = ctx['g']
    hold = np.where(~vis)[0]                       # slots available for queries
    n_slot = len(hold) if chunk is None else min(int(chunk), len(hold))
    Nq = len(q_norm)
    out = np.empty((Nq, CROP), np.float32)

    x_base = np.where(vis[:, None], prep / g, 0.0).astype(np.float32)
    x_t = torch.from_numpy(x_base).unsqueeze(0).to(device)
    vis_t = torch.from_numpy(vis).unsqueeze(0).to(device)

    for s0 in range(0, Nq, n_slot):
        idx = np.arange(s0, min(s0 + n_slot, Nq))
        pad_needed = n_slot - len(idx)
        if pad_needed > 0:
            pool = np.setdiff1d(np.arange(Nq), idx)
            pad = (pool[:pad_needed] if len(pool) >= pad_needed
                   else np.resize(np.concatenate([pool, idx]), pad_needed))
            qsel = np.concatenate([idx, pad])
        else:
            qsel = idx
        coords = coords0.copy()
        coords[hold[:len(qsel)]] = q_norm[qsel]
        torch.manual_seed(seed)                     # fixed shuffle noise
        c_t = torch.from_numpy(coords).unsqueeze(0).to(device)
        _, pred = model.forward_fixedn(x_t, vis_t.clone(), c_t, vis_t,
                                       w_spatial=0.0, w_temporal=0.0)
        rec = model.unpatchify(pred).squeeze(0).squeeze(0).cpu().numpy()
        out[idx] = rec[hold[:len(idx)]] * g
    return out


def tps_field(pts_e, sig_e, dense_raw):
    """Thin-plate-spline control: RBF fitted on the same visible electrodes
    (raw 3D positions, physical signals), evaluated at the dense points."""
    from scipy.interpolate import RBFInterpolator
    try:
        it = RBFInterpolator(pts_e, sig_e, kernel='thin_plate_spline')
        out, kind = it(dense_raw), 'tps'
    except Exception:
        it = RBFInterpolator(pts_e, sig_e, kernel='linear')
        out, kind = it(dense_raw), 'linear'
    return out.astype(np.float32), kind


# --------------------------------------------------------------------------- #
# Dense inverse: same kernel / alphas / CAR / SVD-L-curve path as inverse.py
# --------------------------------------------------------------------------- #

def dense_solver(heart_V, dV, dN):
    """MFS system with Dirichlet+Neumann rows at the dense points.  Same
    convention as assemble_solver: inner sources = heart vertices deflated by
    ALPHA_IN about the heart centroid, outer sources = the boundary (dense)
    points inflated by ALPHA_OUT about their centroid."""
    return MFSSolver.build(heart_V, dV, dN, alpha_in=ALPHA_IN, alpha_out=ALPHA_OUT)


def solve_dense(sysA, lam_floor):
    """L-curve + reconstruct on a pre-factorised dense system (mirrors
    solve_from_field's lam=None branch, incl. the CAR of phi)."""
    lam, _ = sysA.l_curve(lams=np.logspace(np.log10(lam_floor),
                                           np.log10(LAM_CEIL), LAM_N))
    lam = float(max(lam, lam_floor))
    phi, _c, _res = sysA.reconstruct(lam)
    phi = phi - phi.mean(axis=0, keepdims=True)
    return phi, lam


def factorize_dense(solver, field):
    """CAR across ALL dense boundary points (they are all 'active'), then
    pre-factor once so several lam floors reuse one SVD."""
    V = field.astype(np.float64)
    V = V - V.mean(axis=0, keepdims=True)
    return solver.factorize(V, active_mask=np.ones(V.shape[0], bool))


def _atrt(phi, ref):
    """AT/RT MAE + spatial CC vs the commercial reference on its gates."""
    if phi is None or not np.all(np.isfinite(phi)):
        return None
    r = extract_sample(phi)
    if r is None:
        return None
    AT, RT = r['AT'].astype(float), r['RT'].astype(float)
    at, atcc = map_mae_cc(AT, ref['ATcom'].astype(float),
                          ref['qrs_goodcom'].astype(bool))
    rt, rtcc = map_mae_cc(RT, ref['RTcom'].astype(float),
                          ref['t_goodcom'].astype(bool))
    if not (np.isfinite(at) and np.isfinite(rt)):
        return None
    return dict(at_mae=float(at), rt_mae=float(rt),
                at_cc=float(atcc), rt_cc=float(rtcc))


# --------------------------------------------------------------------------- #
# Per-recording dense pipeline (shared by calibrate / eval)
# --------------------------------------------------------------------------- #

def build_dense_fields(model, ctx, keep_idx, torsopath, device, caches):
    """Everything model/TPS-side for one recording: dense points + both dense
    fields at Q_max, physical units, real signal on the electrode rows.
    Returns dict or None."""
    vis_slots = np.where(ctx['visible252'])[0]
    vis_ch = keep_idx[vis_slots]
    key = (os.path.dirname(torsopath), tuple(int(c) for c in vis_ch))
    geo = caches['geo'].get_or(key, lambda: dense_geometry(torsopath, vis_ch))
    n_e = geo['n_elec']
    centroid, scale = _norm_stats(torsopath)

    # Same normalisation statistics as the electrode coords (sanity-checked).
    q_norm_all = ((geo['dense_V'] - centroid) / scale).astype(np.float32)
    coord_err = float(np.abs(q_norm_all[:n_e] - ctx['coords'][vis_slots]).max())

    sig_e = ctx['prep252'][vis_slots]                       # (n_e, T) physical

    fields = {}
    for Q in QS:
        pred = decode_at_points(model, ctx, q_norm_all[n_e:Q], device)
        f = np.empty((Q, CROP), np.float32)
        f[:n_e] = sig_e
        f[n_e:] = pred
        fields[f'model{Q}'] = f
    tps_all, tps_kind = tps_field(geo['dense_V'][:n_e], sig_e,
                                  geo['dense_V'][:max(QS)])
    tps_all[:n_e] = sig_e
    for Q in QS:
        fields[f'tps{Q}'] = tps_all[:Q]
    return dict(geo=geo, key=key, fields=fields, n_elec=n_e,
                coord_err=coord_err, tps_kind=tps_kind)


def get_dense_solver(caches, key, Q, geo):
    return caches['dsolver'].get_or(
        key + (Q,),
        lambda: dense_solver(geo['heart_V'], geo['dense_V'][:Q],
                             geo['dense_N'][:Q]))


def _recs_for(split, n, seed=42):
    buckets, _ = collect_split(seed=seed)
    recs = [r for r in buckets[split] if r[2]]
    base = [recs[i] for i in np.linspace(0, len(recs) - 1, min(80, len(recs))).astype(int)]
    if n and n < len(base):                       # keep subset property on re-runs
        step = int(np.ceil(len(base) / n))
        base = base[::step]
    return base


# --------------------------------------------------------------------------- #
# --sanity : slot-anonymity numeric check
# --------------------------------------------------------------------------- #

def sanity(n_recs=2, device='cuda'):
    rd, args, keep_idx = _setup()
    model, keep_idx, args = load_model(rd, device)
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets['val'] if r[2]]
    out = []
    for rec in recs[:n_recs]:
        sid, ecgpath, torsopath = rec
        ctx = _ctx(rec, keep_idx, args)
        if ctx is None:
            continue
        hold = np.where(~ctx['visible252'])[0]
        # standard per-slot reconstruction (the exact deployed path)
        srcs = reconstruct_sources(model, ctx, device)
        std = srcs['src3_recon_whole'][0][keep_idx][hold]        # physical
        # A) query the decoder at the held-out electrodes' TRUE positions:
        #    one pass, identical coords array to the standard forward.
        qA = decode_at_points(model, ctx, ctx['coords'][hold], device)
        # B) same queries in a PERMUTED slot order (coords set identical, slot
        #    assignment different) — the direct numeric test of slot anonymity.
        perm = np.random.default_rng(0).permutation(len(hold))
        qB = decode_at_points(model, ctx, ctx['coords'][hold][perm], device)
        qB = qB[np.argsort(perm)]                        # back to hold order
        # C) only HALF the held-out coords queried, rest of the slots padded
        #    with duplicates (the padded-pass regime the dense decode hits in
        #    its final pass) — pass-composition sensitivity.
        half = hold[:len(hold) // 2]
        qC = decode_at_points(model, ctx, ctx['coords'][half], device)
        stdC = std[:len(half)]
        ref_std = float(np.abs(std).std())
        row = dict(
            sid=sid,
            n_holdout=int(len(hold)),
            field_std=ref_std,
            exact_max_abs=float(np.abs(qA - std).max()),
            exact_mean_abs=float(np.abs(qA - std).mean()),
            perm_max_abs=float(np.abs(qB - std).max()),
            perm_mean_abs=float(np.abs(qB - std).mean()),
            padded_max_abs=float(np.abs(qC - stdC).max()),
            padded_mean_abs=float(np.abs(qC - stdC).mean()),
        )
        out.append(row)
        print(f"[sanity] {sid[:48]}  field_std={ref_std:.4f}\n"
              f"         exact  max/mean |d|={row['exact_max_abs']:.2e}/"
              f"{row['exact_mean_abs']:.2e}\n"
              f"         perm   max/mean |d|={row['perm_max_abs']:.2e}/"
              f"{row['perm_mean_abs']:.2e}\n"
              f"         padded max/mean |d|={row['padded_max_abs']:.2e}/"
              f"{row['padded_mean_abs']:.2e}", flush=True)
        # coord-normalisation consistency + normals orientation
        caches = dict(geo=LRU(2), dsolver=LRU(2))
        dd = build_dense_fields(model, ctx, keep_idx, torsopath, device, caches)
        row['coord_norm_max_abs'] = dd['coord_err']
        row['normals_outward_frac'] = dd['geo']['outward_frac']
        print(f"         coord-normalisation max|d|={dd['coord_err']:.2e}  "
              f"normals outward frac={dd['geo']['outward_frac']:.3f}", flush=True)
    json.dump(out, open(SANJ, 'w'), indent=2)
    print('wrote', SANJ)


# --------------------------------------------------------------------------- #
# --calibrate : per-(method x density) lambda floor on validation recordings
# --------------------------------------------------------------------------- #

def calibrate(n_recs=12, device='cuda'):
    rd, args, keep_idx = _setup()
    model, keep_idx, args = load_model(rd, device)
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets['val'] if r[2]]
    recs = [recs[i] for i in np.linspace(0, len(recs) - 1, n_recs).astype(int)]

    conds = [f'{m}{q}' for m in ('model', 'tps') for q in QS]
    acc = {(c, lf): [] for c in conds for lf in FLOORS}
    caches = dict(geo=LRU(2), dsolver=LRU(2))
    for i, rec in enumerate(recs):
        sid, ecgpath, torsopath = rec
        try:
            ref = _ref('val', sid)
            if ref is None or 'ATcom' not in ref.files:
                continue
            ctx = _ctx(rec, keep_idx, args)
            if ctx is None:
                continue
            dd = build_dense_fields(model, ctx, keep_idx, torsopath, device, caches)
            for c in conds:
                Q = int(c.replace('model', '').replace('tps', ''))
                solver = get_dense_solver(caches, dd['key'], Q, dd['geo'])
                sysA = factorize_dense(solver, dd['fields'][c])
                seen = {}
                for lf in FLOORS:
                    lam, _ = sysA.l_curve(lams=np.logspace(
                        np.log10(lf), np.log10(LAM_CEIL), LAM_N))
                    lam = float(max(lam, lf))
                    if lam not in seen:
                        phi, _c2, _r = sysA.reconstruct(lam)
                        phi = phi - phi.mean(axis=0, keepdims=True)
                        seen[lam] = _atrt(phi, ref)
                    m = seen[lam]
                    if m is not None:
                        acc[(c, lf)].append(m['at_mae'] + m['rt_mae'])
            print(f'[cal] {i+1}/{len(recs)} {sid[:40]}', flush=True)
        except Exception as e:
            print(f'[cal] {i+1}/{len(recs)} {sid[:40]} FAILED: '
                  f'{type(e).__name__}: {e}', flush=True)

    best = {}
    for c in conds:
        scores = {lf: float(np.mean(acc[(c, lf)])) for lf in FLOORS
                  if acc[(c, lf)]}
        lf = min(scores, key=scores.get)
        best[c] = dict(lam_floor=lf, val_at_plus_rt=scores[lf],
                       all_scores={f'{k:g}': v for k, v in scores.items()},
                       n=len(acc[(c, lf)]))
        print(f'{c:10s}: best floor {lf:g}  '
              + '  '.join(f'{k:g}:{v:.1f}' for k, v in sorted(scores.items())))
    json.dump(best, open(LAMJ, 'w'), indent=2)
    print('wrote', LAMJ)


# --------------------------------------------------------------------------- #
# --eval : per-recording rows with idempotent caching
# --------------------------------------------------------------------------- #

def evaluate(split, n=80, shard=0, nshards=1, device='cuda'):
    rd, args, keep_idx = _setup()
    best = json.load(open(LAMJ))
    floors = {c: best[c]['lam_floor'] for c in best}
    model = None                                  # lazy: cache may cover all
    cdir = os.path.join(CACHED, split)
    os.makedirs(cdir, exist_ok=True)
    recs = _recs_for(split, n)[shard::nshards]

    caches = dict(geo=LRU(2), dsolver=LRU(2), std=LRU(3))
    for i, rec in enumerate(recs):
        sid, ecgpath, torsopath = rec
        cpath = os.path.join(cdir, sid.replace('/', '_') + '.json')
        if os.path.exists(cpath):
            try:
                d = json.load(open(cpath))
                if d.get('ver') == VER and d.get('floors') == floors and (
                        'skip' in d or all(c in d.get('conds', {}) for c in CONDS)):
                    continue
            except Exception:
                pass
        t0 = time.time()
        row = dict(sid=sid, ver=VER, floors=floors, conds={})
        try:
            ref = _ref(split, sid)
            if ref is None or 'ATcom' not in ref.files:
                row['skip'] = 'no commercial reference'
                json.dump(row, open(cpath, 'w'))
                continue
            if model is None:
                model, keep_idx, args = load_model(rd, device)
            ctx = _ctx(rec, keep_idx, args)
            if ctx is None:
                row['skip'] = 'no context'
                json.dump(row, open(cpath, 'w'))
                continue

            # ---- model252: the untouched standard path -------------------
            gdir = os.path.dirname(torsopath)
            solver, tci, hV, hF = caches['std'].get_or(
                gdir, lambda: assemble_solver(gdir))
            srcs = reconstruct_sources(model, ctx, device)
            phi, lam, _r, _m, _na = solve_from_field(solver, tci,
                                                     *srcs['src4_recon_valid'])
            m = _atrt(phi, ref)
            row['conds']['model252'] = (dict(m, lam=lam) if m else None)

            # ---- dense conditions ---------------------------------------
            dd = build_dense_fields(model, ctx, keep_idx, torsopath, device,
                                    caches)
            row['n_elec'] = dd['n_elec']
            row['tps_kind'] = dd['tps_kind']
            row['coord_err'] = dd['coord_err']
            for c in CONDS:
                if c == 'model252':
                    continue
                Q = int(c.replace('model', '').replace('tps', ''))
                dsolver = get_dense_solver(caches, dd['key'], Q, dd['geo'])
                sysA = factorize_dense(dsolver, dd['fields'][c])
                phi, lam = solve_dense(sysA, floors[c])
                m = _atrt(phi, ref)
                row['conds'][c] = (dict(m, lam=lam) if m else None)
        except Exception as e:
            row['skip'] = f'{type(e).__name__}: {e}'
        json.dump(row, open(cpath, 'w'))
        print(f'[{split} s{shard}] {i+1}/{len(recs)} {sid[:40]} '
              f'({time.time()-t0:.0f}s)', flush=True)
    write_csv(split, n)


def write_csv(split, n=80):
    """Assemble the per-recording CSV from the cache (all shards)."""
    cdir = os.path.join(CACHED, split)
    rows = []
    for rec in _recs_for(split, n):
        sid = rec[0]
        cpath = os.path.join(cdir, sid.replace('/', '_') + '.json')
        if not os.path.exists(cpath):
            continue
        d = json.load(open(cpath))
        for c in CONDS:
            m = d.get('conds', {}).get(c)
            if m is None:
                continue
            rows.append(dict(sid=sid, condition=c,
                             at_mae=round(m['at_mae'], 4),
                             rt_mae=round(m['rt_mae'], 4),
                             at_cc=round(m['at_cc'], 4),
                             rt_cc=round(m['rt_cc'], 4),
                             lam=(round(m['lam'], 6) if m.get('lam') else '')))
    out = os.path.join(OUTD, f'{split}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sid', 'condition', 'at_mae',
                                          'rt_mae', 'at_cc', 'rt_cc', 'lam'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {out} ({len(rows)} rows)', flush=True)


# --------------------------------------------------------------------------- #
# --summary : cohort means + paired Wilcoxon vs model252
# --------------------------------------------------------------------------- #

def summary(n=80):
    from scipy.stats import wilcoxon
    res = {'config': dict(qs=QS, floors_grid=FLOORS, ver=VER,
                          alpha_in=ALPHA_IN, alpha_out=ALPHA_OUT)}
    if os.path.exists(LAMJ):
        res['lam_floors'] = json.load(open(LAMJ))
    if os.path.exists(SANJ):
        res['sanity'] = json.load(open(SANJ))

    for split in ('test_internal', 'test_external'):
        cdir = os.path.join(CACHED, split)
        if not os.path.isdir(cdir):
            continue
        per = {c: {} for c in CONDS}          # cond -> sid -> (at, rt)
        for rec in _recs_for(split, n):
            sid = rec[0]
            cpath = os.path.join(cdir, sid.replace('/', '_') + '.json')
            if not os.path.exists(cpath):
                continue
            d = json.load(open(cpath))
            for c in CONDS:
                m = d.get('conds', {}).get(c)
                if m is not None:
                    per[c][sid] = (m['at_mae'], m['rt_mae'])
        tab = {}
        for c in CONDS:
            sids = sorted(per[c])
            at = np.array([per[c][s][0] for s in sids])
            rt = np.array([per[c][s][1] for s in sids])
            e = dict(n=len(sids),
                     at_mae_mean=float(at.mean()) if len(at) else None,
                     rt_mae_mean=float(rt.mean()) if len(rt) else None)
            if c != 'model252':
                common = sorted(set(per[c]) & set(per['model252']))
                if len(common) >= 10:
                    da = np.array([per[c][s][0] - per['model252'][s][0]
                                   for s in common])
                    dr = np.array([per[c][s][1] - per['model252'][s][1]
                                   for s in common])
                    e.update(
                        n_paired=len(common),
                        at_diff_median=float(np.median(da)),
                        rt_diff_median=float(np.median(dr)),
                        at_p=float(wilcoxon(da).pvalue) if np.any(da) else 1.0,
                        rt_p=float(wilcoxon(dr).pvalue) if np.any(dr) else 1.0)
            tab[c] = e
        res[split] = tab

        print(f'\n=== {split} (AT/RT map MAE vs commercial, ms; paired vs '
              f'model252) ===')
        print(f'{"condition":10s} {"n":>4s} {"AT_mae":>8s} {"RT_mae":>8s} '
              f'{"dAT_med":>8s} {"p_AT":>9s} {"dRT_med":>8s} {"p_RT":>9s}')
        for c in CONDS:
            e = tab[c]
            if not e['n']:
                continue
            base = (f'{c:10s} {e["n"]:4d} {e["at_mae_mean"]:8.2f} '
                    f'{e["rt_mae_mean"]:8.2f}')
            if 'at_p' in e:
                base += (f' {e["at_diff_median"]:+8.2f} {e["at_p"]:9.2g}'
                         f' {e["rt_diff_median"]:+8.2f} {e["rt_p"]:9.2g}')
            print(base)

    json.dump(res, open(SUMJ, 'w'), indent=2)
    print('\nwrote', SUMJ)


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sanity', action='store_true')
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--eval', action='store_true')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--split')
    p.add_argument('--n', type=int, default=80)
    p.add_argument('--n_recs', type=int, default=12)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    if a.sanity:
        sanity(device=a.device)
    if a.calibrate:
        calibrate(a.n_recs, device=a.device)
    if a.eval:
        evaluate(a.split, n=a.n, shard=a.shard, nshards=a.nshards,
                 device=a.device)
    if a.summary:
        summary(n=a.n)
