"""
Precompute + cache the MODEL-INDEPENDENT inverse HSP and AT/RT for a split, so
they are reused across every trained reconstruction model.

Cached per recording (phi stored float16 to bound disk):
  refs/<sid>.npz          (VC-independent):
      phi2, AT2, RT2, ARI2, good2         source (2) all-valid  [reference]
      phicom, ATcom, RTcom, ARIcom, goodcom  commercial .potential [reference]
      n_active, n_h, crop_start
  vc<TAG>/<sid>.npz       (VC-selection-dependent):
      phi1, AT1, RT1, ARI1, visible252, n_vc   source (1) VC-only  [baseline]

`refs` are solved once (independent of VC / model). `vc<TAG>` depends on the VC
selection (vc, mode, cluster_method, seed[, region]) but NOT on any trained
model. A region-restricted selection ('front'/'back') gets its own tag suffix
`_reg-<region>` so it never collides with the full-torso caches.

Only keep_idx + args (pp_norm/crop_len) are read from --run_dir — both are
data-config, not model weights — so the cache is valid for any model sharing the
data config.

  python cache_references.py --run_dir <trained run> --split test_internal [--n N]
"""
import os, sys, json, argparse, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction')
sys.path.insert(0, _CODE)

from build_sources import build_context, CROP
from inverse import assemble_solver, solve_from_field, load_commercial
from at_rt import extract_sample, good_vertex_mask
from dataset import collect_split

CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_hsp')   # v2: new entries
from paths import V1_HSP_CACHE as CACHE_ROOT_V1, assert_v2            # noqa: E402  (v1 cache: read-only)


def cache_lookup(split, sub, fname):
    """Existing cache file for (split, sub-folder, file): the v2 cache first, then the
    frozen v1 cache (read-only). None if neither has it."""
    for root in (CACHE_ROOT, CACHE_ROOT_V1):
        p = os.path.join(root, split, sub, fname)
        if os.path.exists(p):
            return p
    return None


def vc_tag(vc, mode, cm, seed, region='all'):
    m = f"{mode}-{cm}" if mode == 'cluster' else mode
    return f"vc{vc}_{m}_s{seed}" + (f"_reg-{region}" if region != 'all' else "")


def _atrt_pack(phi):
    """extract_sample -> (AT,RT,ARI,qrs_good,t_good) int16/bool, or None."""
    if phi is None or not np.all(np.isfinite(phi)):
        return None
    r = extract_sample(phi)
    if r is None:
        return None
    gm = good_vertex_mask(r['sigavg'], *r['windows'])
    return (r['AT'].astype(np.int16), r['RT'].astype(np.int16),
            r['ARI'].astype(np.int16), gm['qrs'], gm['t'])


def cache_split(run_dir, split, n=None, vc=16, mode='cluster',
                cluster_method='greedy', seed=42, verbose=True, region='all'):
    args = json.load(open(os.path.join(run_dir, 'args.json')))
    keep_idx = np.load(os.path.join(run_dir, 'keep_idx.npy')).astype(np.int64)
    hold = None
    if args.get('holdout_file'):   # run trained with --holdout_file: extra role 'holdout' (2026-10-20)
        hold = set(l.strip() for l in open(args['holdout_file']) if l.strip())
    buckets, _ = collect_split(seed=args.get('seed', 42), holdout=hold)
    if split == 'holdout' and not hold:
        raise SystemExit('split=holdout but the run was trained without --holdout_file')
    recs = buckets[split]
    if n:
        recs = recs[:n]

    tag = vc_tag(vc, mode, cluster_method, seed, region)
    refs_dir = os.path.join(CACHE_ROOT, split, 'refs')
    vc_dir = os.path.join(CACHE_ROOT, split, tag)
    os.makedirs(refs_dir, exist_ok=True); os.makedirs(vc_dir, exist_ok=True)

    solver_cache = {}
    n_ref_new = n_vc_new = 0
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        if torsopath is None or not _dedup_ok(sid):
            continue
        sid_safe = sid.replace('/', '_')
        ref_p = os.path.join(refs_dir, sid_safe + '.npz')
        vc_p = os.path.join(vc_dir, sid_safe + '.npz')
        have_ref = cache_lookup(split, 'refs', sid_safe + '.npz') is not None
        have_vc = cache_lookup(split, tag, sid_safe + '.npz') is not None
        if have_ref and have_vc:
            continue
        try:
            geom_dir = os.path.dirname(torsopath)
            if geom_dir not in solver_cache:
                solver_cache[geom_dir] = assemble_solver(geom_dir)
            solver, torso_ch_idx, heart_V, heart_F = solver_cache[geom_dir]
            n_h = heart_V.shape[0]

            ctx = build_context(ecgpath, torsopath, keep_idx, args,
                                vc=vc, mode=mode, cluster_method=cluster_method, seed=seed,
                                region=region)
            if ctx is None:
                continue
            tw = (ctx['crop_start'], ctx['crop_start'] + CROP)

            # ---- refs (VC-independent): src2 all-valid + commercial ----
            if not have_ref:
                phi2, lam2, _, _, na2 = solve_from_field(solver, torso_ch_idx, *ctx['src2_allvalid'])
                pk2 = _atrt_pack(phi2)
                phicom = load_commercial(os.path.splitext(ecgpath)[0] + '.potential', n_h, time_window=tw)
                pkc = _atrt_pack(phicom)
                d = dict(n_active=ctx['n_active'], n_h=n_h, crop_start=ctx['crop_start'],
                         lam2=lam2 if lam2 is not None else np.nan)
                if phi2 is not None:
                    d['phi2'] = phi2.astype(np.float16)     # src2 all-valid HSP (inverse-derived)
                    if pk2: d.update(AT2=pk2[0], RT2=pk2[1], ARI2=pk2[2],
                                     qrs_good2=pk2[3], t_good2=pk2[4])
                # commercial: cache only its AT/RT + good masks (the HSP is the
                # .potential on disk, re-read on the fly when needed for potential-CC).
                if phicom is not None and pkc:
                    d.update(ATcom=pkc[0], RTcom=pkc[1], ARIcom=pkc[2],
                             qrs_goodcom=pkc[3], t_goodcom=pkc[4])
                np.savez_compressed(assert_v2(ref_p), **d); n_ref_new += 1

            # ---- src1 VC-only (VC-dependent) ----
            if not have_vc:
                phi1, lam1, _, _, na1 = solve_from_field(solver, torso_ch_idx, *ctx['src1_vc'])
                pk1 = _atrt_pack(phi1)
                d = dict(visible252=ctx['visible252'], n_vc=ctx['n_vc'],
                         lam1=lam1 if lam1 is not None else np.nan)
                if phi1 is not None:
                    d['phi1'] = phi1.astype(np.float16)
                    if pk1: d.update(AT1=pk1[0], RT1=pk1[1], ARI1=pk1[2])
                np.savez_compressed(assert_v2(vc_p), **d); n_vc_new += 1

            if verbose and (i % 10 == 0 or i + 1 == len(recs)):
                print(f"  [{i+1}/{len(recs)}] {sid_safe[:44]:44s} "
                      f"n_active={ctx['n_active']:3d} n_h={n_h}", flush=True)
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{len(recs)}] {sid[:40]} FAILED: {type(e).__name__}: {e}")

    print(f"[{split}] refs cached +{n_ref_new}, vc[{tag}] cached +{n_vc_new}  "
          f"-> {refs_dir} , {vc_dir}")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--split', default='test_internal',
                   choices=['test_internal', 'test_external', 'val', 'holdout'])
    p.add_argument('--n', type=int, default=None)
    p.add_argument('--vc', type=int, default=16)
    p.add_argument('--mode', default='cluster')
    p.add_argument('--cluster_method', default='greedy')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--region', default='all', choices=['all', 'front', 'back'],
                   help='restrict the VC candidates to one torso half (tag gets _reg-<region>)')
    a = p.parse_args()
    cache_split(a.run_dir, a.split, n=a.n, vc=a.vc, mode=a.mode,
                cluster_method=a.cluster_method, seed=a.seed, region=a.region)
