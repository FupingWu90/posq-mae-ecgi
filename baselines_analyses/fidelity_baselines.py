"""Field-fidelity metrics (BSP CC, HSP CC/MAE vs commercial) for the CLASSICAL
baselines on the doubling grid N in {4,8,16,32,64}, so the manuscript fidelity
figure (learned_baselines.fidelity_figures) can show ALL methods.

The existing artefacts only retained potential-level fidelity for the trained
models (vc_percase hsp_cc via eval_downstream/metrics.hsp_metrics) and LMMSE's
bsp_cc (lmmse_baseline shard CSVs). This script fills the gaps:

  bsp_cc  : laplacian, tps (+ lmmse recomputed as a cross-check) — per-recording
            mean per-channel Pearson CC (over time, physical units) between the
            method's PREDICTED values and the measured preprocessed signal on
            the held-out channels the method actually predicts:
              laplacian : active & on_mesh & ~visible  (the `fill` set)
              tps       : active & coord_ok & ~visible (the `ev` set)
              lmmse     : active & ~visible            (all held-out)
            Channels with std < 1e-8 on either side are skipped — identical to
            lmmse_baseline.evaluate (and engine._compute_metrics' guard).
  hsp_cc / hsp_mae : laplacian, tps, lmmse — assemble each method's 256-ch field
            EXACTLY as interp_baseline / lmmse_baseline do, solve the SAME MFS
            inverse (assemble_solver + solve_from_field defaults), load the
            commercial epicardial potentials with the SAME load_commercial call
            as eval_downstream (time_window = (crop_start, crop_start+CROP),
            preprocess_reference=True, car=True), then score with
            metrics.hsp_metrics(phi_method, phi_commercial) on ALL heart
            vertices (good=None): hsp_cc = ['cc'] = spatial MEAN of per-vertex
            temporal Pearson CC (denominator guard 1e-12, non-finite dropped),
            hsp_mae = ['mae'] = spatial mean of per-vertex temporal MAE.
            This matches eval_downstream's per-case hsp_cc/hsp_mae columns
            (ref='commercial') verbatim.

Selection is the deployment selection used everywhere:
  select_visible_channels(coords, active, N, 'cluster', 'kmean', default_rng(42))

Usage:
  python fidelity_baselines.py --split test_internal --shard 0 --nshards 4
  python fidelity_baselines.py --summary        # aggregate -> generated/fidelity_baselines.json

--src2 mode (supplementary ref2 tables): identical completions + identical MFS
inverse, but each method x N entry ALSO carries hsp_cc_src2 / hsp_mae_src2 —
hsp_metrics(phi_method, phi2) where phi2 is the ALL-VALID inverse loaded from
inverse_evaluation/cache_hsp/<split>/refs/<sid_safe>.npz key 'phi2' (float16 on disk,
cast to float64 — byte-identical to eval_downstream's ref='src2_allvalid'
reference; all vertices, per-vertex temporal Pearson CC, spatial mean). The
commercial-referenced hsp_cc/hsp_mae are recomputed too and cross-checked
against the existing cache/ record (max |diff| stored as 'commercial_check').
Results go to a SEPARATE cache, generated/fidelity_baselines/cache_src2/, so
the validated cache/ is never touched; --summary merges both caches (existing
keys unchanged, hsp_cc_src2/hsp_mae_src2 added per method/N).

Per-recording results are cached as JSON under
generated/fidelity_baselines/cache/<split>/<sid_safe>.json (atomic writes), so
shards are idempotent/resumable. Failures are recorded (marker JSON with
'failed') and skipped on re-run unless --retry_failed.
"""
import os

# CPU-only companion to a running GPU training: throttle BLAS before numpy.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '3')

import argparse
import json
import sys
import warnings

import numpy as np

from common import GEN, PROJ, run_dir

sys.path.insert(0, os.path.join(PROJ, 'inverse_evaluation'))
sys.path.insert(0, os.path.join(PROJ, 'bsp_reconstruction'))
from build_sources import build_context, _to256, CROP                     # noqa: E402
from channel_select import select_visible_channels                        # noqa: E402
from inverse import assemble_solver, solve_from_field, load_commercial    # noqa: E402
from metrics import hsp_metrics                                           # noqa: E402
from mesh_io import load_surface, find_patient_files                      # noqa: E402
from dataset import collect_split                                         # noqa: E402
from interp_baseline import mesh_laplacian, laplacian_interp, tps_interp  # noqa: E402
from lmmse_baseline import _predict, COVF, LAMF                           # noqa: E402

warnings.filterwarnings('ignore')

VCS = [4, 8, 16, 32, 64]
METHODS = ['laplacian', 'tps', 'lmmse']
SPLITS = ['test_internal', 'test_external']
OUTDIR = os.path.join(GEN, 'fidelity_baselines')
CACHE = os.path.join(OUTDIR, 'cache')
CACHE_SRC2 = os.path.join(OUTDIR, 'cache_src2')
REFS_HSP = os.path.join(PROJ, 'inverse_evaluation', 'cache_hsp')   # <split>/refs/<sid>.npz
SUMMARY_JSON = os.path.join(GEN, 'fidelity_baselines.json')
MAX_GEOM_CACHE = 4          # geometries held in RAM (dense L + MFS solver)


def _setup():
    rd = run_dir(model='fixedn_ecgi_wide', vc=16, ws='0', wt='3', rw=100)
    args = json.load(open(os.path.join(rd, 'args.json')))
    keep_idx = np.load(os.path.join(rd, 'keep_idx.npy')).astype(np.int64)
    return args, keep_idx


def _bsp_cc(pairs):
    """pairs: iterable of (pred_1d, meas_1d). Mean per-channel Pearson CC with
    the std<1e-8 guard on either side — same convention as lmmse_baseline."""
    ccs = []
    for a_, b_ in pairs:
        if a_.std() > 1e-8 and b_.std() > 1e-8:
            ccs.append(float(np.corrcoef(a_, b_)[0, 1]))
    return (float(np.mean(ccs)) if ccs else float('nan')), len(ccs)


def _hsp(phi, phicom):
    """eval_downstream's hsp_cc/hsp_mae: hsp_metrics on ALL vertices."""
    if (phi is None or phicom is None or not np.all(np.isfinite(phi))
            or not np.all(np.isfinite(phicom))):
        return float('nan'), float('nan')
    h = hsp_metrics(phi, phicom)
    return h['cc'], h['mae']


def _load_phi2(split, sid_safe):
    """All-valid inverse from the downstream refs cache, cast float16->float64
    exactly like eval_downstream._f(ref.get('phi2')). None if unavailable."""
    p = os.path.join(REFS_HSP, split, 'refs', sid_safe + '.npz')
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        if 'phi2' not in z.files:
            return None
        return np.asarray(z['phi2'], np.float64)


def _atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def process_recording(rec, keep_idx, args, cov, lam_best, geom_cache,
                      phi2=None, want_src2=False):
    """Compute all method x N metrics for one recording. Returns dict.
    want_src2: also score each inverse vs phi2 (all-valid reference) ->
    hsp_cc_src2/hsp_mae_src2 per entry; phi2 normally comes from the refs npz
    (fallback: recomputed from ctx['src2_allvalid'], flagged phi2_cached=False).
    """
    sid, ecgpath, torsopath = rec
    gdir = os.path.dirname(torsopath)
    if gdir not in geom_cache:
        while len(geom_cache) >= MAX_GEOM_CACHE:
            geom_cache.pop(next(iter(geom_cache)))
        solver_t = assemble_solver(gdir)
        tV, tF, tci = load_surface(find_patient_files(gdir)['torso'])
        geom_cache[gdir] = (solver_t, mesh_laplacian(tV, tF), tci)
    (solver, torso_ci, hV, hF), L, chidx = geom_cache[gdir]
    n_vert, n_h = L.shape[0], hV.shape[0]

    ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16,
                        mode='cluster', cluster_method='kmean', seed=42)
    if ctx is None:
        return {'sid': sid, 'failed': 'build_context returned None'}
    prep, coords, active = ctx['prep252'], ctx['coords'], ctx['active252']
    coord_ok = np.abs(coords).sum(1) > 1e-6
    # mesh-vertex <-> 252-channel maps (verbatim from interp_baseline)
    pos252 = np.full(256, -1, np.int64)
    pos252[keep_idx] = np.arange(len(keep_idx))
    v2c = pos252[chidx]
    c2v = np.full(len(keep_idx), -1, np.int64)
    ok = v2c >= 0
    c2v[v2c[ok]] = np.where(ok)[0]
    on_mesh = c2v >= 0

    tw = (ctx['crop_start'], ctx['crop_start'] + CROP)
    phicom = load_commercial(os.path.splitext(ecgpath)[0] + '.potential',
                             n_h, time_window=tw)
    phi2_cached = True
    if want_src2 and phi2 is None:
        phi2 = solve_from_field(solver, torso_ci, *ctx['src2_allvalid'])[0]
        phi2_cached = False

    res = {m: {} for m in METHODS}
    for vc in VCS:
        visible = select_visible_channels(coords, active, vc, 'cluster',
                                          'kmean', np.random.default_rng(42))
        entries = {}          # method -> (bsp_cc, n_cc, f252, amask)

        # ---- mesh-Laplacian (field assembly verbatim from interp_baseline) --
        kn_c = np.where(visible & on_mesh)[0]
        if len(kn_c) >= 4:
            phiV = laplacian_interp(L, c2v[kn_c],
                                    prep[kn_c].astype(np.float64), n_vert)
            f252 = prep.copy().astype(np.float32)
            fill = active & on_mesh & ~visible
            f252[fill] = phiV[c2v[fill]].astype(np.float32)
            cc, n_cc = _bsp_cc((phiV[c2v[c]], prep[c]) for c in np.where(fill)[0])
            entries['laplacian'] = (cc, n_cc, f252, active & (visible | on_mesh))

        # ---- thin-plate-spline RBF (verbatim from interp_baseline) ----------
        kn = visible & coord_ok
        ev = active & coord_ok & ~visible
        if kn.sum() >= 4 and ev.sum() > 0:
            vals = tps_interp(coords[kn].astype(np.float64),
                              prep[kn].astype(np.float64),
                              coords[ev].astype(np.float64))
            if vals is not None:
                f252 = prep.copy().astype(np.float32)
                f252[ev] = vals.astype(np.float32)
                cc, n_cc = _bsp_cc(zip(vals.astype(np.float32), prep[ev]))
                entries['tps'] = (cc, n_cc, f252, active & (visible | coord_ok))

        # ---- population LMMSE (verbatim from lmmse_baseline.evaluate) -------
        pred = _predict(cov, prep, visible, lam_best[str(vc)]['lam_rel'])
        f252 = pred.copy()
        f252[visible] = prep[visible]                 # keep the measured leads
        held = active & ~visible
        cc, n_cc = _bsp_cc((pred[c], prep[c]) for c in np.where(held)[0])
        entries['lmmse'] = (cc, n_cc, f252, active)

        # ---- MFS inverse + HSP metrics vs commercial (+ src2 all-valid) -----
        for meth, (bcc, n_cc, f252, amask) in entries.items():
            V256, a256 = _to256(f252, amask, keep_idx)
            phi = solve_from_field(solver, torso_ci, V256, a256)[0]
            hcc, hmae = _hsp(phi, phicom)
            ent = dict(
                bsp_cc=None if not np.isfinite(bcc) else round(bcc, 5),
                bsp_cc_n=n_cc,
                hsp_cc=None if not np.isfinite(hcc) else round(hcc, 5),
                hsp_mae=None if not np.isfinite(hmae) else round(hmae, 6))
            if want_src2:
                hcc2, hmae2 = _hsp(phi, phi2)
                ent['hsp_cc_src2'] = None if not np.isfinite(hcc2) else round(hcc2, 5)
                ent['hsp_mae_src2'] = None if not np.isfinite(hmae2) else round(hmae2, 6)
            res[meth][str(vc)] = ent

    out = {'sid': sid, 'has_commercial': phicom is not None, 'results': res}
    if want_src2:
        out['has_src2'] = phi2 is not None
        out['phi2_cached'] = phi2_cached
    return out


def _commercial_check(out, split, sid_safe):
    """src2 mode: max |diff| of the recomputed hsp_cc/hsp_mae vs the validated
    commercial cache record (same completions/inverse -> should be ~0)."""
    cpath = os.path.join(CACHE, split, sid_safe + '.json')
    if not os.path.exists(cpath):
        return None
    try:
        old = json.load(open(cpath)).get('results', {})
    except Exception:
        return None
    diffs = []
    for m in METHODS:
        for vc in VCS:
            a = out.get('results', {}).get(m, {}).get(str(vc))
            b = old.get(m, {}).get(str(vc))
            if not a or not b:
                continue
            for k in ('hsp_cc', 'hsp_mae'):
                if a.get(k) is not None and b.get(k) is not None:
                    diffs.append(abs(a[k] - b[k]))
    return round(max(diffs), 9) if diffs else None


def run_shard(split, shard, nshards, n_max, retry_failed, src2=False):
    args, keep_idx = _setup()
    cov = np.load(COVF)['cov']
    lam_best = json.load(open(LAMF))
    cache_dir = os.path.join(CACHE_SRC2 if src2 else CACHE, split)
    os.makedirs(cache_dir, exist_ok=True)
    buckets, _ = collect_split(seed=42)
    recs = [r for r in buckets[split] if r[2]][:n_max][shard::nshards]
    geom_cache = {}
    n_done = n_skip = n_fail = 0
    for i, rec in enumerate(recs):
        sid = rec[0]
        sid_safe = sid.replace('/', '_')
        cpath = os.path.join(cache_dir, sid_safe + '.json')
        if os.path.exists(cpath):
            try:
                prev = json.load(open(cpath))
            except Exception:
                prev = None
            if prev is not None and ('results' in prev
                                     or ('failed' in prev and not retry_failed)):
                n_skip += 1
                continue
        try:
            phi2 = _load_phi2(split, sid_safe) if src2 else None
            out = process_recording(rec, keep_idx, args, cov, lam_best,
                                    geom_cache, phi2=phi2, want_src2=src2)
            if src2 and 'results' in out:
                out['commercial_check'] = _commercial_check(out, split, sid_safe)
        except Exception as e:
            out = {'sid': sid, 'failed': f'{type(e).__name__}: {e}'}
        if 'failed' in out:
            n_fail += 1
            print(f'[{split} s{shard}] {sid[:44]} FAILED: {out["failed"]}',
                  flush=True)
        else:
            n_done += 1
        _atomic_json(cpath, out)
        if (n_done + n_fail) % 5 == 0:
            print(f'[{split} s{shard}] {i + 1}/{len(recs)} '
                  f'(done {n_done}, skip {n_skip}, fail {n_fail})', flush=True)
    print(f'[{split} s{shard}] COMPLETE: done {n_done}, skipped {n_skip}, '
          f'failed {n_fail} of {len(recs)}', flush=True)


def summarize():
    summary = {'meta': dict(
        vcs=VCS, methods=METHODS,
        selection="select_visible_channels(coords, active, N, 'cluster', "
                  "'kmean', default_rng(42))",
        hsp_def="metrics.hsp_metrics(phi_method, load_commercial(...)) on ALL "
                "heart vertices (good=None); hsp_cc = spatial mean of "
                "per-vertex temporal Pearson CC, hsp_mae = spatial mean of "
                "per-vertex temporal MAE; commercial windowed to "
                "(crop_start, crop_start+CROP), FIR-HP@1Hz + Savitzky-Golay, "
                "CAR — identical to eval_downstream ref='commercial'",
        bsp_def="mean per-channel Pearson CC (time, physical units) predicted "
                "vs measured on held-out channels; std<1e-8 guard both sides "
                "(lmmse_baseline convention); laplacian scored on "
                "active&on_mesh&~visible, tps on active&coord_ok&~visible, "
                "lmmse on active&~visible",
        hsp_src2_def="metrics.hsp_metrics(phi_method, phi2) on ALL heart "
                "vertices (good=None), where phi2 is the all-valid MFS inverse "
                "loaded from inverse_evaluation/cache_hsp/<split>/refs/<sid>.npz "
                "(float16 on disk, cast float64) — identical to eval_downstream "
                "ref='src2_allvalid'; same completions/inverse as hsp_cc, only "
                "the reference field differs (cache_src2/)",
        aggregation='cohort mean over recordings of finite per-recording values'),
        'splits': {}}
    for split in SPLITS:
        cache_dir = os.path.join(CACHE, split)
        files = sorted(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else []
        acc = {(m, vc): {'bsp_cc': [], 'hsp_cc': [], 'hsp_mae': []}
               for m in METHODS for vc in VCS}
        n_ok = n_fail = n_nocom = 0
        failed_sids = []
        for fn in files:
            if not fn.endswith('.json'):
                continue
            d = json.load(open(os.path.join(cache_dir, fn)))
            if 'failed' in d:
                n_fail += 1
                failed_sids.append(f"{d.get('sid', fn)}: {d['failed']}")
                continue
            n_ok += 1
            if not d.get('has_commercial', False):
                n_nocom += 1
            for m in METHODS:
                for vc in VCS:
                    e = d.get('results', {}).get(m, {}).get(str(vc))
                    if not e:
                        continue
                    for k in ('bsp_cc', 'hsp_cc', 'hsp_mae'):
                        if e.get(k) is not None:
                            acc[(m, vc)][k].append(e[k])
        # ---- src2 all-valid reference (separate cache, additive keys) -------
        s2_dir = os.path.join(CACHE_SRC2, split)
        s2_files = sorted(os.listdir(s2_dir)) if os.path.isdir(s2_dir) else []
        acc2 = {(m, vc): {'hsp_cc_src2': [], 'hsp_mae_src2': []}
                for m in METHODS for vc in VCS}
        n_ok2 = n_fail2 = 0
        failed2 = []
        checks = []
        for fn in s2_files:
            if not fn.endswith('.json'):
                continue
            d = json.load(open(os.path.join(s2_dir, fn)))
            if 'failed' in d:
                n_fail2 += 1
                failed2.append(f"{d.get('sid', fn)}: {d['failed']}")
                continue
            n_ok2 += 1
            if d.get('commercial_check') is not None:
                checks.append(d['commercial_check'])
            for m in METHODS:
                for vc in VCS:
                    e = d.get('results', {}).get(m, {}).get(str(vc))
                    if not e:
                        continue
                    for k in ('hsp_cc_src2', 'hsp_mae_src2'):
                        if e.get(k) is not None:
                            acc2[(m, vc)][k].append(e[k])
        tab = {}
        for m in METHODS:
            tab[m] = {}
            for vc in VCS:
                cell = {}
                for k, v in acc[(m, vc)].items():
                    cell[k] = round(float(np.mean(v)), 5) if v else None
                    cell[k + '_n'] = len(v)
                for k, v in acc2[(m, vc)].items():
                    cell[k] = round(float(np.mean(v)), 5) if v else None
                    cell[k + '_n'] = len(v)
                tab[m][str(vc)] = cell
        summary['splits'][split] = dict(
            n_cached_ok=n_ok, n_failed=n_fail, n_no_commercial=n_nocom,
            failed=failed_sids, methods=tab,
            n_cached_ok_src2=n_ok2, n_failed_src2=n_fail2, failed_src2=failed2,
            commercial_check_max=(round(float(max(checks)), 9)
                                  if checks else None))
    _atomic_json(SUMMARY_JSON, summary)
    print('wrote', SUMMARY_JSON, '\n')
    for split in SPLITS:
        s = summary['splits'][split]
        print(f"== {split}  (ok {s['n_cached_ok']}, failed {s['n_failed']}, "
              f"no-commercial {s['n_no_commercial']}; src2 ok "
              f"{s['n_cached_ok_src2']}, failed {s['n_failed_src2']}, "
              f"com-check max {s['commercial_check_max']}) ==")
        print(f"{'method':10s} {'N':>3s} {'bsp_cc':>8s} {'n':>4s} "
              f"{'hsp_cc':>8s} {'n':>4s} {'hsp_mae':>9s} "
              f"{'cc_src2':>8s} {'n':>4s} {'mae_src2':>9s}")
        for m in METHODS:
            for vc in VCS:
                e = s['methods'][m][str(vc)]
                b = e.get('bsp_cc'); h = e.get('hsp_cc'); a = e.get('hsp_mae')
                h2 = e.get('hsp_cc_src2'); a2 = e.get('hsp_mae_src2')
                print(f"{m:10s} {vc:3d} "
                      f"{b if b is not None else float('nan'):8.4f} {e.get('bsp_cc_n', 0):4d} "
                      f"{h if h is not None else float('nan'):8.4f} {e.get('hsp_cc_n', 0):4d} "
                      f"{a if a is not None else float('nan'):9.4f} "
                      f"{h2 if h2 is not None else float('nan'):8.4f} "
                      f"{e.get('hsp_cc_src2_n', 0):4d} "
                      f"{a2 if a2 is not None else float('nan'):9.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=SPLITS)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--n', type=int, default=10**9)
    p.add_argument('--retry_failed', action='store_true')
    p.add_argument('--src2', action='store_true',
                   help='also score vs the all-valid inverse (phi2); results '
                        'go to cache_src2/ (the commercial cache/ is untouched)')
    p.add_argument('--summary', action='store_true')
    a = p.parse_args()
    if a.summary:
        summarize()
    elif a.split:
        run_shard(a.split, a.shard, a.nshards, a.n, a.retry_failed, src2=a.src2)
    else:
        p.error('need --split (shard mode) or --summary')
