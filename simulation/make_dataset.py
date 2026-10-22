"""
Generate the Route A simulated cohort in the vendor .ecgdata format so that the
clinical pipeline (detector -> preprocessing -> PPE-MAE -> MFS inverse) runs on
it unchanged.

Layout (SIM_ROOT = $ECGI_SIM_ROOT):
  <SIM_ROOT>/<split>/<cohort>-<patient>/            symlinks: .heart .torso .valve .landmark
  <SIM_ROOT>/<split>/<cohort>-<patient>/CI/<id>.ecgdata      raw (L, 256) float32 + int32 header
  <SIM_ROOT>/<split>/<cohort>-<patient>/CI/<id>.gt.npz       ground truth + scenario metadata
  <SIM_ROOT>/<split>/manifest.json

Splits reuse the clinical patient split (bsp_reconstruction/dataset.collect_split, seed 42):
  train : geometries of the training patients (cohort A + cohort B train)
  val   : cohort B validation patients
  test  : cohort B test patients + the first --n_external external patients
so simulated training data never sees a test geometry.

  python simulation/make_dataset.py --split train --n_per_geom 12 --workers 24
  python simulation/make_dataset.py --split test  --n_per_geom 6  --n_external 40 --store_phi
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import glob
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.join(_HERE, '..', 'bsp_reconstruction'), os.path.join(_HERE, '..', 'inverse_evaluation')):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import simcore as S                                     # noqa: E402
import noise as NZ                                      # noqa: E402
import dataset as D                                     # noqa: E402
from mesh_io import find_patient_files                  # noqa: E402
from torso_loader import load_torso_coords              # noqa: E402

SIM_ROOT = os.environ.get('ECGI_SIM_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sim'))
L_REC = 7000


def write_ecgdata(path, sig256):
    """(256, L) float32 -> vendor layout: int32 L, int32 C, float32 (L, C)."""
    sig = np.ascontiguousarray(sig256.T.astype(np.float32))
    with open(path, 'wb') as f:
        np.array([sig.shape[0], sig.shape[1]], np.int32).tofile(f)
        sig.tofile(f)


def geometry_lists(seed=42, n_external=40):
    """{'train': [(cohort, patient, torso_path)], 'val': ..., 'test': ...}"""
    buckets, cohortB = D.collect_split(seed=seed)
    out = {'train': [], 'val': [], 'test': []}
    seen = set()

    def add(role, sid, tp):
        coh, pat = sid.split('__', 2)[:2]
        if tp is None or (coh, pat) in seen:
            return
        seen.add((coh, pat)); out[role].append((coh, pat, tp))

    for sid, _, tp in buckets['train']:
        add('train', sid, tp)
    for sid, _, tp in buckets['val']:
        add('val', sid, tp)
    for sid, _, tp in buckets['test_internal']:
        add('test', sid, tp)
    ext = 0
    for sid, _, tp in buckets['test_external']:
        if ext >= n_external:
            break
        n0 = len(out['test']); add('test', sid, tp); ext += len(out['test']) - n0
    return out


def missing_fingerprint(seed=42):
    """Per-channel missing rate of the TRAINING recordings on the 252 kept
    channels (the clinical 'spatial fingerprint'), plus keep_idx."""
    buckets, _ = D.collect_split(seed=seed)
    keep = np.where(D.geometry_union(buckets['train']))[0].astype(np.int64)
    masks = D.precompute_masks(buckets['train'], keep, verbose=False)
    rate = np.mean([masks[fp] for _, fp, _ in buckets['train']], axis=0)
    return rate, keep


def simulate_patient(task):
    (coh, pat, torso_path, split, n_scen, seed, realism, out_root, store_phi,
     rate252, keep_idx) = task
    gdir = os.path.dirname(torso_path)
    pdir = os.path.join(out_root, split, f'{coh}-{pat}')
    ci = os.path.join(pdir, 'CI'); os.makedirs(ci, exist_ok=True)
    files = find_patient_files(gdir)
    for k, src in files.items():
        if src is None:
            continue
        dst = os.path.join(pdir, os.path.basename(src))
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    geo = S.SimGeometry(gdir, torso_path)
    coords = load_torso_coords(torso_path)[keep_idx]
    recs = []
    t0 = time.time()
    for k in range(n_scen):
        if all(os.path.exists(os.path.join(ci, f'{coh}-{pat}-s{k:03d}{ext}')) for ext in ('.ecgdata', '.gt.npz')):
            recs.append(f'{coh}-{pat}-s{k:03d}'); continue
        rng = np.random.default_rng([seed, k])          # per-scenario stream: reproducible even when files are skipped
        jitter = realism != 'clean'
        fwd = S.DoubleLayerForward(geo, rng, jitter=jitter)
        scen = S.make_scenario(geo, rng)
        beat = S.simulate_beat(geo, fwd, scen)
        bsp, phi, starts, rr = NZ.assemble_beats(beat['bsp'], beat['phi'], rng, L=L_REC,
                                                 rr_ms=scen['meta'].get('rr_ms'),
                                                 min_rr_frac=(0.7 if scen['meta']['kind'] == 'vt' else None))
        gain = NZ.calibrate_gain(bsp, rng)
        bsp *= gain; phi *= gain
        sig = np.zeros((NZ.NCH, L_REC)); sig[geo.t_ch] = bsp
        clean = sig.copy()
        if realism in ('missing', 'full'):
            miss = NZ.sample_missing(rate252, coords, rng, keep_idx)
        else:
            miss = np.ones(NZ.NCH, bool); miss[keep_idx] = False
        active_idx = np.where(~miss)[0]
        sig = NZ.add_noise(sig, active_idx, rng, level=realism)
        if realism in ('noise', 'full'):
            o_idx, n_idx = NZ.bad_channels(rng, active_idx)
            sig = NZ.apply_bad_channels(sig, o_idx, n_idx, rng)
        else:
            o_idx, n_idx = np.array([], int), np.array([], int)
        sig[miss] = 0.0
        sig[miss] += 0.002 * rng.normal(size=(int(miss.sum()), L_REC)) * (realism in ('noise', 'full'))
        stem = f'{coh}-{pat}-s{k:03d}'
        write_ecgdata(os.path.join(ci, stem + '.ecgdata'), sig)
        gt = dict(AT=scen['AT'].astype(np.float32), RT=scen['RT'].astype(np.float32),
                  APD=scen['APD'].astype(np.float32), amp=scen['amp'].astype(np.float32),
                  meta=json.dumps(scen['meta']), beat_starts=starts, pre_ms=beat['pre_ms'],
                  rr_ms=rr, gain=gain, missing=miss, outlier_ch=o_idx, noisy_ch=n_idx,
                  realism=realism, n_epi=geo.n_epi, t_ch=geo.t_ch, kappa=fwd.kappa)
        if store_phi:
            gt['phi_beat'] = (beat['phi'] * gain).astype(np.float16)
            gt['bsp_clean'] = clean[keep_idx].astype(np.float16)
        np.savez_compressed(os.path.join(ci, stem + '.gt.npz'), **gt)
        recs.append(stem)
    return coh, pat, recs, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    p.add_argument('--n_per_geom', type=int, default=10)
    p.add_argument('--n_external', type=int, default=40)
    p.add_argument('--limit_geoms', type=int, default=0, help='pilot: only the first K geometries')
    p.add_argument('--realism', default='full', choices=['clean', 'missing', 'noise', 'full'])
    p.add_argument('--out_root', default=SIM_ROOT)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--store_phi', action='store_true', help='store GT epicardial beat + clean BSP (test)')
    a = p.parse_args()
    geoms = geometry_lists(a.seed, a.n_external)[a.split]
    if a.limit_geoms:
        geoms = geoms[:a.limit_geoms]
    rate252, keep_idx = missing_fingerprint(a.seed)
    out_root = a.out_root if a.realism == 'full' else a.out_root + '_' + a.realism
    os.makedirs(os.path.join(out_root, a.split), exist_ok=True)
    split_id = {'train': 1, 'val': 2, 'test': 3}[a.split]
    tasks = [(coh, pat, tp, a.split, a.n_per_geom, a.seed * 1000003 + i * 7919 + split_id,
              a.realism, out_root, a.store_phi, rate252, keep_idx) for i, (coh, pat, tp) in enumerate(geoms)]
    print(f'[{a.split}] {len(tasks)} geometries x {a.n_per_geom} scenarios, realism={a.realism} -> {out_root}', flush=True)
    manifest = {}
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for i, (coh, pat, recs, dt) in enumerate(pool.imap_unordered(simulate_patient, tasks)):
                manifest[f'{coh}-{pat}'] = recs
                print(f'  [{i+1}/{len(tasks)}] {coh}-{pat}: {len(recs)} recordings ({dt:.0f}s)', flush=True)
    else:
        for i, t in enumerate(tasks):
            coh, pat, recs, dt = simulate_patient(t); manifest[f'{coh}-{pat}'] = recs
            print(f'  [{i+1}/{len(tasks)}] {coh}-{pat}: {len(recs)} recordings ({dt:.0f}s)', flush=True)
    with open(os.path.join(out_root, a.split, 'manifest.json'), 'w') as f:
        json.dump(dict(split=a.split, realism=a.realism, n_per_geom=a.n_per_geom, seed=a.seed,
                       geometries=manifest, n_recordings=sum(len(v) for v in manifest.values())), f, indent=1)
    print(f'done: {sum(len(v) for v in manifest.values())} recordings in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
