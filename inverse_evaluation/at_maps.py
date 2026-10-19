"""Per-recording epicardial activation / repolarisation maps of a trained run (source src4 = completed field,
measured on the visible electrodes), saved raw per vertex for reference-free analyses (plausibility metric,
spatial / propagation figures).  CPU-only by default (CUDA hidden) so it can run next to a training job.
  python inverse_evaluation/at_maps.py --run_dir <run> --split test_internal --layout template [--n N]
  -> baselines_analyses/generated/atmaps/<ID>/<split>_<layout>/<sid>.npz  (AT4, RT4 int16 ms; qrs_good4, t_good4; visible252)
"""
import os, sys, argparse, warnings
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
import torch
from build_sources import load_model, build_context, reconstruct_sources
from inverse import assemble_solver, solve_from_field
from at_rt import extract_sample, good_vertex_mask
from dataset import collect_split
from paths import V2_GENERATED, assert_v2

LAYOUTS = {'template': ('template', 'greedy'), 'kmeans': ('cluster', 'kmean')}


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True); ap.add_argument('--split', required=True)
    ap.add_argument('--layout', default='template', choices=sorted(LAYOUTS)); ap.add_argument('--n', type=int, default=10**9)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run = a.run_dir.rstrip('/'); rid = os.path.basename(run).split('__')[0]
    out_dir = assert_v2(os.path.join(V2_GENERATED, 'atmaps', rid, f'{a.split}_{a.layout}')); os.makedirs(out_dir, exist_ok=True)
    dev = torch.device('cpu')
    model, keep_idx, args = load_model(run, dev)
    vc = int(args['vc']); mode, cm = LAYOUTS[a.layout]
    recs = [r for r in collect_split(seed=args.get('seed', 42))[0][a.split] if r[2]][:a.n]
    solver_cache = {}; n_new = n_skip = n_fail = 0
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        p = os.path.join(out_dir, sid.replace('/', '_') + '.npz')
        if os.path.exists(p):
            n_skip += 1; continue
        try:
            gdir = os.path.dirname(torsopath)
            if gdir not in solver_cache:
                if len(solver_cache) > 3:
                    solver_cache.clear()
                solver_cache[gdir] = assemble_solver(gdir)
            solver, tci, hV, hF = solver_cache[gdir]
            ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=vc, mode=mode, cluster_method=cm, seed=args.get('seed', 42))
            if ctx is None:
                np.savez_compressed(p, valid=False); n_fail += 1; continue
            with torch.no_grad():
                recon = reconstruct_sources(model, ctx, dev)
            phi = solve_from_field(solver, tci, *recon['src4_recon_valid'])[0]
            r = extract_sample(phi) if (phi is not None and np.all(np.isfinite(phi))) else None
            if r is None:
                np.savez_compressed(p, valid=False); n_fail += 1; continue
            g = good_vertex_mask(r['sigavg'], *r['windows'])
            np.savez_compressed(p, valid=True, AT4=np.asarray(r['AT'], np.int16), RT4=np.asarray(r['RT'], np.int16),
                                qrs_good4=np.asarray(g['qrs'], bool), t_good4=np.asarray(g['t'], bool),
                                visible252=np.asarray(ctx['visible252'], bool), n_h=int(hV.shape[0]))
            n_new += 1
        except Exception as e:
            np.savez_compressed(p, valid=False, error=f'{type(e).__name__}: {e}'); n_fail += 1
        if (i + 1) % 25 == 0:
            print(f'[{a.split} {a.layout}] {i+1}/{len(recs)} new {n_new} skip {n_skip} fail {n_fail}', flush=True)
    print(f'[{a.split} {a.layout}] DONE new {n_new} skip {n_skip} fail {n_fail} of {len(recs)} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
