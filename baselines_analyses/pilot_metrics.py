"""Extra VALIDATION metrics of a trained run (v1 or v2 model), under the fixed template layout and under
per-recording k-means:  held-out BSP CC / MAE (units of g), CC of the first temporal difference, and the
patch-boundary discontinuity ratio (mean |step| across patch boundaries / mean |step| inside patches;
1.0 for measured signals).  Aggregates only.

  python baselines_analyses/pilot_metrics.py --run_dir <run> [--out <json>] [--split val]
"""
import os, sys, json, argparse, warnings
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import torch
from paths import assert_v2
from build_sources import load_model, build_context, reconstruct_sources, held_out_bsp_cc
from dataset import collect_split


def _ratio(sig, held, S=50):
    d = np.abs(np.diff(sig[held], axis=1))
    bidx = np.arange(S - 1, sig.shape[1] - 1, S)
    inner = np.ones(d.shape[1], bool); inner[bidx] = False
    return float(d[:, bidx].mean() / (d[:, inner].mean() + 1e-12))


def _dcc(a, b):
    da, db = np.diff(a, axis=1), np.diff(b, axis=1)
    out = []
    for i in range(da.shape[0]):
        if da[i].std() > 1e-10 and db[i].std() > 1e-10:
            out.append(np.corrcoef(da[i], db[i])[0, 1])
    return float(np.mean(out)) if out else float('nan')


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True); p.add_argument('--out', default=None)
    p.add_argument('--split', default='val'); p.add_argument('--n', type=int, default=None)
    a = p.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, keep_idx, args = load_model(a.run_dir, dev)
    vc = args['vc']
    recs = [r for r in collect_split(seed=args.get('seed', 42))[0][a.split] if r[2] is not None]
    if a.n:
        recs = recs[:a.n]
    res = {'run': os.path.basename(a.run_dir.rstrip('/')), 'split': a.split, 'vc': vc}
    for mode, cm in (('template', 'greedy'), ('cluster', 'kmean')):
        M = {k: [] for k in ('bsp_cc', 'bsp_mae_g', 'dcc', 'boundary_ratio', 'boundary_ratio_measured')}
        for sid, ecg, tor in recs:
            ctx = build_context(ecg, tor, keep_idx, args, vc=vc, mode=mode, cluster_method=cm, seed=args.get('seed', 42))
            if ctx is None or ctx['n_active'] < vc + args.get('min_recon', 1):
                continue
            rec = reconstruct_sources(model, ctx, dev)['recon252']
            held = ctx['active252'] & ~ctx['visible252']
            if not held.any():
                continue
            M['bsp_cc'].append(held_out_bsp_cc(ctx, rec)[0])
            M['bsp_mae_g'].append(float(np.abs(rec[held] - ctx['prep252'][held]).mean() / ctx['g']))
            M['dcc'].append(_dcc(rec[held], ctx['prep252'][held]))
            M['boundary_ratio'].append(_ratio(rec, held)); M['boundary_ratio_measured'].append(_ratio(ctx['prep252'], held))
        lab = 'template' if mode == 'template' else 'kmeans'
        res[lab] = {k: float(np.nanmean(v)) for k, v in M.items()}; res[lab]['n'] = len(M['bsp_cc'])
        print(lab, json.dumps(res[lab]))
    out = a.out or os.path.join(a.run_dir, f'pilot_metrics_{a.split}.json')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(assert_v2(out), 'w'), indent=1)
    print('wrote', out)
    # cross-attention summary of the query decoder (which heads retrieve time-specific content); never fatal
    if a.out is None:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from diag_cross_attention import analyse, report
            rx = analyse(a.run_dir, n=10)
            if rx is not None:
                report(rx)
                json.dump(rx, open(assert_v2(os.path.join(a.run_dir, 'cross_attention_val.json')), 'w'), indent=1)
        except Exception as e:
            print(f'[warn] cross-attention summary skipped: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
