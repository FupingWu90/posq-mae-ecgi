"""Phase-0 parity test: the v2 copy of the evaluation chain must reproduce v1 exactly.

(1) BSP completion metrics of the v1 final checkpoint, computed with the v2 engine/dataset code,
    vs. the numbers stored in the v1 run directory (validation best MAE; test_compare.json, k-means).
(2) Downstream per-recording metrics of the first N test recordings computed with the v2 downstream
    code vs. the v1 per-recording CSV used by the paper.
Read-only w.r.t. v1.  Writes results/parity_report.md.
"""
import os, sys, json, argparse, warnings
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import torch
from torch.utils.data import DataLoader
from paths import V1_RUNS, V1_GENERATED
from dataset import build_experiment_datasets
from engine import run_eval
from build_sources import load_model
import eval_downstream as E

RUN = os.path.join(V1_RUNS, 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800')


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--n_down', type=int, default=20); a = ap.parse_args()
    dev = torch.device('cuda')
    lines = ['# Parity report: v2 evaluation chain vs v1 (v1 final wide model, N=16)', '']
    args = json.load(open(os.path.join(RUN, 'args.json')))
    stats = json.load(open(os.path.join(RUN, 'training_stats.json')))
    comp = json.load(open(os.path.join(RUN, 'test_compare.json')))
    train_ds, val_ds, ti_ds, te_ds, info = build_experiment_datasets(
        crop_len=args['crop_len'], preprocess=True, seed=args['seed'], aug_times=1, pp_norm=args['pp_norm'], tag='exp',
        min_active=args['vc'] + args['min_recon'], prep_cache=True, norm_mode=args['norm_mode'], verbose=False)
    model, keep_idx, _ = load_model(RUN, dev)
    assert np.array_equal(keep_idx, info['keep_idx'])
    dl = lambda ds: DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)
    ok = True
    lines += ['## 1. BSP completion (k-means deployment, seed 42)', '', '| split | metric | v1 stored | v2 recomputed | abs diff |', '|---|---|---|---|---|']
    r = run_eval(model, dl(val_ds), dev, args['vc'], 'cluster', 'kmean', seed=args['seed'], norm_mode=args['norm_mode'])
    d = abs(r['mae'] - stats['val_best_mae']); ok &= d < 1e-4
    lines.append(f"| val | MAE | {stats['val_best_mae']:.6f} | {r['mae']:.6f} | {d:.2e} |")
    for name, ds in (('test_internal', ti_ds), ('test_external', te_ds)):
        r = run_eval(model, dl(ds), dev, args['vc'], 'cluster', 'kmean', seed=args['seed'], norm_mode=args['norm_mode'])
        ref = comp['results'][name]['cluster-kmean']
        for k in ('mae', 'rmse', 'cc'):
            d = abs(r[k] - ref[k]); ok &= d < 1e-4
            lines.append(f"| {name} | {k.upper()} | {ref[k]:.6f} | {r[k]:.6f} | {d:.2e} |")
    # ---- downstream ----
    # Model-free rows (raw N-electrode inverse; references) must match exactly. Model-dependent rows
    # (completion -> inverse) carry GPU floating-point noise (two runs of the SAME code differ at the
    # 1e-7 level in the completed field, which occasionally moves a marker by one sample), so they are
    # checked against a tolerance and the run-to-run noise of v2 itself is reported next to it.
    import csv, io, contextlib
    lines += ['', f'## 2. Downstream per-recording metrics (first {a.n_down} recordings per cohort, k-means, vs v1 CSV)', '',
              '| split | rows | model-free cells: max abs diff | model-dependent cells: max abs diff vs v1 | same, between two v2 runs |', '|---|---|---|---|---|']
    SKIP = ('sid', 'source', 'ref', 'region', 'bsp_cc', 'n_region_fallback')
    num = lambda v: isinstance(v, (int, float, np.floating, np.integer)) and v is not None and np.isfinite(v)
    for split in ('test_internal', 'test_external'):
        with contextlib.redirect_stdout(io.StringIO()):
            rows = E.evaluate(RUN, split=split, n=a.n_down, vc=16, mode='cluster', cluster_method='kmean', seed=42, out_csv=None, verbose=False)
            rows2 = E.evaluate(RUN, split=split, n=a.n_down, vc=16, mode='cluster', cluster_method='kmean', seed=42, out_csv=None, verbose=False)
        v1 = {}
        with open(os.path.join(V1_GENERATED, 'vc_percase', f'vc16_{split}.csv')) as f:
            for row in csv.DictReader(f):
                v1[(row['sid'], row['source'], row['ref'])] = row
        mx_free = mx_dep = mx_rr = 0.0; nrow = 0
        for r, r2 in zip(rows, rows2):
            key = (r['sid'], r['source'], r['ref'])
            if key not in v1:
                continue
            nrow += 1
            for k, v in r.items():
                if k in SKIP or not num(v):
                    continue
                if num(r2.get(k)):
                    mx_rr = max(mx_rr, abs(float(v) - float(r2[k])))
                if k not in v1[key] or v1[key][k] == '':
                    continue
                dd = abs(float(v) - float(v1[key][k]))
                if r['source'] == 'src1_vc':
                    mx_free = max(mx_free, dd)
                else:
                    mx_dep = max(mx_dep, dd)
        ok &= (nrow > 0 and mx_free < 1e-6 and mx_dep < 0.05)
        lines.append(f"| {split} | {nrow} | {mx_free:.2e} | {mx_dep:.2e} | {mx_rr:.2e} |")
    lines += ['', 'Pass criteria: BSP metrics within 1e-4; model-free downstream cells within 1e-6; model-dependent cells within 0.05 (ms or CC units).']
    lines += ['', f"**Result: {'PASS' if ok else 'FAIL'}**", '']
    out = os.path.join(V2, 'results', 'parity_report.md')
    open(out, 'w').write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
