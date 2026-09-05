#!/usr/bin/env python3
"""Live dashboard for the select_mode sweep. Reads each run's training_stats.json
(in-progress: epochs done + best val MAE) and test_compare.json (final: per test
set, per selection mode). Run anytime:  python training_sweeps/collect_results.py"""
import os, json, glob
_PROJ = os.environ.get('ECGI_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS = _PROJ + "/runs/data-cohortA+cohortB_vs_external"
CFGS = ["cluster-greedy", "cluster-kmean", "patch", "random"]


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def find_dir(cfg):
    hits = glob.glob(os.path.join(RUNS, f"fixedn_ecgi_base_vc16_{cfg}_nm-gl_*"))
    return hits[0] if hits else None


print(f"{'config':16s} {'epochs':>8s} {'val_MAE':>9s} {'ep/min':>7s} {'gpuGB':>6s}   status")
print("-" * 70)
done = {}
for cfg in CFGS:
    d = find_dir(cfg)
    if not d:
        print(f"{cfg:16s} {'--':>8s}  (not started)")
        continue
    st = _load(os.path.join(d, 'training_stats.json')) or {}
    ep = f"{st.get('epochs_done','?')}/{st.get('epochs_total','?')}"
    vmae = st.get('val_best_mae'); vmae = f"{vmae:.4f}" if isinstance(vmae, (int, float)) else '--'
    epm = st.get('mean_epoch_sec'); epm = f"{epm/60:.1f}" if isinstance(epm, (int, float)) else '--'
    gpu = st.get('gpu_peak_reserved_gb', '--')
    tc = _load(os.path.join(d, 'test_compare.json'))
    status = 'COMPLETE' if tc else 'running'
    print(f"{cfg:16s} {ep:>8s} {vmae:>9s} {epm:>7s} {str(gpu):>6s}   {status}")
    if tc:
        done[cfg] = tc

if done:
    print("\n=== FINAL TEST (MAE / CC) — reconstructed channels, per test set x eval mode ===")
    modes = ["cluster-greedy", "cluster-kmean", "patch", "random"]
    for split in ("test_internal", "test_external"):
        print(f"\n{split}:")
        print(f"  {'train_cfg':16s} " + " ".join(f"{m:>16s}" for m in modes))
        for cfg in CFGS:
            if cfg not in done:
                continue
            res = done[cfg]['results'].get(split, {})
            cells = []
            for m in modes:
                mm = res.get(m, {})
                cells.append(f"{mm.get('mae', float('nan')):.3f}/{mm.get('cc', float('nan')):.2f}"
                             if mm else "--")
            print(f"  {cfg:16s} " + " ".join(f"{c:>16s}" for c in cells))
    print("\n(lower MAE / higher CC = better; external = generalization signal)")
