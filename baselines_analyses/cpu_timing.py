"""Single-core CPU completion time of a run (one 5-s recording, N = 16, template layout), median of repeats,
measured like the v1 cost table (S10).  Prints aggregates only.
  python baselines_analyses/cpu_timing.py --run_dir <run> [--reps 20] [--threads 1]"""
import os, sys, time, json, argparse, warnings
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import numpy as np, torch
from build_sources import load_model, build_context, reconstruct_sources
from dataset import collect_split

ap = argparse.ArgumentParser(); ap.add_argument('--run_dir', required=True); ap.add_argument('--reps', type=int, default=20); ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args(); warnings.filterwarnings('ignore')
torch.set_num_threads(a.threads)
dev = torch.device('cpu')
model, keep_idx, args = load_model(a.run_dir, dev)
rec = [r for r in collect_split(seed=42)[0]['test_internal'] if r[2]][0]
ctx = build_context(rec[1], rec[2], keep_idx, args, vc=16, mode='template', cluster_method='greedy', seed=42)
with torch.no_grad():
    reconstruct_sources(model, ctx, dev)   # warm-up
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); reconstruct_sources(model, ctx, dev); ts.append(time.perf_counter() - t0)
print(json.dumps(dict(run=os.path.basename(a.run_dir.rstrip('/')), threads=a.threads, reps=a.reps, median_s=float(np.median(ts)), min_s=float(min(ts)), max_s=float(max(ts)))))
