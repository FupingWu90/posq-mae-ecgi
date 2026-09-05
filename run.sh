#!/bin/bash
# FixdElectrodsN_Recon training launcher.
# Reconstruct a full 252-electrode ECGi recording from a FIXED number (VC) of
# visible valid channels + electrode coordinates.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
LOG=$ROOT/train_run.out

cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- full run (base model, VC=32, cluster/greedy selection, smoothness on) ----
# Tunables shown in the output dir name: model, vc, mode, w_spatial/w_temporal,
# reg_scope, reg_knn/stride, effective batch (batch*accum), crop, epochs.
python train.py \
  --model fixedn_ecgi_base --patch_size 1 50 \
  --vc 32 --select_mode cluster --cluster_method kmean \
  --w_spatial 0.1 --w_temporal 0.1 --reg_scope recon --reg_knn 6 --reg_time_stride 10 \
  --epochs 800 --batch_size 64 --accum_iter 1 --aug_times 2 \
  --use_amp --num_workers 8 --save_every 40 --eval_every 20 \
  >> "$LOG" 2>&1

# ---- example: VC sweep / patch or random modes ----
#   --vc 16 / --vc 64
#   --select_mode patch
#   --select_mode random
#   --cluster_method kmean      (only affects --select_mode cluster)
#   --reg_scope masked          (smoothness over all masked channels)
