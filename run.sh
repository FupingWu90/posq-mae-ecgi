#!/bin/bash
# PosQ-MAE training launcher: the final configuration of the paper (N = 16 measured electrodes).
# Completes the full 252-electrode ECGi recording from a FIXED number (--vc) of measured channels
# plus the electrode coordinates; checkpoints are selected on the validation split under the
# fixed template layout.  Set the ECGI_* cohort roots first (see README.md).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
LOG=$ROOT/train_run.out

cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py --tag FINAL --model posq_ecgi_wide --head subattn --inner_depth 1 --sub_len 5 --inner_dim 32 \
  --batch_size 32 --accum_iter 2 --patch_size 1 50 --vc 16 --norm_mode global \
  --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 --reg_scope recon \
  --select_mode cluster --cluster_method greedy --val_select_mode template \
  --aug_times 1 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20 --downstream_splits val \
  >> "$LOG" 2>&1

# ---- variants (see training_sweeps/pilots.py and training_sweeps/phase3.py for every run of the paper) ----
#   --vc 4 / 8 / 32 / 64                      electrode count
#   --pos_embed index                         ST-MEM-style channel-index embedding instead of physical positions
#   --head linear                             without the sub-token attention head (base query decoder)
#   --w_temporal 0                            without the temporal-smoothness loss
#   --select_mode cluster --cluster_method kmean | --select_mode patch | --select_mode random
#   --select_region front | back              half-torso training (validation: in-region k-means)
#   --train_frac 0.5 --aug_times 2            data scaling (compute matched)
#   --train_sources sim --val_source sim      simulated data only;  --train_sources sim+both  simulated + clinical
