#!/bin/bash
# Multi-seed variance: Ours@16 and pe-index@16, training-RNG seeds 43 & 44
# (seed 42 = the existing runs). Split + deployment selection stay on seed 42
# via --rng_seed, so all seeds are evaluated on identical splits/selections.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
RUNS=$ROOT/runs/data-cohortA+cohortB_vs_external
LOGDIR=$ROOT/training_sweeps/logs
LOG=$LOGDIR/seed_variance.log
cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--model fixedn_ecgi_wide --batch_size 32 --accum_iter 2 --patch_size 1 50 \
  --vc 16 --norm_mode global --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 \
  --reg_scope recon --select_mode cluster --cluster_method greedy \
  --val_select_mode cluster --val_cluster_method kmean \
  --aug_times 1 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20"
BASE="fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800"
echo "[$(date '+%F %T')] ===== seed-variance queue started (pid $$) =====" | tee -a "$LOG"
while pgrep -f "run_baseline_models" >/dev/null || pgrep -f "train\.py .*--model" >/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] GPU free -> starting" | tee -a "$LOG"
run_one () {
  local rs="$1" tag="$2"; shift 2
  local sd="$RUNS/${BASE}_rs${rs}${tag}"
  if [ -f "$sd/test_compare.json" ]; then echo "[$(date '+%F %T')] SKIP rs$rs$tag" | tee -a "$LOG"; return; fi
  local resume=""; [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
  echo "[$(date '+%F %T')] START rs$rs$tag ${resume:+(resume)}" | tee -a "$LOG"
  python train.py $COMMON --rng_seed "$rs" $@ $resume >> "$LOGDIR/sv_rs${rs}${tag}.out" 2>&1
  echo "[$(date '+%F %T')] END   rs$rs$tag (exit $?)" | tee -a "$LOG"
}
run_one 43 ""
run_one 44 ""
run_one 43 "_pe-index" --pos_embed index
run_one 44 "_pe-index" --pos_embed index
echo "[$(date '+%F %T')] ===== seed-variance queue COMPLETE =====" | tee -a "$LOG"
