#!/bin/bash
# ---------------------------------------------------------------------------
# Data-scaling + composition experiments with the ADOPTED recipe
# (wide, N=16, ws=0, wt=3 rw100, scope recon, greedy train / kmean deploy).
#   A. learning curve : training PATIENTS subsampled to 1/8, 1/4, 1/2
#                       (stratified per cohort, subsample_seed 0); 1/1 = the
#                       existing wide VC=16 run
#   B. composition    : cohort-A-only / cohort-B-only training
# aug_times is scaled so the number of optimiser updates per epoch stays
# ~constant (986 recs/epoch at 1/1):  1/8 x7 (931), 1/4 x3 (873), 1/2 x2
# (1016), cohortA x2 (1018), cohortB x2 (954)  -> within +-12% of the reference.
# Waits for the running VC=72 job to release the GPU, then runs SERIALLY
# (~19 h each); idempotent; resumes checkpoint_last. No watchdog: if the
# weekly reboot interrupts a run, re-run this script by hand.
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
RUNS=$ROOT/runs/data-cohortA+cohortB_vs_external
LOGDIR=$ROOT/training_sweeps/logs
LOG=$LOGDIR/data_scaling.log
mkdir -p "$LOGDIR"; cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--model fixedn_ecgi_wide --batch_size 32 --accum_iter 2 \
  --patch_size 1 50 --vc 16 --norm_mode global \
  --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 --reg_scope recon \
  --select_mode cluster --cluster_method greedy \
  --val_select_mode cluster --val_cluster_method kmean \
  --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20"
BASE="fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800"

echo "[$(date '+%F %T')] ===== data-scaling sweep started (pid $$) =====" | tee -a "$LOG"
# wait for the VC=72 training (and its post-training downstream report) to finish
while pgrep -f "train\.py .*--vc 72( |$)" >/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] GPU free (VC=72 finished) -> starting" | tee -a "$LOG"

run_cfg () {
  local tag="$1"; shift
  local extra="$*"
  local sd="$RUNS/${BASE}_${tag}"
  if [ -f "$sd/test_compare.json" ]; then
    echo "[$(date '+%F %T')] SKIP  $tag (complete)" | tee -a "$LOG"; return 0
  fi
  local resume=""; [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
  echo "[$(date '+%F %T')] START $tag [$extra] ${resume:+(resume)}" | tee -a "$LOG"
  python train.py $COMMON $extra $resume >> "$LOGDIR/ds_${tag}.out" 2>&1
  echo "[$(date '+%F %T')] END   $tag (exit $?)" | tee -a "$LOG"
}

run_cfg tf0.5       --train_frac 0.5   --aug_times 2
run_cfg tf0.25      --train_frac 0.25  --aug_times 3
run_cfg tf0.125     --train_frac 0.125 --aug_times 7
run_cfg src-cohortA  --train_sources cohortA --aug_times 2
run_cfg src-cohortB     --train_sources cohortB    --aug_times 2
echo "[$(date '+%F %T')] ===== data-scaling sweep COMPLETE =====" | tee -a "$LOG"
