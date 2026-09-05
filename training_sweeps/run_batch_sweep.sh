#!/bin/bash
# ---------------------------------------------------------------------------
# Effective-batch sweep at the TUNED config (ws=0, wt=3, rw100, reg_scope=recon,
# greedy train / kmean deploy, base VC=16).  eff in {128, 256}; baseline is the
# existing eff=64 run (wt3-rw100-recon, aug_times=1, lr=1e-3).
#
# Physical batch fixed at 64 (batch 128 OOMs); effective batch via gradient
# accumulation.  LR is NOT scaled (stays 1e-3); instead aug_times is scaled with
# the batch so OPTIMIZER UPDATES per epoch stay constant:
#     updates/epoch = n_rec * aug_times / eff_batch = const  <=>  aug_times = accum_iter
#   eff128 -> batch 64, accum 2, aug_times 2   (updates/epoch = baseline)
#   eff256 -> batch 64, accum 4, aug_times 4   (updates/epoch = baseline)
# So the effective-batch effect is isolated at fixed LR AND fixed update count;
# each update just sees a larger, more-augmented batch.
# NOTE: iterations/epoch scale with aug_times -> eff128 ~2x, eff256 ~4x wall-clock.
# Explicit --run_name carries the aug tag (aug_times is not in build_run_name).
# Runs SERIALLY; idempotent; resumes checkpoint_last.
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
RUNS=$ROOT/runs/data-cohortA+cohortB_vs_external
LOGDIR=$ROOT/training_sweeps/logs
mkdir -p "$LOGDIR"; cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--model fixedn_ecgi_base --patch_size 1 50 --vc 16 --norm_mode global \
  --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 --reg_scope recon \
  --select_mode cluster --cluster_method greedy \
  --val_select_mode cluster --val_cluster_method kmean \
  --batch_size 64 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20"

run_cfg () {
  local name="$1"; local runname="$2"; shift 2
  local sd="$RUNS/$runname"
  if [ -f "$sd/test_compare.json" ]; then
    echo "[$(date '+%F %T')] SKIP  $name (complete)" | tee -a "$LOGDIR/batch_sweep.log"; return 0
  fi
  local resume=""; [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
  echo "[$(date '+%F %T')] START $name ${resume:+(resume)}" | tee -a "$LOGDIR/batch_sweep.log"
  python train.py $COMMON --run_name "$runname" "$@" $resume >> "$LOGDIR/$name.out" 2>&1
  echo "[$(date '+%F %T')] END   $name (exit $?)" | tee -a "$LOGDIR/batch_sweep.log"
}

B=fixedn_ecgi_base_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10
echo "[$(date '+%F %T')] ===== batch sweep started (pid $$) =====" | tee -a "$LOGDIR/batch_sweep.log"
run_cfg ebs128 ${B}_ebs128-aug2_crop5000_ep800 --accum_iter 2 --aug_times 2
run_cfg ebs256 ${B}_ebs256-aug4_crop5000_ep800 --accum_iter 4 --aug_times 4
echo "[$(date '+%F %T')] ===== batch sweep COMPLETE =====" | tee -a "$LOGDIR/batch_sweep.log"
