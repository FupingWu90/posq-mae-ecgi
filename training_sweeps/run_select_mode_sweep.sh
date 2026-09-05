#!/bin/bash
# ---------------------------------------------------------------------------
# select_mode sweep for FixdElectrodsN_Recon.
#   4 configs: cluster-greedy, cluster-kmean, patch, random
#   base model, VC=16, regularizers OFF (isolate select_mode),
#   GLOBAL (ratio-preserving) normalization, batch 64 / accum 1 / aug 1, 800 epochs.
# Runs SERIALLY on the single GPU (~15 iters/epoch, ~1 min/epoch -> ~14h/config).
# Idempotent: skips configs whose test_compare.json exists; resumes a partial
# config from checkpoint_last.pth. Safe to re-run after a reboot.
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
RUNS=$ROOT/runs/data-cohortA+cohortB_vs_external
LOGDIR=$ROOT/training_sweeps/logs
mkdir -p "$LOGDIR"
cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--model fixedn_ecgi_base --patch_size 1 50 --vc 16 --norm_mode global \
  --w_spatial 0 --w_temporal 0 --reg_scope recon \
  --epochs 800 --batch_size 64 --accum_iter 1 --aug_times 1 \
  --use_amp --num_workers 8 --save_every 40 --eval_every 20"

run_cfg () {
  local name="$1"; local runname="$2"; shift 2
  local sd="$RUNS/$runname"
  if [ -f "$sd/test_compare.json" ]; then
    echo "[$(date '+%F %T')] SKIP  $name (complete)" | tee -a "$LOGDIR/sweep.log"
    return 0
  fi
  local resume=""
  [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
  echo "[$(date '+%F %T')] START $name ${resume:+(resume)}" | tee -a "$LOGDIR/sweep.log"
  python train.py $COMMON "$@" $resume >> "$LOGDIR/$name.out" 2>&1
  echo "[$(date '+%F %T')] END   $name (exit $?)" | tee -a "$LOGDIR/sweep.log"
}

echo "[$(date '+%F %T')] ===== select_mode sweep started (pid $$) =====" | tee -a "$LOGDIR/sweep.log"
run_cfg cluster-greedy fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800 --select_mode cluster --cluster_method greedy
run_cfg cluster-kmean  fixedn_ecgi_base_vc16_cluster-kmean_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800  --select_mode cluster --cluster_method kmean
run_cfg patch          fixedn_ecgi_base_vc16_patch_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800          --select_mode patch
run_cfg random         fixedn_ecgi_base_vc16_random_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800         --select_mode random
echo "[$(date '+%F %T')] ===== select_mode sweep COMPLETE =====" | tee -a "$LOGDIR/sweep.log"
