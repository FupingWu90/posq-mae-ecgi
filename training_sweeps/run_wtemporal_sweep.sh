#!/bin/bash
# ---------------------------------------------------------------------------
# w_temporal coordinate sweep for FixdElectrodsN_Recon (at the best w_spatial=0).
#   TRAIN greedy, VAL/TEST kmean.  base model, VC=16, w_spatial=0,
#   reg_scope=recon, GLOBAL norm, batch 64 / accum 1 / aug 1, 800 epochs.
#   Runs SERIALLY on the single GPU (~14h/config).  Idempotent: skips configs
#   whose test_compare.json exists; resumes a partial config from checkpoint_last.
#
#   w_temporal in {0.01, 0.1, 1.0} (mirrors the w_spatial grid).  Note the
#   temporal penalty is front-loaded: L_temporal ~1.96 at init but collapses to
#   ~0.005 at convergence, so these weights give strong EARLY pressure and mild
#   steady-state pressure.  The ws=0/wt=0 reference is the existing train-greedy
#   run's kmean deploy (identical training at ws=wt=0).
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
  --w_spatial 0 --reg_scope recon \
  --select_mode cluster --cluster_method greedy \
  --val_select_mode cluster --val_cluster_method kmean \
  --epochs 800 --batch_size 64 --accum_iter 1 --aug_times 1 \
  --use_amp --num_workers 8 --save_every 40 --eval_every 20"

run_cfg () {
  local name="$1"; local runname="$2"; shift 2
  local sd="$RUNS/$runname"
  if [ -f "$sd/test_compare.json" ]; then
    echo "[$(date '+%F %T')] SKIP  $name (complete)" | tee -a "$LOGDIR/wtemporal_sweep.log"
    return 0
  fi
  local resume=""
  [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
  echo "[$(date '+%F %T')] START $name ${resume:+(resume)}" | tee -a "$LOGDIR/wtemporal_sweep.log"
  python train.py $COMMON "$@" $resume >> "$LOGDIR/$name.out" 2>&1
  echo "[$(date '+%F %T')] END   $name (exit $?)" | tee -a "$LOGDIR/wtemporal_sweep.log"
}

BASE=fixedn_ecgi_base_vc16_cluster-greedy_val-cluster-kmean_nm-gl
TAIL=sc-recon_k6s10_ebs64_crop5000_ep800
echo "[$(date '+%F %T')] ===== w_temporal sweep started (pid $$) =====" | tee -a "$LOGDIR/wtemporal_sweep.log"
run_cfg wt0.01 ${BASE}_ws0-wt0.01_${TAIL} --w_temporal 0.01
run_cfg wt0.1  ${BASE}_ws0-wt0.1_${TAIL}  --w_temporal 0.1
run_cfg wt1    ${BASE}_ws0-wt1_${TAIL}    --w_temporal 1.0
echo "[$(date '+%F %T')] ===== w_temporal sweep COMPLETE =====" | tee -a "$LOGDIR/wtemporal_sweep.log"
