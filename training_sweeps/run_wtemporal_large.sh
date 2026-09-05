#!/bin/bash
# ---------------------------------------------------------------------------
# w_temporal STRONG-regime probe (extends {0.01,0.1,1.0} upward), WITH a
# regularizer-weight warmup so the large raw temporal penalty at init does not
# dominate early training.
#   wt in {3, 10}, ws=0, reg_warmup_epochs=100 (ramp wt 0->target over first
#   100 of 800 epochs; the LR warmup is 80 epochs).  Otherwise identical to the
#   earlier w_temporal sweep: train greedy / val-test kmean, base VC=16,
#   reg_scope=recon, 800 epochs, ~14h/config.  Idempotent; resumes.
#
#   NOTE: the earlier wt in {0.01,0.1,1.0} runs used NO warmup (rw0). At those
#   mild weights the early penalty is small so warmup barely matters, but the
#   comparison to wt=1 carries a small warmup confound -- flagged in analysis.
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
  --w_spatial 0 --reg_scope recon --reg_warmup_epochs 100 \
  --select_mode cluster --cluster_method greedy \
  --val_select_mode cluster --val_cluster_method kmean \
  --epochs 800 --batch_size 64 --accum_iter 1 --aug_times 1 \
  --use_amp --num_workers 8 --save_every 40 --eval_every 20"

run_cfg () {
  local name="$1"; local runname="$2"; shift 2
  local sd="$RUNS/$runname"
  if [ -f "$sd/test_compare.json" ]; then
    echo "[$(date '+%F %T')] SKIP  $name (complete)" | tee -a "$LOGDIR/wtemporal_large.log"
    return 0
  fi
  local resume=""
  [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
  echo "[$(date '+%F %T')] START $name ${resume:+(resume)}" | tee -a "$LOGDIR/wtemporal_large.log"
  python train.py $COMMON "$@" $resume >> "$LOGDIR/$name.out" 2>&1
  echo "[$(date '+%F %T')] END   $name (exit $?)" | tee -a "$LOGDIR/wtemporal_large.log"
}

BASE=fixedn_ecgi_base_vc16_cluster-greedy_val-cluster-kmean_nm-gl
TAIL=sc-recon_k6s10_ebs64_crop5000_ep800
echo "[$(date '+%F %T')] ===== w_temporal STRONG probe (warmup) started (pid $$) =====" | tee -a "$LOGDIR/wtemporal_large.log"
run_cfg wt3  ${BASE}_ws0-wt3-rw100_${TAIL}  --w_temporal 3.0
run_cfg wt10 ${BASE}_ws0-wt10-rw100_${TAIL} --w_temporal 10.0
echo "[$(date '+%F %T')] ===== w_temporal STRONG probe COMPLETE =====" | tee -a "$LOGDIR/wtemporal_large.log"
