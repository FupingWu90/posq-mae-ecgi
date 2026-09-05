#!/bin/bash
# ---------------------------------------------------------------------------
# Model-architecture sweep: controlled single-axis variants around `base`,
# at the TUNED config (ws=0, wt=3, rw100, reg_scope=recon, greedy train /
# kmean deploy, VC=16, patch 1x50).  All at EFFECTIVE BATCH 64 (the tuned
# batch), aug 1, lr 1e-3 -> updates/epoch identical to base; only the
# architecture changes.  Baseline is the existing `base` run.
#   WIDTH axis: narrow (embed 64), wide (embed 256, batch 32 accum 2)
#   DEPTH axis: shallow (enc4/dec2), deep (enc9/dec5)
# Ordered so the WIDTH axis completes before the ~Aug6 02:08 reboot; the DEPTH
# axis runs after (resume manually).  Idempotent; resumes checkpoint_last.
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
RUNS=$ROOT/runs/data-cohortA+cohortB_vs_external
LOGDIR=$ROOT/training_sweeps/logs
mkdir -p "$LOGDIR"; cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--patch_size 1 50 --vc 16 --norm_mode global \
  --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 --reg_scope recon \
  --select_mode cluster --cluster_method greedy \
  --val_select_mode cluster --val_cluster_method kmean \
  --aug_times 1 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20"

run_cfg () {   # name  model  batch  accum
  local name="$1" model="$2" bs="$3" acc="$4"
  local eff=$((bs*acc))
  local runname="${model}_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs${eff}_crop5000_ep800"
  local sd="$RUNS/$runname"
  if [ -f "$sd/test_compare.json" ]; then
    echo "[$(date '+%F %T')] SKIP  $name (complete)" | tee -a "$LOGDIR/model_sweep.log"; return 0
  fi
  local resume=""; [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
  echo "[$(date '+%F %T')] START $name ($model, batch $bs x accum $acc = eff $eff) ${resume:+(resume)}" | tee -a "$LOGDIR/model_sweep.log"
  python train.py --model "$model" --batch_size "$bs" --accum_iter "$acc" $COMMON $resume >> "$LOGDIR/$name.out" 2>&1
  echo "[$(date '+%F %T')] END   $name (exit $?)" | tee -a "$LOGDIR/model_sweep.log"
}

echo "[$(date '+%F %T')] ===== model sweep started (pid $$) =====" | tee -a "$LOGDIR/model_sweep.log"
# priority: the "more capacity" direction first (does base underfit?)
run_cfg wide    fixedn_ecgi_wide    32 2     # width x2
run_cfg deep    fixedn_ecgi_deep    64 1     # depth 9/5
run_cfg narrow  fixedn_ecgi_narrow  64 1     # width x0.5
run_cfg shallow fixedn_ecgi_shallow 64 1     # depth 4/2
echo "[$(date '+%F %T')] ===== model sweep COMPLETE =====" | tee -a "$LOGDIR/model_sweep.log"
