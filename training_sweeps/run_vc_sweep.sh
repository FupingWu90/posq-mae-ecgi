#!/bin/bash
# ---------------------------------------------------------------------------
# VC (fixed visible-channel count) sweep with the ADOPTED wide model, at the
# TUNED config (ws=0, wt=3, rw100, reg_scope=recon, greedy train / kmean deploy).
# VC in {4,8,24,32,48,64}; VC=16 is already done (the wide baseline run).
# wide fits at batch 32 for all VC (VC=64 -> 27.6GB) -> eff64 via accum 2.
# The src1 (VC-only) downstream caches for these VC are pre-built, so each
# post-training HSP report is fast.  Ordered low->high (few-channel end first).
# Runs SERIALLY (~19h/VC); idempotent; resumes checkpoint_last.
#   NOTE: with the Aug 13 reboot, the last VC(s) may be interrupted -> resume
#   manually by re-running this script (idempotent).
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
RUNS=$ROOT/runs/data-cohortA+cohortB_vs_external
LOGDIR=$ROOT/training_sweeps/logs
mkdir -p "$LOGDIR"; cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--model fixedn_ecgi_wide --batch_size 32 --accum_iter 2 \
  --patch_size 1 50 --norm_mode global \
  --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 --reg_scope recon \
  --select_mode cluster --cluster_method greedy \
  --val_select_mode cluster --val_cluster_method kmean \
  --aug_times 1 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20"

run_vc () {
  local vc="$1"
  local runname="fixedn_ecgi_wide_vc${vc}_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800"
  local sd="$RUNS/$runname"
  if [ -f "$sd/test_compare.json" ]; then
    echo "[$(date '+%F %T')] SKIP  VC=$vc (complete)" | tee -a "$LOGDIR/vc_sweep.log"; return 0
  fi
  local resume=""; [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
  echo "[$(date '+%F %T')] START VC=$vc ${resume:+(resume)}" | tee -a "$LOGDIR/vc_sweep.log"
  python train.py --vc "$vc" $COMMON $resume >> "$LOGDIR/vc${vc}.out" 2>&1
  echo "[$(date '+%F %T')] END   VC=$vc (exit $?)" | tee -a "$LOGDIR/vc_sweep.log"
}

echo "[$(date '+%F %T')] ===== VC sweep (wide) started (pid $$) =====" | tee -a "$LOGDIR/vc_sweep.log"
for vc in 4 8 24 32 48 64; do run_vc "$vc"; done
echo "[$(date '+%F %T')] ===== VC sweep COMPLETE =====" | tee -a "$LOGDIR/vc_sweep.log"
