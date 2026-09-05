#!/bin/bash
# ---------------------------------------------------------------------------
# Region-restricted electrode sweep for FixdElectrodsN_Recon.
#   2 configs: --select_region front (anterior half) / back (posterior half),
#   trained cluster-greedy with the TRAINING selection restricted to one torso
#   half (channel_select.region_mask; convention = inverse_evaluation/heart_coords),
#   then scored under k-means deployment restricted to the SAME half via a
#   DK-style json (at_rt_results_deploy-cluster-kmean_reg-<region>.json), i.e.
#   EXACTLY like the selection-ablation rows (run_select_mode_sweep.sh COMMON,
#   no val override, deploy kmean through report_at_rt); comparator = the
#   existing full-torso 'greedy & k-means & 0 & base' row.
# Run names (train.build_run_name):
#   fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800_reg-front
#   fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800_reg-back
# Waits at the start until no run_seed_variance queue and no train.py job is
# using the GPU (poll every 300 s, like run_seed_variance.sh), then runs
# SERIALLY. Idempotent: skips configs whose test_compare.json exists; resumes a
# partial config from checkpoint_last.pth. Safe to re-run after a reboot.
#   nohup bash training_sweeps/run_region_sweep.sh > training_sweeps/logs/region_sweep_driver.out 2>&1 &
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
RUNS=$ROOT/runs/data-cohortA+cohortB_vs_external
LOGDIR=$ROOT/training_sweeps/logs
LOG=$LOGDIR/region_sweep.log
mkdir -p "$LOGDIR"
cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--model fixedn_ecgi_base --patch_size 1 50 --vc 16 --norm_mode global \
  --w_spatial 0 --w_temporal 0 --reg_scope recon \
  --epochs 800 --batch_size 64 --accum_iter 1 --aug_times 1 \
  --use_amp --num_workers 8 --save_every 40 --eval_every 20"
SELECT="--select_mode cluster --cluster_method greedy"
BASE="fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800"
DOWN=$ROOT/inverse_evaluation

run_cfg () {
  local name="$1"; local runname="$2"; shift 2
  local sd="$RUNS/$runname"
  if [ -f "$sd/test_compare.json" ]; then
    echo "[$(date '+%F %T')] SKIP  $name (complete)" | tee -a "$LOG"
    return 0
  fi
  local resume=""
  [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
  echo "[$(date '+%F %T')] START $name ${resume:+(resume)} -> $runname" | tee -a "$LOG"
  python train.py $COMMON $SELECT "$@" $resume >> "$LOGDIR/rg_$name.out" 2>&1
  echo "[$(date '+%F %T')] END   $name (exit $?)" | tee -a "$LOG"
}

echo "[$(date '+%F %T')] ===== region sweep started (pid $$) =====" | tee -a "$LOG"
while pgrep -f "run_seed_variance" >/dev/null || pgrep -f "train\.py .*--model" >/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] GPU free -> starting" | tee -a "$LOG"
post_cfg () {   # deploy k-means restricted to the training region (idempotent)
  local runname="$1"; local region="$2"; local sd="$RUNS/$runname"
  local out="$sd/at_rt_results_deploy-cluster-kmean_reg-$region.json"
  [ -f "$out" ] && { echo "[$(date '+%F %T')] SKIP  post $region (exists)" | tee -a "$LOG"; return 0; }
  [ -f "$sd/test_compare.json" ] || { echo "[$(date '+%F %T')] SKIP  post $region (training incomplete)" | tee -a "$LOG"; return 0; }
  echo "[$(date '+%F %T')] POST  $region -> $out" | tee -a "$LOG"
  (cd "$DOWN" && python -c "import report_at_rt as r; r.report('$sd', deploy_mode='cluster', deploy_cluster_method='kmean', deploy_region='$region', out_json='$out')") >> "$LOGDIR/rg_post_$region.out" 2>&1
  echo "[$(date '+%F %T')] POST  $region done (exit $?)" | tee -a "$LOG"
}
run_cfg reg-front "${BASE}_reg-front" --select_region front
post_cfg "${BASE}_reg-front" front
run_cfg reg-back  "${BASE}_reg-back"  --select_region back
post_cfg "${BASE}_reg-back" back
echo "[$(date '+%F %T')] ===== region sweep COMPLETE =====" | tee -a "$LOG"
