#!/bin/bash
# ---------------------------------------------------------------------------
# Learned-baseline trainings over the FULL VC grid {4..72} (user request):
#   pe-index : ST-MEM-style per-channel-index embedding in place of the PPE
#   mcma     : MCMA-style 1-D U-Net conv autoencoder
# Interleaved by VC priority (16 first, then 4/64/32/8/24/48/72) so both
# baselines complete N-by-N and the comparison tables can grow progressively.
# After each training, the per-recording downstream CSVs for the paired
# significance tests are produced immediately (eval_downstream, kmean deploy).
# Idempotent; resumes checkpoint_last; no watchdog (re-run by hand after the
# weekly reboot).  ~16 x 19 h serial if run from scratch.
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
set -u
CODE=$ROOT/bsp_reconstruction
DOWN=$ROOT/inverse_evaluation
RUNS=$ROOT/runs/data-cohortA+cohortB_vs_external
OUT=$ROOT/baselines_analyses/generated/vc_percase
LOGDIR=$ROOT/training_sweeps/logs
LOG=$LOGDIR/baseline_models.log
cd "$CODE" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--batch_size 32 --accum_iter 2 --patch_size 1 50 --norm_mode global \
  --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 --reg_scope recon \
  --select_mode cluster --cluster_method greedy \
  --val_select_mode cluster --val_cluster_method kmean \
  --aug_times 1 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20"
BASE="cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800"

echo "[$(date '+%F %T')] ===== baseline-model VC-grid queue started (pid $$) =====" | tee -a "$LOG"
while pgrep -f "run_data_scaling" >/dev/null || pgrep -f "train\.py .*--model" >/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] GPU free -> starting" | tee -a "$LOG"

run_cfg () {
  local model="$1" vc="$2" tag="$3" csvtag="$4"; shift 4
  local extra="$*"
  local sd="$RUNS/${model}_vc${vc}_${BASE}${tag}"
  if [ ! -f "$sd/test_compare.json" ]; then
    local resume=""; [ -f "$sd/checkpoints/checkpoint_last.pth" ] && resume="--resume $sd/checkpoints/checkpoint_last.pth"
    echo "[$(date '+%F %T')] START ${csvtag} vc=$vc ${resume:+(resume)}" | tee -a "$LOG"
    python train.py --model "$model" --vc "$vc" $COMMON $extra $resume >> "$LOGDIR/bl_${csvtag}_vc${vc}.out" 2>&1
    echo "[$(date '+%F %T')] END   ${csvtag} vc=$vc (exit $?)" | tee -a "$LOG"
  else
    echo "[$(date '+%F %T')] SKIP  ${csvtag} vc=$vc (complete)" | tee -a "$LOG"
  fi
  # per-recording downstream CSVs for the paired tests (fast; skip if present)
  if [ -f "$sd/test_compare.json" ]; then
    for split in test_internal test_external; do
      [ -s "$OUT/${csvtag}_vc${vc}_${split}.csv" ] || (cd "$DOWN" && OMP_NUM_THREADS=4 \
        python eval_downstream.py --run_dir "$sd" --n 1000000 --vc "$vc" \
          --mode cluster --cluster_method kmean --split "$split" \
          --out "$OUT/${csvtag}_vc${vc}_${split}.csv" \
          > "$OUT/${csvtag}_vc${vc}_${split}.log" 2>&1)
    done
    echo "[$(date '+%F %T')]       ${csvtag} vc=$vc percase done" | tee -a "$LOG"
  fi
}

for vc in 16 4 64 32 8 24 48 72; do
  run_cfg fixedn_ecgi_wide "$vc" _pe-index peindex --pos_embed index
  if python -c "import models_fixedn; models_fixedn.mcma_unet" 2>/dev/null; then
    run_cfg mcma_unet "$vc" "" mcma
  fi
done
echo "[$(date '+%F %T')] ===== baseline-model VC-grid queue COMPLETE =====" | tee -a "$LOG"
