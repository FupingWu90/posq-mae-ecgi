#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Difference-matching loss comparison (reviewer minor point 5; user 2026-10-23: run it, keep the result as backup material).
# Waits for the half-A training (OFFGRIDA) to free the GPU, then trains DIFFL (LOC3 recipe, smoothness penalty REPLACED by the
# difference-matching loss, w_diff = 3) and DIFFLB (added to the penalty, w_t = 3 + w_diff = 3), evaluates both layouts on both
# test cohorts (generated/percase_candidate/<ID>_vc16_<split>_<layout>.csv) and runs the paired comparison vs LOC3.
#   setsid nohup bash training_sweeps/run_diffloss.sh > training_sweeps/logs/diffloss_driver.out 2>&1 < /dev/null &
set -u
cd "$ROOT"
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  RUNA=$(ls -d runs/data-*/OFFGRIDA__*_reg-subA 2>/dev/null | head -n 1)
  if [ -n "$RUNA" ] && [ -f "$RUNA/test_compare.json" ]; then break; fi
  sleep 300
done
echo "[diffloss] GPU free, starting $(date)"
BASE="--model posq_ecgi_wide --batch_size 32 --accum_iter 2 --patch_size 1 50 --vc 16 --norm_mode global --w_spatial 0 --reg_warmup_epochs 100 --reg_scope recon --select_mode cluster --cluster_method greedy --val_select_mode template --aug_times 1 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20 --downstream_splits val --head subattn --inner_depth 1"
OUT=baselines_analyses/generated/percase_candidate; mkdir -p $OUT
for spec in "DIFFL|--w_temporal 0 --w_diff 3" "DIFFLB|--w_temporal 3 --w_diff 3"; do
  id=${spec%%|*}; extra=${spec#*|}
  if ! ls -d runs/data-*/${id}__* >/dev/null 2>&1 || [ ! -f "$(ls -d runs/data-*/${id}__* | head -n 1)/test_compare.json" ]; then
    (cd bsp_reconstruction && python train.py $BASE $extra --tag $id > ../training_sweeps/logs/${id,,}_train.out 2>&1)
    echo "[diffloss] $id training finished (exit $?) $(date)"
  fi
  run=$(ls -d runs/data-*/${id}__* | head -n 1)
  for split in test_internal test_external; do
    for lay in template kmeans; do
      case "$lay" in template) SEL="--mode template";; kmeans) SEL="--mode cluster --cluster_method kmean";; esac
      out=$OUT/${id}_vc16_${split}_${lay}.csv
      [ -f "$out" ] || (nice -n 10 python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc 16 $SEL --split $split --out "$out" > "$OUT/${id}_vc16_${split}_${lay}.log" 2>&1; echo "[done] $id $split $lay $(date)") &
    done
  done
  wait
done
for split in test_internal test_external; do for lay in template kmeans; do
  python baselines_analyses/compare_pilots.py --split $split --layout $lay LOC3 DIFFL DIFFLB >> training_sweeps/logs/diffloss_compare.out 2>&1
done; done
echo "[diffloss] ALL-DONE $(date)"
