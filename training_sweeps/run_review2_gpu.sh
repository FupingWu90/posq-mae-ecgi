#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Reviewer-2 (user) comments 2026-10-25, GPU queue: (B8) CARNZ = final recipe + visible-set common-average reference of input and
# targets + noise augmentation (white / 5-45 Hz, SNR 5-30 dB, p = 0.5); (B9) VARN = final recipe with a random number of visible
# electrodes per batch (log-uniform 4-64; 128 would multiply the encoder attention memory 64-fold).  Then the evaluations of both models (eval_downstream + sources_rerun_v2 variants).
#   setsid nohup bash training_sweeps/run_review2_gpu.sh > training_sweeps/logs/review2_gpu.out 2>&1 < /dev/null &
set -u
cd "$ROOT"
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--model posq_ecgi_wide --batch_size 32 --accum_iter 2 --patch_size 1 50 --vc 16 --norm_mode global --w_spatial 0 --reg_warmup_epochs 100 --reg_scope recon --select_mode cluster --cluster_method greedy --val_select_mode template --aug_times 1 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20 --downstream_splits val --head subattn --inner_depth 1 --w_temporal 3"
RUNS=runs/data-cohortA+cohortB_vs_external
CAND=baselines_analyses/generated/percase_candidate; RER=baselines_analyses/generated/percase_rerun; LOG=training_sweeps/logs/review2; mkdir -p $CAND $RER $LOG
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -gt 1500 ]; do sleep 60; done
echo "[review2-gpu] GPU free, starting $(date)"
# smoke test of the new training options (1 epoch); abort the queue if it fails
(cd bsp_reconstruction && python train.py $BASE --epochs 1 --train_reref car_vis --noise_aug 0.5 --vc_random 4 64 --tag SMOKE2 > ../training_sweeps/logs/smoke2_train.out 2>&1); rc=$?
d=$(ls -d $RUNS/SMOKE2__* 2>/dev/null | head -n 1); [ -n "$d" ] && rm -rf "$d"
if [ $rc -ne 0 ]; then echo "[review2-gpu] SMOKE FAILED (exit $rc) $(date)"; exit 1; fi
echo "[review2-gpu] smoke OK $(date)"
for spec in "CARNZ|--train_reref car_vis --noise_aug 0.5 --noise_aug_snr 5 30" "VARN|--vc_random 4 64"; do
  id=${spec%%|*}; extra=${spec#*|}
  if ls -d $RUNS/${id}__* > /dev/null 2>&1 && [ -f "$(ls -d $RUNS/${id}__* | head -n 1)/test_compare.json" ]; then echo "[skip] $id trained"; else
    (cd bsp_reconstruction && python train.py $BASE $extra --tag $id > ../training_sweeps/logs/${id,,}_train.out 2>&1)
    echo "[review2-gpu] $id training finished (exit $?) $(date)"
  fi
done
MAXJ=4
waitslot() { while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done; }
ev() {  # id run vc split mode cm
  local id=$1 run=$2 vc=$3 split=$4 mode=$5 cm=$6 lay out SEL
  if [ "$mode" = template ]; then lay=template; SEL="--mode template"; else lay=kmeans; SEL="--mode cluster --cluster_method kmean"; fi
  out=$CAND/${id}_vc${vc}_${split}_${lay}.csv; [ -f "$out" ] && { echo "[skip] $out"; return; }
  waitslot
  ( nice -n 5 python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc $vc $SEL --split $split --out "$out" > "$CAND/${id}_vc${vc}_${split}_${lay}.log" 2>&1; echo "[done] eval $id vc$vc $split $lay (exit $?) $(date)" ) &
}
rr() {  # tag run vc split extra...
  local tag=$1 run=$2 vc=$3 split=$4; shift 4
  [ -f "$RER/${tag}_vc${vc}_${split}_template.csv" ] && { echo "[skip] $tag vc$vc $split"; return; }
  waitslot
  ( nice -n 5 python baselines_analyses/sources_rerun_v2.py --run_dir "$run" --split $split --vc $vc --threads 4 --tag $tag "$@" > $LOG/${tag}_vc${vc}_${split}.log 2>&1; echo "[done] $tag vc$vc $split (exit $?) $(date)" ) &
}
RC=$(ls -d $RUNS/CARNZ__* | head -n 1); RV=$(ls -d $RUNS/VARN__* | head -n 1)
for sp in test_internal test_external; do
  ev CARNZ "$RC" 16 $sp template x; ev CARNZ "$RC" 16 $sp cluster kmean
  rr carnz_base "$RC" 16 $sp; rr carnz_reref_car "$RC" 16 $sp --reref car; rr carnz_reref_ch0 "$RC" 16 $sp --reref ch0
  rr carnz_noise_emg10 "$RC" 16 $sp --noise emg 10; rr carnz_noise_white10 "$RC" 16 $sp --noise white 10; rr carnz_noise_white5 "$RC" 16 $sp --noise white 5
  for vc in 4 8 16 32 64; do ev VARN "$RV" $vc $sp template x; done
  ev VARN "$RV" 16 $sp cluster kmean
  rr varn_base "$RV" 16 $sp; rr varn_visall "$RV" 16 $sp --visible_all; rr varn_mdn "$RV" 16 $sp --model_denoise cross --sources src2_allvalid src6_mdn src4_recon_valid; rr varn_nosub "$RV" 16 $sp --nosub
done
wait
echo "[review2-gpu] ALL-DONE $(date)"
