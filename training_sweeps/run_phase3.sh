#!/bin/bash
# Phase 3 queue (sequential, idempotent, resumes from checkpoint_last.pth).  Waits until the pilot queue has ended and
# the training length is decided (EPL3), then trains every run of training_sweeps/phase3.py; after each run it starts the
# two-layout test-cohort evaluation in the background (CPU) and refreshes results/phase3_log.md.
#   setsid nohup bash training_sweeps/run_phase3.sh >> training_sweeps/logs/phase3_driver.out 2>&1 < /dev/null &
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
cd "$ROOT/bsp_reconstruction" || exit 1
echo "[phase3] started $(date '+%F %T') pid $$"
until grep -q '^\[queue\] ended' ../training_sweeps/logs/queue_driver.out 2>/dev/null && ! pgrep -f "^python train.py" > /dev/null; do sleep 600; done
echo "[phase3] pilot queue ended; starting $(date '+%F %T')"
while true; do
  line=$(python ../training_sweeps/phase3.py next 2>>../training_sweeps/logs/phase3_errors.log | tail -1)
  case "$line" in
    DONE*|"") echo "[phase3] $(date '+%F %T') ${line:-empty answer}"; break;;
    WAIT*) echo "[phase3] $(date '+%F %T') $line"; sleep 1800; continue;;
  esac
  id=$(printf '%s' "$line" | cut -f1); run_dir=$(printf '%s' "$line" | cut -f2); args=$(printf '%s' "$line" | cut -f3)
  resume=""; [ -f "$run_dir/checkpoints/checkpoint_last.pth" ] && resume="--resume $run_dir/checkpoints/checkpoint_last.pth"
  echo "[phase3] $(date '+%F %T') start $id ${resume:+(resuming)}"
  python train.py $args $resume >> "../training_sweeps/logs/$id.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$run_dir/test_compare.json" ]; then
    n=$(cat "../training_sweeps/logs/$id.fail" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "../training_sweeps/logs/$id.fail"
    echo "[phase3] $(date '+%F %T') $id FAILED (exit $rc, attempt $n)"; [ $n -ge 2 ] && touch "../training_sweeps/failed/$id"; sleep 30; continue
  fi
  python ../baselines_analyses/val_engine_metrics.py --run_dir "$run_dir" >> "../training_sweeps/logs/$id.out" 2>&1
  ( cd "$ROOT" && setsid nohup bash training_sweeps/eval_test_runs.sh "$id" >> "training_sweeps/logs/eval_test_$id.out" 2>&1 < /dev/null & )
  python ../training_sweeps/phase3.py summary > /dev/null 2>>../training_sweeps/logs/phase3_errors.log
  echo "[phase3] $(date '+%F %T') finished $id"
done
python ../training_sweeps/phase3.py summary > /dev/null 2>>../training_sweeps/logs/phase3_errors.log
echo "[phase3] ended $(date '+%F %T')"
