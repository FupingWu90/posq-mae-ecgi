#!/bin/bash
# Unattended Tier-1 pilot queue (sequential, idempotent: finished runs are skipped, interrupted runs resume
# from checkpoint_last.pth).  Start detached and stop with an explicit PID:
#   cd "$ROOT" && setsid nohup bash training_sweeps/run_queue.sh > training_sweeps/logs/queue_driver.out 2>&1 < /dev/null &
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
cd "$ROOT/bsp_reconstruction" || exit 1
echo "[queue] started $(date '+%F %T') pid $$"
while true; do
  line=$(python ../training_sweeps/pilots.py next 2>>../training_sweeps/logs/queue_errors.log | tail -1)
  case "$line" in
    DONE*|STOP*|"") echo "[queue] $(date '+%F %T') ${line:-empty answer from pilots.py}"; break;;
  esac
  id=$(printf '%s' "$line" | cut -f1); run_dir=$(printf '%s' "$line" | cut -f2); args=$(printf '%s' "$line" | cut -f3)
  resume=""
  [ -f "$run_dir/checkpoints/checkpoint_last.pth" ] && resume="--resume $run_dir/checkpoints/checkpoint_last.pth"
  echo "[queue] $(date '+%F %T') start $id ${resume:+(resuming)}"
  python train.py $args $resume >> "../training_sweeps/logs/$id.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$run_dir/test_compare.json" ]; then
    n=$(cat "../training_sweeps/logs/$id.fail" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "../training_sweeps/logs/$id.fail"
    echo "[queue] $(date '+%F %T') $id FAILED (exit $rc, attempt $n)"
    [ $n -ge 2 ] && touch "../training_sweeps/failed/$id"
    sleep 30
    continue
  fi
  [ -f "$run_dir/at_rt_results_val.json" ] || python -c "
import sys, os
sys.path.insert(0, '../inverse_evaluation')
from report_at_rt import report
report('$run_dir', splits=('val',), out_json=os.path.join('$run_dir', 'at_rt_results_val.json'))" >> "../training_sweeps/logs/$id.out" 2>&1
  python ../baselines_analyses/pilot_metrics.py --run_dir "$run_dir" >> "../training_sweeps/logs/$id.out" 2>&1
  python ../training_sweeps/pilots.py summary > /dev/null 2>>../training_sweeps/logs/queue_errors.log
  echo "[queue] $(date '+%F %T') finished $id"
done
python ../training_sweeps/pilots.py summary > /dev/null 2>>../training_sweeps/logs/queue_errors.log
echo "[queue] ended $(date '+%F %T')"
