#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Follows the pilot queue: for every pilot the queue has finished, makes sure the TWO-LAYOUT validation
# evaluations exist (engine MAE under template + k-means -> <run>/val_engine.json; per-recording downstream
# CSVs under both layouts -> results/val_percase/<ID>_vc16_val_{template,kmeans}.csv), then refreshes
# results/results_log.md.  Detached, idempotent, polls every 10 min, exits when the queue has ended.
#   setsid nohup bash training_sweeps/follow_queue.sh > training_sweeps/logs/follow_queue.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=6
cd "$ROOT"
QLOG=training_sweeps/logs/queue_driver.out
echo "[follow] started $(date '+%F %T') pid $$"
while true; do
  for id in $(grep -o 'finished [A-Z0-9]*' $QLOG | awk '{print $2}' | sort -u); do
    run=$(ls -d runs/data-cohortA+cohortB_vs_external/${id}__* 2>/dev/null | head -1); [ -z "$run" ] && continue
    if [ ! -f "$run/val_engine.json" ]; then
      python baselines_analyses/val_engine_metrics.py --run_dir "$run" > training_sweeps/logs/${id}_val_engine.out 2>&1
      echo "[follow] $(date '+%F %T') $id val_engine: $(grep -E '^(template|kmeans)' training_sweeps/logs/${id}_val_engine.out | tr '\n' ' ')"
    fi
    for lay in template kmeans; do
      f=results/val_percase/${id}_vc16_val_${lay}.csv
      [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 600 ] && continue
      LAYOUT=$lay bash training_sweeps/val_percase.sh $id > /dev/null 2>&1
      echo "[follow] $(date '+%F %T') $id $lay percase: $(wc -l < "$f" 2>/dev/null) rows"
    done
  done
  python training_sweeps/pilots.py summary > /dev/null 2>&1
  grep -q '^\[queue\] .* ended' $QLOG && { echo "[follow] queue ended; finished $(date '+%F %T')"; break; }
  sleep 600
done
