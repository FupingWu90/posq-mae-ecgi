#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Follows the Phase 3 queue (training_sweeps/logs/phase3_driver.out): for every finished run makes sure the validation-side
# two-layout evaluations exist (engine MAE -> <run>/val_engine.json; per-recording validation CSVs under both layouts
# -> results/val_percase/<ID>_vc<N>_val_{template,kmeans}.csv) and refreshes results/phase3_log.md.  The test-cohort
# evaluations are launched by the driver itself (training_sweeps/eval_test_runs.sh), not here.  Exits when the driver has ended.
#   setsid nohup bash training_sweeps/follow_phase3.sh >> training_sweeps/logs/follow_phase3.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=6
cd "$ROOT"
PLOG=training_sweeps/logs/phase3_driver.out
echo "[follow3] started $(date '+%F %T') pid $$"
while true; do
  for id in $(grep -o 'finished [A-Z0-9]*' $PLOG | awk '{print $2}' | sort -u | tr '\n' ' '); do
    run=$(ls -d runs/data-cohortA+cohortB_vs_external/${id}__* 2>/dev/null | head -1); [ -z "$run" ] && continue
    vc=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['vc'])" "$run/args.json")
    [ -f "$run/val_engine.json" ] || { python baselines_analyses/val_engine_metrics.py --run_dir "$run" > training_sweeps/logs/${id}_val_engine.out 2>&1; echo "[follow3] $(date '+%F %T') $id val_engine done"; }
    for lay in template kmeans; do
      f=results/val_percase/${id}_vc${vc}_val_${lay}.csv
      [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 600 ] && continue
      case "$lay" in template) SEL="--mode template";; kmeans) SEL="--mode cluster --cluster_method kmean";; esac
      python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc $vc $SEL --split val --out "$f" > results/val_percase/${id}_${lay}.log 2>&1
      echo "[follow3] $(date '+%F %T') $id $lay val percase: $(wc -l < "$f" 2>/dev/null) rows"
    done
  done
  python training_sweeps/phase3.py summary > /dev/null 2>&1
  grep -q '^\[phase3\] ended' $PLOG && { echo "[follow3] phase3 ended; finished $(date '+%F %T')"; break; }
  sleep 900
done
