#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Ground-truth evaluation on the SIMULATED test cohort (inverse_evaluation/eval_sim_gt.py, 408 recordings on held-out
# geometries) for the simulation group of Phase 3 under BOTH deployment layouts: LOC3 (clinical only), P3SIM
# (simulated only), P3SIM2R (simulated -> clinical), P3SIMR (simulated + clinical).  Waits for each run's
# test_compare.json; idempotent (skips complete CSVs); GPU inference next to the running training.
# Output: baselines_analyses/generated/sim_gt/<ID>_vc16_{template,kmeans}.csv (+ .log).  Marker: SIM-GT-FINISHED.
#   setsid nohup bash training_sweeps/eval_sim_gt_runs.sh >> training_sweeps/logs/eval_sim_gt_runs.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=6
cd "$ROOT" || exit 1
OUT=baselines_analyses/generated/sim_gt; mkdir -p $OUT
IDS=${@:-LOC3 P3SIM P3SIM2R P3SIMR}
echo "[simgt] started $(date '+%F %T') pid $$ ids: $IDS"
for id in $IDS; do
  until run=$(ls -d runs/data-cohortA+cohortB_vs_external/${id}__* 2>/dev/null | head -1) && [ -n "$run" ] && [ -f "$run/test_compare.json" ]; do sleep 600; done
  for lay in template kmeans; do
    f=$OUT/${id}_vc16_${lay}.csv
    [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 1200 ] && continue
    case "$lay" in template) SEL="--mode template";; kmeans) SEL="--mode cluster --cluster_method kmean";; esac
    python inverse_evaluation/eval_sim_gt.py --run_dir "$run" --n 1000000 --vc 16 $SEL --seed 42 --out "$f" > $OUT/${id}_${lay}.log 2>&1
    echo "[simgt] $(date '+%F %T') $id $lay: $(wc -l < "$f" 2>/dev/null) rows (exit $?)"
  done
done
echo "SIM-GT-FINISHED $(date '+%F %T')"
