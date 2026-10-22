#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# TEST-cohort evaluation of every finished pilot under both layouts (user request 2026-10-08: compare test results
# alongside the validation results).  Waits for the running candidate evaluation to finish, then processes the
# backlog in priority order and afterwards every newly finished pilot; exits when the queue has ended and
# nothing is left.  Detached, idempotent (eval_candidate_test.sh skips complete CSVs):
#   setsid nohup bash training_sweeps/eval_pilots_test.sh > training_sweeps/logs/eval_pilots_test.out 2>&1 < /dev/null &
cd "$ROOT"
OUT=baselines_analyses/generated/percase_candidate
QLOG=training_sweeps/logs/queue_driver.out
PRIORITY="DEC1 LOC4 LOC5 LOC3D DECH DEC2 LOC2 LOC1 MSK1 MSK3 EMB1 DEC4"
echo "[pilots-test] started $(date '+%F %T') pid $$"
until grep -q CANDIDATE-TEST-FINISHED training_sweeps/logs/eval_candidate_test.out 2>/dev/null; do sleep 300; done
complete() { id=$1; for split in test_internal test_external; do want=$([ "$split" = test_internal ] && echo 1650 || echo 1500)
  for lay in template kmeans; do f="$OUT/${id}_vc16_${split}_${lay}.csv"; { [ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$want" ]; } || return 1; done; done; return 0; }
while true; do
  finished=$(grep -o 'finished [A-Z0-9]*' $QLOG | awk '{print $2}' | sort -u | tr '\n' ' ')
  todo=""
  for id in $PRIORITY $finished; do
    echo " $todo " | grep -q " $id " && continue
    echo " $finished " | grep -q " $id " || continue
    ls -d runs/data-cohortA+cohortB_vs_external/${id}__* > /dev/null 2>&1 || continue
    complete $id || todo="$todo $id"
  done
  for id in $todo; do
    echo "[pilots-test] $(date '+%F %T') start $id"
    bash training_sweeps/eval_candidate_test.sh $id 2>&1 | grep -v "^\[skip\]"
    source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null; python training_sweeps/pilots.py summary > /dev/null 2>&1
  done
  grep -q '^\[queue\] .* ended' $QLOG && [ -z "$todo" ] && { echo "[pilots-test] queue ended, all done $(date '+%F %T')"; break; }
  sleep 900
done
