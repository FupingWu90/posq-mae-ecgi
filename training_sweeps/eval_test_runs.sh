#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Adoption-gate evaluation (Phase 2): per-recording downstream metrics of the candidate configuration(s) on BOTH
# test cohorts under BOTH deployment layouts (fixed template, per-recording k-means).  No training; GPU inference
# is light, the MFS inverse (CPU) dominates.  Idempotent; 3 jobs in parallel; low priority.
#   usage: setsid nohup bash training_sweeps/eval_test_runs.sh LOC3 LOC3B > training_sweeps/logs/eval_candidate_test.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=6
cd "$ROOT"
OUT=baselines_analyses/generated/percase_candidate
job() { id=$1; split=$2; lay=$3
  run=$(ls -d runs/data-cohortA+cohortB_vs_external/${id}__* | head -1)
  vc=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['vc'])" "$run/args.json")
  out="$OUT/${id}_vc${vc}_${split}_${lay}.csv"
  want=$([ "$split" = test_internal ] && echo 1650 || echo 1500)
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$want" ]; then echo "[skip] $out"; return; fi
  case "$lay" in template) SEL="--mode template";; kmeans) SEL="--mode cluster --cluster_method kmean";; esac
  nice -n 10 python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc $vc $SEL --split $split --out "$out" > "$OUT/${id}_vc${vc}_${split}_${lay}.log" 2>&1
  echo "[$(date '+%F %T')] done $id $split $lay ($(grep -c FAILED "$OUT/${id}_vc${vc}_${split}_${lay}.log") failed, $(wc -l < "$out") rows)"
}
export -f job; export OUT
{ for id in "$@"; do for split in test_internal test_external; do for lay in template kmeans; do echo "$id $split $lay"; done; done; done; } \
  | xargs -P 3 -L 1 bash -c 'job "$0" "$1" "$2"'
echo "CANDIDATE-TEST-FINISHED $(date '+%F %T')"
