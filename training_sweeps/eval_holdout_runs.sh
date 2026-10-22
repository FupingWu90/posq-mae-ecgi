#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Hold-out (invasive-comparison) patients: per-recording downstream evaluation of a run trained with
# --holdout_file, under both deployment layouts (2026-10-20; mirrors training_sweeps/eval_test_runs.sh).
# usage: bash training_sweeps/eval_holdout_runs.sh ID [ID...]   -> baselines_analyses/generated/percase_holdout/
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=6
cd "$ROOT"
OUT=baselines_analyses/generated/percase_holdout
mkdir -p "$OUT"
job() { id=$1; lay=$2
  run=$(ls -d runs/data-cohortA+cohortB_vs_external/${id}__* | head -1)
  vc=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['vc'])" "$run/args.json")
  out="$OUT/${id}_vc${vc}_holdout_${lay}.csv"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 2 ]; then echo "[skip] $out"; return; fi
  case "$lay" in template) SEL="--mode template";; kmeans) SEL="--mode cluster --cluster_method kmean";; esac
  nice -n 10 python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc $vc $SEL --split holdout --out "$out" > "$OUT/${id}_vc${vc}_holdout_${lay}.log" 2>&1
  echo "[$(date '+%F %T')] done $id holdout $lay ($(grep -c FAILED "$OUT/${id}_vc${vc}_holdout_${lay}.log") failed, $( [ -f "$out" ] && wc -l < "$out" || echo 0) rows)"
}
export -f job; export OUT
{ for id in "$@"; do for lay in template kmeans; do echo "$id $lay"; done; done; } \
  | xargs -P 2 -L 1 bash -c 'job "$0" "$1"'
echo "HOLDOUT-EVAL-FINISHED $(date '+%F %T')"
