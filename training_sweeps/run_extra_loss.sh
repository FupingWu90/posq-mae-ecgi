#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Emulated heavy channel loss (npj DM R7, 2026-10-20): per-recording downstream metrics of the final model at N=16 when an
# extra 25 / 50 / 75 % of the ACTIVE channels of every test recording are flagged missing (seed 1); sources = VC-only@16,
# Ours@16, the reduced vest's own all-valid inverse ('src2_reduced'); references = cached commercial + original all-valid.
# Internal: both layouts; external: template layout.  3 jobs in parallel, low priority, idempotent.
#   usage: setsid nohup bash training_sweeps/run_extra_loss.sh > training_sweeps/logs/extra_loss.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=6
cd "$ROOT"
OUT=baselines_analyses/generated/percase_loss
run=$(ls -d runs/data-cohortA+cohortB_vs_external/LOC3__* | head -1)
job() { split=$1; lay=$2; loss=$3
  out="$OUT/LOC3_vc16_${split}_${lay}_loss${loss}s1.csv"
  want=$([ "$split" = test_internal ] && echo 2200 || echo 2000)
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$want" ]; then echo "[skip] $out"; return; fi
  case "$lay" in template) SEL="--mode template";; kmeans) SEL="--mode cluster --cluster_method kmean";; esac
  nice -n 10 python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc 16 $SEL --split $split \
      --extra_loss 0.$loss --extra_loss_seed 1 --out "$out" > "$OUT/LOC3_vc16_${split}_${lay}_loss${loss}s1.log" 2>&1
  echo "[$(date '+%F %T')] done $split $lay loss=$loss ($(grep -c FAILED "$OUT/LOC3_vc16_${split}_${lay}_loss${loss}s1.log") failed, $(wc -l < "$out") rows)"
}
export -f job; export OUT run
{ for loss in 25 50 75; do echo "test_internal template $loss"; done
  for loss in 25 50 75; do echo "test_external template $loss"; done
  for loss in 25 50 75; do echo "test_internal kmeans $loss"; done; } | xargs -P 3 -L 1 bash -c 'job "$0" "$1" "$2"'
echo "EXTRA-LOSS-FINISHED $(date '+%F %T')"
