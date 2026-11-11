#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=6
cd "$ROOT"
for m in kmean greedy; do
  for s in test_internal test_external; do
    nice -n 10 python baselines_analyses/diag_geometry_only_selection.py --split $s --method $m > results/diagnostics_20261004/geomonly_${m}_vc16_$s.log 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] $m finished"
done
echo GEOMONLY-FINISHED
