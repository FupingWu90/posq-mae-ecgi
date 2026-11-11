#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# After the LOC3 map extraction, extract the maps of the attribution ablations (DEC1: no local head; LOC4: no smoothness loss)
# and then compute the plausibility metric for all three.  CPU only.
cd "$ROOT"
until grep -q "ATMAPS-FINISHED LOC3" training_sweeps/logs/extract_atmaps_LOC3.out 2>/dev/null; do sleep 120; done
for id in DEC1 LOC4; do bash training_sweeps/extract_atmaps.sh $id > training_sweeps/logs/extract_atmaps_$id.out 2>&1; done
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null; export OMP_NUM_THREADS=4
for id in LOC3 DEC1 LOC4; do python baselines_analyses/plausibility.py --id $id > training_sweeps/logs/plausibility_$id.out 2>&1; done
echo "PLAUSIBILITY-CHAIN-FINISHED $(date '+%F %T')"
