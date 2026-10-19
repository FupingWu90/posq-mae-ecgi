#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Per-vertex AT/RT maps of a run on both test cohorts under both layouts (CPU only; 2 jobs in parallel).
#   setsid nohup bash training_sweeps/extract_atmaps.sh LOC3 > training_sweeps/logs/extract_atmaps_LOC3.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
id=$1; run=$(ls -d runs/data-cohortA+cohortB_vs_external/${id}__* | head -1)
job() { split=$1; lay=$2; nice -n 10 python inverse_evaluation/at_maps.py --run_dir "$run" --split $split --layout $lay --threads 6 > training_sweeps/logs/atmaps_${id}_${split}_${lay}.out 2>&1; echo "[$(date '+%F %T')] done $split $lay: $(tail -1 training_sweeps/logs/atmaps_${id}_${split}_${lay}.out)"; }
export -f job; export run id
printf "test_internal template\ntest_internal kmeans\ntest_external template\ntest_external kmeans\n" | xargs -P 2 -L 1 bash -c 'job "$0" "$1"'
echo "ATMAPS-FINISHED $id $(date '+%F %T')"
