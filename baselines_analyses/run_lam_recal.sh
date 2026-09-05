#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
cd $ROOT/baselines_analyses
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
OMP_NUM_THREADS=4 python lam_recal.py --calibrate --n_recs 20 \
  > generated/lam_recal/calibrate.log 2>&1
for split in test_internal test_external; do
  for s in 0 1; do
    OMP_NUM_THREADS=3 python lam_recal.py --eval --split $split \
      --shard $s --nshards 2 > generated/lam_recal/${split}_s${s}.log 2>&1 &
  done
done
wait
echo "LAM RECAL DONE $(date)"
