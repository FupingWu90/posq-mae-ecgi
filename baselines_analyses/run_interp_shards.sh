#!/bin/bash
# 3 shards per split, both splits in parallel (6 CPU procs, GPU untouched)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
cd $ROOT/baselines_analyses
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
for split in test_internal test_external; do
  for s in 0 1 2; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python interp_baseline.py \
      --split $split --shard $s --nshards 3 \
      > generated/interp/${split}_s${s}.log 2>&1 &
  done
done
wait
echo "INTERP SHARDS DONE $(date)"
