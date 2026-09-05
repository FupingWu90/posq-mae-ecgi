#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
cd $ROOT/baselines_analyses
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -f generated/lmmse_cov.npz ] || python -u lmmse_baseline.py --fit > generated/lmmse_fit.log 2>&1
[ -f generated/lmmse_lam.json ] || python -u lmmse_baseline.py --calibrate > generated/lmmse_cal.log 2>&1
for split in test_internal test_external; do
  for s in 0 1; do
    [ -s "generated/interp/${split}_shardlm_${s}.csv" ] || OMP_NUM_THREADS=3 python -u lmmse_baseline.py --eval --split $split --shard $s --nshards 2 \
      > generated/interp/${split}_lm_s${s}.log 2>&1 &
  done
done
wait
echo "LMMSE ALL DONE $(date)"
