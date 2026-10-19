#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# S6 lambda-fairness under the fixed template layout: calibrate per-N floor (validation) -> evaluate both test cohorts
# (3 shards each) -> table.  CPU only, low priority, idempotent.
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
cd "$ROOT"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
LAY=${1:-template}
echo "[lam] started $(date '+%F %T') layout $LAY"
[ -f baselines_analyses/generated/lam_recal_$LAY/lam_per_n.json ] || nice -n 15 python baselines_analyses/lam_recal_v2.py --layout $LAY --calibrate
for split in test_internal test_external; do
  for k in 0 1 2; do nice -n 15 python baselines_analyses/lam_recal_v2.py --layout $LAY --eval --split $split --shard $k --nshards 3 > training_sweeps/logs/lam_recal_${LAY}_${split}_$k.out 2>&1 & done
  wait
done
python baselines_analyses/lam_recal_v2.py --layout $LAY --table
echo "LAM-RECAL-FINISHED $LAY $(date '+%F %T')"
