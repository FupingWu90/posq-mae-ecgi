#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Dense-field validation of PosQ-MAE under both layouts: calibrate the lambda floors on validation recordings,
# evaluate 80 recordings per test cohort (2 shards), summarise.  CPU only.
#   setsid nohup bash training_sweeps/run_dense_field.sh > training_sweeps/logs/dense_field.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
for lay in template kmeans; do
  echo "[dense] $(date '+%F %T') calibrate $lay"
  python baselines_analyses/dense_field_v2.py --calibrate --device cpu --layout $lay > training_sweeps/logs/dense_cal_$lay.out 2>&1
  for split in test_internal test_external; do
    for k in 0 1; do
      nice -n 10 python baselines_analyses/dense_field_v2.py --eval --split $split --shard $k --nshards 2 --device cpu --layout $lay > training_sweeps/logs/dense_${lay}_${split}_$k.out 2>&1 &
    done
    wait
    echo "[dense] $(date '+%F %T') eval $lay $split done"
  done
  python baselines_analyses/dense_field_v2.py --summary --layout $lay > training_sweeps/logs/dense_summary_$lay.out 2>&1
done
echo "DENSE-FIELD-FINISHED $(date '+%F %T')"
