#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Classical baselines (Laplacian / TPS / LMMSE) under the FIXED TEMPLATE layout, doubling grid,
# both test cohorts.  CPU only (CUDA hidden), low priority, idempotent (per-recording JSON cache).
#   setsid nohup bash training_sweeps/eval_classical_template.sh > training_sweeps/logs/eval_classical_template.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
LOG=baselines_analyses/generated/classical_template
job() { split=$1; k=$2; n=$3
  nice -n 15 python baselines_analyses/classical_baselines_template.py --layout template --split $split --shard $k --nshards $n \
      > $LOG/shard_template_${split}_${k}.log 2>&1
  echo "[$(date '+%F %T')] done $split shard $k/$n: $(tail -1 $LOG/shard_template_${split}_${k}.log)"
}
export -f job; export LOG
printf "test_internal 0 2\ntest_internal 1 2\ntest_external 0 2\ntest_external 1 2\n" | xargs -P 4 -L 1 bash -c 'job "$0" "$1" "$2"'
nice -n 15 python baselines_analyses/classical_baselines_template.py --layout template --summary > $LOG/summary_template.log 2>&1
echo "CLASSICAL-TEMPLATE-FINISHED $(date '+%F %T')"
