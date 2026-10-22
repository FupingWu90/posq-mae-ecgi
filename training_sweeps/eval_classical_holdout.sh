#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Classical baselines (Laplacian / TPS / LMMSE) on the hold-out split under both layouts (2026-10-20).
# The LMMSE covariance is refitted on the training recordings WITHOUT the hold-out patients
# (lmmse_cov_holdout_excl.npz); the per-N ridge weights (validation-calibrated) are reused.
# Idempotent (per-recording cache); end marker CLASSICAL-HOLDOUT-FINISHED.
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
LOG=baselines_analyses/generated/classical_template
HF=training_sweeps/private/holdout_invasive.txt
COV=$LOG/lmmse_cov_holdout_excl.npz
if [ ! -f "$COV" ]; then
  nice -n 15 python baselines_analyses/classical_baselines_template.py --fit_cov "$COV" --holdout_file "$HF" > $LOG/fit_cov_holdout.log 2>&1
  echo "[$(date '+%F %T')] covariance: $(tail -1 $LOG/fit_cov_holdout.log | cut -c1-80)"
fi
[ -f "$COV" ] || { echo "covariance fit failed"; exit 1; }
job() { lay=$1; k=$2; n=$3
  nice -n 15 python baselines_analyses/classical_baselines_template.py --layout $lay --split holdout --shard $k --nshards $n \
      --holdout_file "$HF" --cov "$COV" > $LOG/shard_${lay}_holdout_${k}.log 2>&1
  echo "[$(date '+%F %T')] done holdout $lay shard $k/$n: $(tail -1 $LOG/shard_${lay}_holdout_${k}.log | cut -c1-90)"
}
export -f job; export LOG HF COV
printf "template 0 4\ntemplate 1 4\ntemplate 2 4\ntemplate 3 4\nkmeans 0 4\nkmeans 1 4\nkmeans 2 4\nkmeans 3 4\n" | xargs -P 8 -L 1 bash -c 'job "$0" "$1" "$2"'
for lay in template kmeans; do
  nice -n 15 python baselines_analyses/classical_baselines_template.py --layout $lay --summary > $LOG/summary_${lay}_holdout.log 2>&1
  echo "[$(date '+%F %T')] summary $lay: $(grep -c . $LOG/summary_${lay}_holdout.log) lines"
done
echo "CLASSICAL-HOLDOUT-FINISHED $(date '+%F %T')"
