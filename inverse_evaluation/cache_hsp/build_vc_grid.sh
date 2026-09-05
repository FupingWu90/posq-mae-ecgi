#!/bin/bash
# Pre-build the src1 (VC-only) HSP cache for a grid of VC values, at the deploy
# selection (cluster-kmean, seed 42), so future VC-model downstream evals are
# ready.  src1 is MODEL-INDEPENDENT -> built without any VC models.  refs are
# shared (already cached) and skipped.  Pure CPU; capped threads + 2-way
# concurrency so it does not starve the GPU sweep's dataloaders.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repository root
set -u
DL=$ROOT/inverse_evaluation
LOGDIR=$DL/cache_hsp/logs; mkdir -p "$LOGDIR"
cd "$DL" || exit 1
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
RUN=$ROOT/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800
VCS="4 8 24 32 48 64"
MAXPAR=2

build_vc () {
  local vc=$1
  for split in test_internal test_external; do
    echo "[$(date '+%F %T')] VC=$vc $split START" >> "$LOGDIR/vcgrid.log"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python cache_references.py --run_dir "$RUN" \
      --vc "$vc" --mode cluster --cluster_method kmean --split "$split" \
      >> "$LOGDIR/vc${vc}_kmean.out" 2>&1
  done
  echo "[$(date '+%F %T')] VC=$vc DONE (int=$(ls cache_hsp/test_internal/vc${vc}_cluster-kmean_s42/*.npz 2>/dev/null|wc -l), ext=$(ls cache_hsp/test_external/vc${vc}_cluster-kmean_s42/*.npz 2>/dev/null|wc -l))" | tee -a "$LOGDIR/vcgrid.log"
}

echo "[$(date '+%F %T')] ===== VC-grid src1 build started (pid $$) VCS=[$VCS] =====" | tee -a "$LOGDIR/vcgrid.log"
for vc in $VCS; do
  build_vc "$vc" &
  while [ "$(jobs -rp | wc -l)" -ge $MAXPAR ]; do sleep 10; done
done
wait
echo "[$(date '+%F %T')] ===== VC-grid src1 build COMPLETE =====" | tee -a "$LOGDIR/vcgrid.log"
