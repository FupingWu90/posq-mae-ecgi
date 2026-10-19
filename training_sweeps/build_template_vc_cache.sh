#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Raw N-electrode inverse (VC-only baseline) under the FIXED TEMPLATE layout for every N of the grid.
# Model-independent, CPU only, low priority; writes only to the v2 cache (inverse_evaluation/cache_hsp).
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=4
cd "$ROOT"
RUN="${ECGI_V1_ROOT:-$ROOT/v1}/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800"
for n in 4 8 32 64; do
  for s in test_internal test_external val; do
    echo "[$(date '+%F %T')] vc=$n split=$s"
    nice -n 19 python inverse_evaluation/cache_references.py --run_dir "$RUN" --split $s --vc $n --mode template 2>&1 | tail -1
  done
done
echo "TEMPLATE-CACHE-FINISHED $(date '+%F %T')"
