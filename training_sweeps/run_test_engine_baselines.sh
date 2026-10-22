#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Engine held-out BSP MAE / CC of the frozen v1 baseline runs (ST-MEM = pe-index, MCMA) on the test cohorts under
# both layouts (for the S13 amplitude tables under the template layout).  Idempotent.
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
cd "$ROOT"
STD=cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800
OUT=baselines_analyses/generated/engine_test_baselines; mkdir -p $OUT
for vc in 4 8 16 32 64; do
  for pair in "stmem:fixedn_ecgi_wide_vc${vc}_${STD}_pe-index" "mcma:mcma_unet_vc${vc}_${STD}"; do
    key=${pair%%:*}; run=${pair#*:}
    [ -f "$OUT/${key}_vc${vc}.json" ] && { echo "[skip] $key vc$vc"; continue; }
    echo "[$(date '+%F %T')] $key vc$vc"
    nice -n 10 python baselines_analyses/test_engine_metrics.py --run_dir ${ECGI_V1_ROOT:-$ROOT/v1}/runs/data-cohortA+cohortB_vs_external/$run --out $OUT/${key}_vc${vc}.json 2>&1 | grep -v Warning
  done
done
echo "TEST-ENGINE-BASELINES-FINISHED $(date '+%F %T')"
