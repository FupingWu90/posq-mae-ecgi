#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repository root
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
cd $ROOT/inverse_evaluation
for split in test_internal test_external; do
  echo "===== kmean vc-cache $split $(date) ====="
  python cache_references.py --run_dir "$ROOT/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800" --split $split --mode cluster --cluster_method kmean
done
echo "===== KMEAN VC-CACHE DONE $(date) ====="
