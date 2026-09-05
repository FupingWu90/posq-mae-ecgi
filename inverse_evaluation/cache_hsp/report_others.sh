#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repository root
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
cd $ROOT/inverse_evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in cluster-kmean patch random; do
  d="$ROOT/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_base_vc16_${m}_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800"
  echo "===== $m $(date) ====="
  python report_at_rt.py --run_dir "$d" --splits test_internal test_external
done
echo "===== ALL REPORTS DONE $(date) ====="
