#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repository root
cd $ROOT/inverse_evaluation
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
RUN=$ROOT/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800
for split in test_internal test_external; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python cache_references.py --run_dir "$RUN" \
    --vc 72 --mode cluster --cluster_method kmean --split $split \
    >> cache_hsp/logs/vc72_kmean.out 2>&1 &
done
wait
echo "VC72 CACHE DONE $(date)"
