#!/bin/bash
# Full-set train x deploy downstream matrix (all cached recordings, --n large).
# The 4 (train,deploy) combos are independent -> run them in PARALLEL, each doing
# both splits sequentially. BLAS threads capped so 4 procs share 32 cores cleanly.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repository root
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
cd $ROOT/inverse_evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
N=100000
declare -A DIRS=( [greedy]="$ROOT/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800" [kmean]="$ROOT/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_base_vc16_cluster-kmean_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800" )
mkdir -p out_matrix cache_hsp/logs

run_combo () {
  local train=$1 deploy=$2
  for split in test_internal test_external; do
    echo "===== [combo $train/$deploy] START $split $(date) ====="
    python eval_downstream.py --run_dir "${DIRS[$train]}" \
      --n $N --mode cluster --cluster_method $deploy --split $split \
      --out out_matrix/train-${train}_deploy-${deploy}_${split}.csv 2>&1 \
      | grep -vE "precompute" | tail -3
    echo "===== [combo $train/$deploy] END   $split $(date) ====="
  done
  echo "##### COMBO DONE $train/$deploy $(date) #####"
}

pids=()
for train in greedy kmean; do
  for deploy in greedy kmean; do
    run_combo $train $deploy > cache_hsp/logs/combo_${train}_${deploy}.out 2>&1 &
    pids+=($!)
    echo "launched combo train=$train deploy=$deploy pid=$! -> logs/combo_${train}_${deploy}.out"
  done
done
echo "waiting on ${#pids[@]} combos: ${pids[*]}"
wait
echo "===== DEPLOY MATRIX DONE $(date) ====="
