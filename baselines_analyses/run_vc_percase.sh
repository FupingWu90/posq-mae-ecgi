#!/bin/bash
# Per-recording downstream CSVs for the wide model at every N (kmean deploy).
# Reuses cached refs + per-N src1; computes src3/src4 fresh. 4-way parallel.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
cd $ROOT/inverse_evaluation
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
OUT=$ROOT/baselines_analyses/generated/vc_percase
RUNS=$ROOT/runs/data-cohortA+cohortB_vs_external
run_one () {
  local vc=$1 split=$2
  local rd=$RUNS/fixedn_ecgi_wide_vc${vc}_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800
  [ -s "$OUT/vc${vc}_${split}.csv" ] && { echo "skip vc$vc $split"; return; }
  python eval_downstream.py --run_dir "$rd" --n 1000000 --vc $vc \
    --mode cluster --cluster_method kmean --split $split \
    --out "$OUT/vc${vc}_${split}.csv" > "$OUT/vc${vc}_${split}.log" 2>&1
  echo "done vc$vc $split"
}
for split in test_internal test_external; do
  for vc in 4 8 16 24 32 48 64; do
    run_one $vc $split &
    while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 15; done
  done
done
wait
echo "VC PERCASE DONE $(date)"
