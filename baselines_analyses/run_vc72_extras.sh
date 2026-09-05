#!/bin/bash
# Extend the Number-of-electrodes analysis to N=72 (after the VC=72 run):
#  (a) per-recording eval CSVs for vc72 (src1/src3/src4 vs both refs)
#  (b) Laplacian/TPS interpolation baselines at N=72
#  (c) per-N lambda floor calibration + VC-only re-evaluation at N=72
# CPU-bound except the model inference in (a); runs next to the training job.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
cd $ROOT/baselines_analyses
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$ROOT/baselines_analyses/generated/vc_percase
RD=$ROOT/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_wide_vc72_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800
echo "[$(date '+%F %T')] vc72 extras start"
for split in test_internal test_external; do
  [ -s "$OUT/vc72_${split}.csv" ] || (cd ../inverse_evaluation && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python eval_downstream.py --run_dir "$RD" --n 1000000 --vc 72 \
      --mode cluster --cluster_method kmean --split $split --out "$OUT/vc72_${split}.csv" > "$OUT/vc72_${split}.log" 2>&1) &
  for s in 0 1; do
    [ -s "generated/interp/${split}_shard72_${s}.csv" ] || OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python interp_baseline.py --split $split --shard $s --nshards 2 --vcs 72 \
      > generated/interp/${split}_vc72_s${s}.log 2>&1 &
  done
done
( OMP_NUM_THREADS=3 python lam_recal.py --calibrate --n_recs 20 --vcs 72 > generated/lam_recal/calibrate_vc72.log 2>&1
  for split in test_internal test_external; do for s in 0 1; do
    [ -s "generated/lam_recal/${split}_shard72_${s}.csv" ] || OMP_NUM_THREADS=3 python lam_recal.py --eval --split $split --shard $s --nshards 2 --vcs 72 \
      > generated/lam_recal/${split}_vc72_s${s}.log 2>&1 &
  done; done; wait ) &
wait
echo "[$(date '+%F %T')] VC72 EXTRAS DONE"
