#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Reviewer sensitivity reruns (2026-10-23): see baselines_analyses/sources_rerun_v2.py. Idempotent: a job whose CSV exists is skipped.
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
cd "$ROOT"
RUN="$ROOT/runs/data-cohortA+cohortB_vs_external/LOC3__posq_ecgi_wide_hd-subattn_s5c32l1_vc16_cluster-greedy_val-template_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
job() {  # tag split vc extra-args...
  local tag=$1 split=$2 vc=$3; shift 3
  local out=baselines_analyses/generated/percase_rerun/${tag}_vc${vc}_${split}_template.csv
  if [ -s "$out" ]; then echo "[skip] $tag $split vc$vc"; return 0; fi
  python baselines_analyses/sources_rerun_v2.py --run_dir "$RUN" --split $split --vc $vc --tag $tag --threads 4 "$@" > training_sweeps/logs/review_rerun_${tag}_${split}_vc${vc}.out 2>&1
  echo "[done] $tag $split vc$vc (exit $?)"
}
export -f job; export RUN
{
for split in test_internal test_external; do
  echo "base $split 16"
  echo "neumann $split 16 --neumann_all"
  echo "reref_car $split 16 --reref car --sources src1_vc src4_recon_valid"
  echo "reref_ch0 $split 16 --reref ch0 --sources src1_vc src4_recon_valid"
  echo "nosub $split 16 --nosub --sources src1_vc src4_recon_valid"
  echo "hp05 $split 16 --hp 0.5"
  echo "qrs120 $split 16 --qrs -50 120"
  echo "rpeaks_bsp $split 16 --rpeaks bsp"
  for vc in 4 8 32 64; do
    echo "base $split $vc --sources src1_vc"
    echo "neumann $split $vc --neumann_all --sources src1_vc"
  done
done
} | xargs -P 6 -L 1 bash -c 'job "$@"' _
echo "[review-reruns] ALL-DONE $(date)"
