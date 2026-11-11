#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Early start of the batch-2 jobs that do not need the extended-column re-run (oracle lambda, all-visible), 2026-10-23.
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
cd "$ROOT"
RUN="$ROOT/runs/data-cohortA+cohortB_vs_external/LOC3__posq_ecgi_wide_hd-subattn_s5c32l1_vc16_cluster-greedy_val-template_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
job() {
  local tag=$1 split=$2 vc=$3; shift 3
  local out=baselines_analyses/generated/percase_rerun/${tag}_vc${vc}_${split}_template.csv
  if [ -s "$out" ]; then echo "[skip] $tag $split vc$vc"; return 0; fi
  python baselines_analyses/sources_rerun_v2.py --run_dir "$RUN" --split $split --vc $vc --tag $tag --threads 2 "$@" > training_sweeps/logs/review_rerun_${tag}_${split}_vc${vc}.out 2>&1
  echo "[done] $tag $split vc$vc (exit $?)"
}
export -f job; export RUN
{
for split in test_internal test_external; do
  echo "oracle $split 16 --oracle_lam --sources src1_vc"
  echo "visall $split 16 --visible_all --sources src2_allvalid src3_recon_whole"
done
} | xargs -P 4 -L 1 bash -c 'job "$@"' _
echo "[review-reruns-extra] ALL-DONE $(date)"
