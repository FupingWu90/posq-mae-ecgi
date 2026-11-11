#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Reviewer sensitivity reruns, batch 3 (2026-10-23 afternoon): noise injection (point 2b) and denoising variants (point 2d)
# on the stored evaluation of the fixed template layout at N = 16, both cohorts.  baselines_analyses/sources_rerun_v2.py; outputs
# generated/percase_rerun/<tag>_vc16_<split>_template.csv ; logs training_sweeps/logs/review_rerun_<tag>_<split>_vc16.out (aggregates only).
#   setsid nohup bash training_sweeps/run_review_reruns3.sh > training_sweeps/logs/review_reruns3_driver.out 2>&1 < /dev/null &
set -u
cd "$ROOT"
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
RUN=$(ls -d runs/data-*/LOC3__posq_ecgi_wide_hd-subattn_s5c32l1_vc16_cluster-greedy_val-template_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800)
OUT=baselines_analyses/generated/percase_rerun
MAXJ=6
job() {  # tag split opts...
  local tag=$1 split=$2; shift 2
  if [ -f "$OUT/${tag}_vc16_${split}_template.csv" ]; then echo "[skip] $tag $split"; return; fi
  nice -n 5 python baselines_analyses/sources_rerun_v2.py --run_dir "$RUN" --split $split --vc 16 --tag $tag --threads 3 "$@" > training_sweeps/logs/review_rerun_${tag}_${split}_vc16.out 2>&1
  echo "[done] $tag $split (exit $?) $(date)"
}
SRC3="--sources src1_vc src2_allvalid src4_recon_valid"
for split in test_internal test_external; do
  job noise_white20 $split --noise white 20 $SRC3 &
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
  job noise_white10 $split --noise white 10 $SRC3 &
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
  job noise_white5 $split --noise white 5 $SRC3 &
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
  job noise_emg10 $split --noise emg 10 $SRC3 &
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
  job noise_bw0 $split --noise bw 0 $SRC3 &
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
  job svd8 $split --svd_k 8 --sources src2_allvalid src5_svd &
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
  job svd16 $split --svd_k 16 --sources src2_allvalid src5_svd &
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
  job svd32 $split --svd_k 32 --sources src2_allvalid src5_svd &
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
  job svd16in $split --svd_k 16 --svd_input --sources src1_vc src2_allvalid src4_recon_valid src5_svd &
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
done
wait
echo "[review-reruns3] ALL-DONE $(date)"
