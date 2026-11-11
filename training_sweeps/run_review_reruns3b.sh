#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Batch 3b (2026-10-23): waits for batch 3, then the cross-completion denoiser variant (src6_mdn, both cohorts) and the
# GP/kriging baseline shards (baselines_analyses/gp_baseline_v2.py), then the GP summary.
#   setsid nohup bash training_sweeps/run_review_reruns3b.sh > training_sweeps/logs/review_reruns3b_driver.out 2>&1 < /dev/null &
set -u
cd "$ROOT"
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
while ! grep -q "ALL-DONE" training_sweeps/logs/review_reruns3_driver.out 2>/dev/null; do sleep 120; done
echo "[3b] batch 3 finished, starting $(date)"
RUN=$(ls -d runs/data-*/LOC3__posq_ecgi_wide_hd-subattn_s5c32l1_vc16_cluster-greedy_val-template_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800)
OUT=baselines_analyses/generated/percase_rerun
for split in test_internal test_external; do
  if [ ! -f "$OUT/mdn_vc16_${split}_template.csv" ]; then
    (nice -n 5 python baselines_analyses/sources_rerun_v2.py --run_dir "$RUN" --split $split --vc 16 --tag mdn --threads 3 --model_denoise --sources src2_allvalid src4_recon_valid src6_mdn > training_sweeps/logs/review_rerun_mdn_${split}_vc16.out 2>&1; echo "[done] mdn $split $(date)") &
  fi
done
for split in test_internal test_external; do
  for sh in 0 1 2; do
    (nice -n 5 python baselines_analyses/gp_baseline_v2.py --split $split --shard $sh --nshards 3 > training_sweeps/logs/gp_baseline_${split}_shard${sh}.out 2>&1; echo "[done] gp $split shard $sh $(date)") &
  done
done
wait
python baselines_analyses/gp_baseline_v2.py --summary > training_sweeps/logs/gp_baseline_summary.out 2>&1
tail -n 3 training_sweeps/logs/gp_baseline_summary.out
echo "[review-reruns3b] ALL-DONE $(date)"
