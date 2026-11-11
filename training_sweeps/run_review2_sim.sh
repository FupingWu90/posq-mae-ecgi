#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Reviewer-2 (user) comment: discriminating experiment A -- noise ladder on the SIMULATED test cohort (ground truth): full vest,
# 16-electrode completion and cross-completion under channel-independent white noise (none, 20, 10, 5 dB).  CPU only; waits for
# the CPU batch of run_review2_cpu.sh to finish.  Outputs generated/sim_gt/noise_<tag>_template.csv
#   setsid nohup bash training_sweeps/run_review2_sim.sh > training_sweeps/logs/review2_sim.out 2>&1 < /dev/null &
set -u
cd "$ROOT"
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
while ! grep -q "ALL-DONE" training_sweeps/logs/review2_cpu.out 2>/dev/null; do sleep 300; done
echo "[review2-sim] start $(date)"
RUN=$(ls -d runs/data-cohortA+cohortB_vs_external/LOC3__* | head -n 1); OUT=baselines_analyses/generated/sim_gt
for spec in "none|" "white20|--noise white 20" "white10|--noise white 10" "white5|--noise white 5"; do
  tag=${spec%%|*}; extra=${spec#*|}
  [ -f "$OUT/noise_${tag}_template.csv" ] && { echo "[skip] $tag"; continue; }
  ( CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 5 python inverse_evaluation/eval_sim_gt.py --run_dir "$RUN" --n 1000000 --vc 16 --mode template --cross $extra --out "$OUT/noise_${tag}_template.csv" > "$OUT/noise_${tag}_template.log" 2>&1; echo "[done] sim noise $tag (exit $?) $(date)" ) &
done
wait
echo "[review2-sim] ALL-DONE $(date)"
