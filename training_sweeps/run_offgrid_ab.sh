#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Off-grid A/B evaluation (reviewer point 9, 2026-10-23): waits for the half-A training run (OFFGRIDA, --select_region subA)
# to finish, then evaluates the control model (LOC3, trained on all positions) and the half-A model on half B
# (positions never seen as inputs by the half-A model) and on half A, fixed template and k-means layouts, N = 16,
# both cohorts -> generated/percase_offgrid/<ID>_vc16_<split>_<cond>.csv ; then baselines_analyses/offgrid_ab.py --summary.
#   setsid nohup bash training_sweeps/run_offgrid_ab.sh > training_sweeps/logs/offgrid_ab_driver.out 2>&1 < /dev/null &
set -u
cd "$ROOT"
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
LOC3=$(ls -d runs/data-*/LOC3__posq_ecgi_wide_hd-subattn_s5c32l1_vc16_cluster-greedy_val-template_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800)
OUT=baselines_analyses/generated/percase_offgrid; mkdir -p $OUT
echo "[offgrid] waiting for the half-A run to finish $(date)"
while true; do
  RUNA=$(ls -d runs/data-*/OFFGRIDA__*_reg-subA 2>/dev/null | head -n 1)
  if [ -n "$RUNA" ] && [ -f "$RUNA/test_compare.json" ]; then break; fi
  sleep 300
done
echo "[offgrid] half-A run finished $(date)"
MAXJ=6
run_one() {  # id run split cond
  local id=$1 run=$2 split=$3 cond=$4 SEL
  case $cond in
    tplB) SEL="--mode template --region subB";;
    kmB)  SEL="--mode cluster --cluster_method kmean --region subB";;
    tplA) SEL="--mode template --region subA";;
    kmA)  SEL="--mode cluster --cluster_method kmean --region subA";;
  esac
  local out=$OUT/${id}_vc16_${split}_${cond}.csv
  if [ -f "$out" ]; then echo "[skip] $id $split $cond"; return; fi
  nice -n 5 python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc 16 $SEL --split $split --out "$out" > "$OUT/${id}_vc16_${split}_${cond}.log" 2>&1
  echo "[done] $id $split $cond (exit $?) $(date)"
}
for cond in tplB kmB tplA kmA; do
  for split in test_internal test_external; do
    for id in LOC3 OFFGRIDA; do
      run=$LOC3; [ $id = OFFGRIDA ] && run=$RUNA
      run_one $id "$run" $split $cond &
      while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 30; done
    done
  done
done
wait
python baselines_analyses/offgrid_ab.py --summary
echo "[offgrid] ALL-DONE $(date)"
