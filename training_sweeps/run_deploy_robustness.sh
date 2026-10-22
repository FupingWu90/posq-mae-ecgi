#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Deployment-layout robustness (S8): 4 parallel shards of baselines_analyses/deploy_robustness.py (k-means seeds x3, templates x5,
# all methods, 80 recordings per cohort) -> aggregate; then the half-torso deployment of the final model on the full
# test cohorts (k-means restricted to a half; half-torso templates) -> generated/percase_region/.  Idempotent.
#   setsid nohup bash training_sweeps/run_deploy_robustness.sh > training_sweeps/logs/deploy_robustness.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
echo "[deploy] started $(date '+%F %T')"
for k in 0 1 2 3; do
  nice -n 10 python baselines_analyses/deploy_robustness.py --shard $k --nshards 4 > training_sweeps/logs/deploy_robustness_shard$k.out 2>&1 &
done
wait
echo "[deploy] shards done $(date '+%F %T')"
python baselines_analyses/deploy_robustness.py --agg_only > training_sweeps/logs/deploy_robustness_agg.out 2>&1
echo "[deploy] aggregated $(date '+%F %T')"
OUT=baselines_analyses/generated/percase_region; mkdir -p $OUT
run=$(ls -d runs/data-cohortA+cohortB_vs_external/LOC3__* | head -1)
TPL=baselines_analyses/generated/deploy_robustness/templates
job() { split=$1; cond=$2
  out="$OUT/LOC3_vc16_${split}_${cond}.csv"
  want=$([ "$split" = test_internal ] && echo 1650 || echo 1500)
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$want" ]; then echo "[skip] $out"; return; fi
  case "$cond" in
    km_front) SEL="--mode cluster --cluster_method kmean --region front";;
    km_back)  SEL="--mode cluster --cluster_method kmean --region back";;
    tpl_front) SEL="--mode template --template_npz $TPL/tpl_front.npz";;
    tpl_back)  SEL="--mode template --template_npz $TPL/tpl_back.npz";;
  esac
  nice -n 10 python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc 16 $SEL --split $split --out "$out" > "$OUT/LOC3_vc16_${split}_${cond}.log" 2>&1
  echo "[$(date '+%F %T')] done $split $cond ($(grep -c FAILED "$OUT/LOC3_vc16_${split}_${cond}.log") failed, $(wc -l < "$out") rows)"
}
export -f job; export OUT TPL run
{ for split in test_internal test_external; do for cond in km_front km_back tpl_front tpl_back; do echo "$split $cond"; done; done; } \
  | xargs -P 4 -L 1 bash -c 'job "$0" "$1"'
echo "DEPLOY-ROBUSTNESS-FINISHED $(date '+%F %T')"
