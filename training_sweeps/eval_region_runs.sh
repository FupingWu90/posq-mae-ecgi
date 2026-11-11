#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Half-torso deployment evaluation of finished runs (both protocols): k-means restricted to a half (--region) and the
# half-torso templates (baselines_analyses/deploy_robustness.py --build_templates).  Output generated/percase_region/<ID>_vc16_<split>_<cond>.csv
#   setsid nohup bash training_sweeps/eval_region_runs.sh P3REGF P3REGB > training_sweeps/logs/eval_region_runs.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=6
cd "$ROOT"
OUT=baselines_analyses/generated/percase_region; mkdir -p $OUT
TPL=baselines_analyses/generated/deploy_robustness/templates
job() { id=$1; split=$2; cond=$3
  run=$(ls -d runs/data-cohortA+cohortB_vs_external/${id}__* | head -1)
  out="$OUT/${id}_vc16_${split}_${cond}.csv"
  want=$([ "$split" = test_internal ] && echo 1650 || echo 1500)
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$want" ]; then echo "[skip] $out"; return; fi
  case "$cond" in
    km_front) SEL="--mode cluster --cluster_method kmean --region front";;
    km_back)  SEL="--mode cluster --cluster_method kmean --region back";;
    tpl_front) SEL="--mode template --template_npz $TPL/tpl_front.npz";;
    tpl_back)  SEL="--mode template --template_npz $TPL/tpl_back.npz";;
  esac
  nice -n 10 python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc 16 $SEL --split $split --out "$out" > "$OUT/${id}_vc16_${split}_${cond}.log" 2>&1
  echo "[$(date '+%F %T')] done $id $split $cond ($(grep -c FAILED "$OUT/${id}_vc16_${split}_${cond}.log") failed, $(wc -l < "$out") rows)"
}
export -f job; export OUT TPL
{ for id in "$@"; do for split in test_internal test_external; do for cond in km_front km_back tpl_front tpl_back; do echo "$id $split $cond"; done; done; done; } \
  | xargs -P 4 -L 1 bash -c 'job "$0" "$1" "$2"'
echo "EVAL-REGION-RUNS-FINISHED $(date '+%F %T')"
