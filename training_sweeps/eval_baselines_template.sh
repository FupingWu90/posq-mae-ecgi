#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Per-recording downstream evaluation of the EXISTING learned models (v1 final model, ST-MEM = v1 backbone
# with index embedding, MCMA) under the FIXED TEMPLATE layout, every N of the grid, both test cohorts.
# No training.  CPU only (CUDA hidden) so that the GPU pilots are not disturbed; low priority; idempotent.
#   setsid nohup bash training_sweeps/eval_baselines_template.sh > training_sweeps/logs/eval_baselines_template.out 2>&1 < /dev/null &
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
V1R="${ECGI_V1_ROOT:-$ROOT/v1}/runs/data-cohortA+cohortB_vs_external"
STD="cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800"
OUT=baselines_analyses/generated/percase_template
job() {
  name=$1; run=$2; n=$3; split=$4
  out="$OUT/${name}_vc${n}_${split}.csv"
  want=$([ "$split" = test_internal ] && echo 1650 || echo 1500)      # 6 rows per recording (276 / 251) minus header tolerance
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$want" ]; then echo "[skip] $out"; return; fi
  nice -n 15 python inverse_evaluation/eval_downstream.py --run_dir "$V1R/$run" --n 1000000 --vc $n --mode template --split $split --out "$out" > "$OUT/${name}_vc${n}_${split}.log" 2>&1
  echo "[$(date '+%F %T')] done $name vc$n $split ($(grep -c FAILED "$OUT/${name}_vc${n}_${split}.log") failed)"
}
export -f job; export V1R STD OUT
{
for n in 16 4 8 32 64; do
  for split in test_internal test_external; do
    echo "mcma mcma_unet_vc${n}_${STD} $n $split"
    echo "ours-v1 fixedn_ecgi_wide_vc${n}_${STD} $n $split"
    echo "stmem fixedn_ecgi_wide_vc${n}_${STD}_pe-index $n $split"
  done
done
} | xargs -P 3 -L 1 bash -c 'job "$0" "$1" "$2" "$3"'
echo "BASELINES-TEMPLATE-FINISHED $(date '+%F %T')"
