#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# 2026-11-11 pre-submission audit fix: re-run every CARNZ evaluation after two inverse_evaluation/build_sources.py fixes
#   (1) CARNZ deploy normalisation g = training convention (torch std ddof=1 + lower median; was np.std + np.median),
#   (2) the full-vest source is no longer built from the array whose visible rows were re-referenced (--reref runs).
# Same jobs as training_sweeps/run_review2_gpu.sh (CARNZ part); CPU only (the GPU is in use by another project).
# Old outputs: baselines_analyses/generated/_backup_pre_auditfix_20261111/. Logs contain sample ids: never print them.
set -u
cd "$ROOT"
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=3
CAND=baselines_analyses/generated/percase_candidate; RER=baselines_analyses/generated/percase_rerun; LOG=training_sweeps/logs/review2_auditfix; mkdir -p $LOG
RC=$(ls -d runs/data-*/CARNZ__* | head -n 1)
MAXJ=6
waitslot() { while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 20; done; }
ev() {  # id run vc split mode
  local id=$1 run=$2 vc=$3 split=$4 mode=$5 lay SEL out
  if [ "$mode" = template ]; then lay=template; SEL="--mode template"; else lay=kmeans; SEL="--mode cluster --cluster_method kmean"; fi
  out=$CAND/${id}_vc${vc}_${split}_${lay}.csv; [ -f "$out" ] && { echo "[skip] ${id} ${split} ${lay}"; return; }
  waitslot
  ( nice -n 5 python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc $vc $SEL --split $split --out "$out" > "$LOG/${id}_vc${vc}_${split}_${lay}.log" 2>&1; echo "[done] eval ${id} ${split} ${lay} (exit $?) $(date '+%F %T')" ) &
}
rr() {  # tag run vc split extra...
  local tag=$1 run=$2 vc=$3 split=$4; shift 4
  [ -f "$RER/${tag}_vc${vc}_${split}_template.csv" ] && { echo "[skip] ${tag} ${split}"; return; }
  waitslot
  ( nice -n 5 python baselines_analyses/sources_rerun_v2.py --run_dir "$run" --split $split --vc $vc --threads 3 --tag $tag "$@" > $LOG/${tag}_vc${vc}_${split}.log 2>&1; echo "[done] ${tag} ${split} (exit $?) $(date '+%F %T')" ) &
}
echo "[carnz-auditfix] start $(date '+%F %T')"
for sp in test_external test_internal; do
  ev CARNZ "$RC" 16 $sp template; ev CARNZ "$RC" 16 $sp kmeans
  rr carnz_base "$RC" 16 $sp; rr carnz_reref_car "$RC" 16 $sp --reref car; rr carnz_reref_ch0 "$RC" 16 $sp --reref ch0
  rr carnz_noise_emg10 "$RC" 16 $sp --noise emg 10; rr carnz_noise_white10 "$RC" 16 $sp --noise white 10; rr carnz_noise_white5 "$RC" 16 $sp --noise white 5
done
wait
echo "[carnz-auditfix] ALL-DONE $(date '+%F %T')"
