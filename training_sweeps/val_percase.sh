#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Per-recording downstream metrics on the VALIDATION recordings for paired comparisons between pilots,
# under ONE deployment layout (both layouts are evaluated for every pilot; user decision 2026-10-06).
#   usage:  [LAYOUT=template|kmeans] bash training_sweeps/val_percase.sh ID [ID ...]     (ID = pilot id, or 'v1')
#   ->  results/val_percase/<ID>_vc16_val_<layout>.csv    (log: results/val_percase/<ID>[_kmeans].log)
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
LAYOUT=${LAYOUT:-template}
case "$LAYOUT" in
  template) SEL="--mode template"; SUF="";;
  kmeans)   SEL="--mode cluster --cluster_method kmean"; SUF="_kmeans";;
  *) echo "unknown LAYOUT=$LAYOUT (template|kmeans)"; exit 1;;
esac
cd "$ROOT"
for id in "$@"; do
  if [ "$id" = v1 ]; then run="${ECGI_V1_ROOT:-$ROOT/v1}/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800"
  else run=$(ls -d runs/data-cohortA+cohortB_vs_external/${id}__* | head -1); fi
  out=results/val_percase/${id}_vc16_val_${LAYOUT}.csv
  [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 600 ] && { echo "[skip] $out"; continue; }
  python inverse_evaluation/eval_downstream.py --run_dir "$run" --n 1000000 --vc 16 $SEL --split val --out "$out" > results/val_percase/${id}${SUF}.log 2>&1 &
done
wait
echo VAL-PERCASE-FINISHED
