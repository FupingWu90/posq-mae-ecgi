#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# Reviewer-2 (user) comments 2026-10-25, CPU batch: (J1) channel-independent noise + cross-completion on the clinical cohorts,
# (J2) shared time base from the 16 visible electrodes, (J3) VC-only lambda calibration per N on the validation split,
# (J4) VC-only inverse with the calibrated lambda on both test cohorts.  Outputs -> generated/percase_rerun/<tag>_vc<N>_<split>_template.csv
#   setsid nohup bash training_sweeps/run_review2_cpu.sh > training_sweeps/logs/review2_cpu.out 2>&1 < /dev/null &
set -u
cd "$ROOT"
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
RUN=$(ls -d runs/data-cohortA+cohortB_vs_external/LOC3__* | head -n 1)
OUT=baselines_analyses/generated/percase_rerun; LOG=training_sweeps/logs/review2; mkdir -p $LOG
MAXJ=6
waitslot() { while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 20; done; }
run() {  # tag vc split extra...
  local tag=$1 vc=$2 split=$3; shift 3
  [ -f "$OUT/${tag}_vc${vc}_${split}_template.csv" ] && { echo "[skip] $tag vc$vc $split"; return; }
  waitslot
  ( nice -n 5 python baselines_analyses/sources_rerun_v2.py --run_dir "$RUN" --split $split --vc $vc --threads 4 --tag $tag "$@" > $LOG/${tag}_vc${vc}_${split}.log 2>&1; echo "[done] $tag vc$vc $split (exit $?) $(date)" ) &
}
echo "[review2] start $(date)"
# J3: lambda grid on the validation split (VC-only only)
for vc in 4 8 16 32 64; do run lamgrid $vc val --lam_grid_vc --sources src1_vc; done
# J1: noise + cross-completion
for snr in 20 10 5; do for sp in test_internal test_external; do run noisecross_white$snr 16 $sp --noise white $snr --model_denoise cross --sources src1_vc src2_allvalid src4_recon_valid src6_mdn; done; done
# J2: shared time base from the visible electrodes
for sp in test_internal test_external; do run rpeaks_vis 16 $sp --rpeaks vis; done
wait
echo "[review2] J1-J3 done $(date)"
# calibration: lambda* per N = grid weight with the lowest mean AT MAE vs the commercial reference over the validation recordings
python - <<'PY' > $LOG/lambda_calibration.txt 2>&1
import pandas as pd, numpy as np, json, os
out = {}
for vc in (4, 8, 16, 32, 64):
    d = pd.read_csv(f'baselines_analyses/generated/percase_rerun/lamgrid_vc{vc}_val_template.csv')
    d = d[(d.source == 'src1_vc') & (d.ref == 'commercial') & (d.success == 1)]
    d['lam_r'] = d.lam.round(6); g = d.groupby('lam_r').agg(AT=('AT_mae', 'mean'), RT=('RT_mae', 'mean'), n=('sid', 'nunique'))
    best = g.AT.idxmin(); out[vc] = dict(lam=float(best), AT_val=float(g.AT[best]), RT_val=float(g.RT[best]), n=int(g.n[best]),
                                        grid={str(k): [float(v.AT), float(v.RT)] for k, v in g.iterrows()})
    print(f'N={vc}: lambda*={best:g}  val AT {g.AT[best]:.2f} RT {g.RT[best]:.2f} (n={g.n[best]}); grid AT: ' + ' '.join(f'{k:g}:{v:.1f}' for k, v in g.AT.items()))
json.dump(out, open('baselines_analyses/generated/lambda_calibration_v2.json', 'w'), indent=1)
PY
cat $LOG/lambda_calibration.txt
# J4: VC-only inverse with the calibrated lambda on both test cohorts
for vc in 4 8 16 32 64; do
  lam=$(python -c "import json; print(json.load(open('baselines_analyses/generated/lambda_calibration_v2.json'))['$vc']['lam'])")
  for sp in test_internal test_external; do run lamcal $vc $sp --fixed_lam_vc $lam --sources src1_vc; done
done
wait
echo "[review2] ALL-DONE $(date)"
