#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repository root
# One short epoch of every Tier-1 variant through train.py (small batch, alongside the running queue).
source "${VENV:-$ROOT/venv}/bin/activate" 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
cd "$ROOT/bsp_reconstruction"
COMMON="--model posq_ecgi_wide --batch_size 8 --accum_iter 2 --patch_size 1 50 --vc 16 --norm_mode global --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 --reg_scope recon --select_mode cluster --cluster_method greedy --val_select_mode template --aug_times 1 --epochs 1 --use_amp --num_workers 4 --save_every 1 --eval_every 1 --test_limit 2 --no_downstream_eval"
run() { id=$1; shift; python train.py $COMMON --tag smoke-$id "$@" > ../training_sweeps/logs/smoke_$id.out 2>&1; echo "$id exit $? | $(grep -c 'Done. Results' ../training_sweeps/logs/smoke_$id.out) done | $(grep 'VAL ' ../training_sweeps/logs/smoke_$id.out | tail -1 | cut -c23-80) | $(grep -i 'error\|nan\|Traceback' ../training_sweeps/logs/smoke_$id.out | head -1 | cut -c1-120)"; }
run DEC2 --dec_tsa
run LOC3 --head subattn --inner_depth 1
run LOC3t --head subattn --inner_depth 1 --dec_tsa
run LOC2 --tokenizer tnt
run LOC1 --tokenizer conv
run MSK1 --tmask_mode dropout --tmask_ratio 0.25
run MSK3 --tmask_mode alternate
run MSK3t --tmask_mode alternate --dec_tsa
run EMB1 --time_embed learned
echo ALL-SMOKE-FINISHED
