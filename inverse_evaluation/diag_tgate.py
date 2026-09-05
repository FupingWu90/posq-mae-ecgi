"""Verify the T-wave gate actually restricts to vertices with a real T-wave.

For one recording, plot the commercial signal-averaged beat for:
  (top block)    vertices the gate marks T-GOOD  -> should show a clear T-wave in
                 the T-window, with RT on its upslope
  (bottom block) vertices the gate EXCLUDES (qrs-good but not t-good) -> should be
                 flat / no real T-wave in the window
Also prints, over T-good vertices, where RT lands relative to the QRS end (is it a
true T-wave, or the S-wave recovery right after a wide QRS?).
"""
import os, sys, json, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction'))
from build_sources import build_context, CROP
from inverse import assemble_solver, solve_from_field, load_commercial
from at_rt import extract_sample, good_vertex_mask
from dataset import collect_split
_PROJ = os.environ.get('ECGI_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

RUN = _PROJ + '/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800'
IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 1     # which test_internal recording
args = json.load(open(RUN + '/args.json')); keep = np.load(RUN + '/keep_idx.npy').astype('int64')
buckets, _ = collect_split(seed=42); sid, ecg, tor = [r for r in buckets['test_internal'] if r[2]][IDX]
solver, tci, hV, hF = assemble_solver(os.path.dirname(tor)); nh = hV.shape[0]
ctx = build_context(ecg, tor, keep, args); tw = (ctx['crop_start'], ctx['crop_start'] + CROP)
com = load_commercial(os.path.splitext(ecg)[0] + '.potential', nh, time_window=tw)
r = extract_sample(com); qs, qe, ts, te = r['windows']; sig = r['sigavg']; L = sig.shape[1]
gm = good_vertex_mask(sig, *r['windows']); qgood, tgood = gm['qrs'], gm['t']
excl = qgood & ~tgood
print(f"{sid.split('__')[-1][:20]}  HR-window T[{ts},{te}] (R={r['rel_R']}, QRS_end={qe})")
print(f"  qrs_good={qgood.sum()}  t_good={tgood.sum()}  excluded(qrs&~t)={excl.sum()}")
# where does RT land, relative to QRS end, on T-good vertices?
rt_after_qrs = r['RT'][tgood] - qe
print(f"  RT-minus-QRSend on t_good vertices (ms): median={np.median(rt_after_qrs):.0f} "
      f"[{np.percentile(rt_after_qrs,10):.0f}, {np.percentile(rt_after_qrs,90):.0f}]  "
      f"(small -> landing on S-wave recovery, not a late T-wave)")
# T amplitude of t_good vs excluded
tamp = (sig[:, ts:te].max(1) - sig[:, ts:te].min(1))
qamp = (sig[:, qs:qe].max(1) - sig[:, qs:qe].min(1))
print(f"  T/QRS amp ratio: t_good median={np.median((tamp/qamp)[tgood]):.2f}  "
      f"excluded median={np.median((tamp/qamp)[excl]):.2f}")

def block(ax_list, idxs, tag):
    t = np.arange(L)
    for ax, v in zip(ax_list, idxs):
        s = (sig[v] - sig[v].mean()) / (sig[v].std() + 1e-9)
        ax.plot(t, s, 'steelblue', lw=1.0)
        ax.axvspan(ts, te, color='green', alpha=0.08); ax.axvspan(qs, qe, color='red', alpha=0.08)
        ax.axvline(r['RT'][v], color='green', lw=1.3); ax.axvline(r['AT'][v], color='red', lw=1.0)
        ax.set_ylabel(f"{tag} v{v}", fontsize=8); ax.tick_params(labelsize=7)

rng = np.random.default_rng(0)
tg = np.where(tgood)[0]; ex = np.where(excl)[0]
pick_t = rng.choice(tg, min(5, len(tg)), replace=False) if len(tg) else []
pick_e = rng.choice(ex, min(5, len(ex)), replace=False) if len(ex) else []
fig, axes = plt.subplots(10, 1, figsize=(11, 15), sharex=True, constrained_layout=True)
block(axes[:5], pick_t, 'T-GOOD')
block(axes[5:], pick_e, 'EXCL')
axes[0].set_title(f"{sid.split('__')[-1][:20]}: T-GOOD (top 5) vs EXCLUDED qrs-good-only (bottom 5). "
                  f"green=RT, red=AT; shaded=T/QRS windows", fontsize=9)
axes[-1].set_xlabel('sample (ms) in signal-averaged beat', fontsize=9)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validate_out')
fig.savefig(os.path.join(OUT, f"tgate_{sid.split('__')[-1][:16]}.png"), dpi=110); plt.close(fig)
print(f"  figure -> {OUT}/tgate_{sid.split('__')[-1][:16]}.png")
