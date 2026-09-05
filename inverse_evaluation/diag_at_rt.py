"""Diagnose the AT/RT computation (esp. RT). For a few recordings it computes
src2 (all-valid MFS) and commercial sigavg + AT/RT, then checks:
  - detected heart rate / RR (is the fixed T-window [R+150,R+400]ms appropriate?)
  - fraction of RT / AT landing AT the window edges (window too narrow -> clipping)
  - AT/RT src2-vs-commercial agreement on good vertices
  - renders sigavg for several vertices with AT (red) and RT (green) marked, for
    src2 (solid) and commercial (dashed), so RT placement on the T-wave is visible.
"""
import os, sys, json, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction'))
from build_sources import build_context, CROP
from inverse import assemble_solver, solve_from_field, load_commercial
from at_rt import extract_sample, good_vertex_mask, detect_beats
from dataset import collect_split
_PROJ = os.environ.get('ECGI_PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    _PROJ + '/runs/data-cohortA+cohortB_vs_external/fixedn_ecgi_base_vc16_cluster-greedy_nm-gl_ws0-wt0_sc-recon_k6s10_ebs64_crop5000_ep800'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validate_out', 'diag_atrt')
os.makedirs(OUT, exist_ok=True)

args = json.load(open(RUN + '/args.json')); keep = np.load(RUN + '/keep_idx.npy').astype('int64')
buckets, _ = collect_split(seed=42); recs = [r for r in buckets['test_internal'] if r[2]][:N]
sc = {}
for ri, (sid, ecg, tor) in enumerate(recs):
    gd = os.path.dirname(tor)
    if gd not in sc: sc[gd] = assemble_solver(gd)
    solver, tci, hV, hF = sc[gd]; nh = hV.shape[0]
    ctx = build_context(ecg, tor, keep, args); tw = (ctx['crop_start'], ctx['crop_start'] + CROP)
    phi2 = solve_from_field(solver, tci, *ctx['src2_allvalid'])[0]
    com = load_commercial(os.path.splitext(ecg)[0] + '.potential', nh, time_window=tw)
    # heart rate from commercial RMS beat detection
    rp = detect_beats(com); rr = np.diff(rp) if len(rp) > 1 else np.array([np.nan])
    r2 = extract_sample(phi2); rc = extract_sample(com)
    if r2 is None or rc is None:
        print(f"{sid.split('__')[-1][:16]}: extract failed"); continue
    qs, qe, ts, te = rc['windows']; L = rc['sigavg'].shape[1]
    gm = good_vertex_mask(rc['sigavg'], *rc['windows'])
    qgood, tgood = gm['qrs'], gm['t']
    rt_edge = float(((rc['RT'][tgood] <= ts + 1).mean() + (rc['RT'][tgood] >= te - 1).mean())) \
        if tgood.sum() else float('nan')
    at_mae = float(np.abs(r2['AT'][qgood] - rc['AT'][qgood]).mean()) if qgood.sum() else float('nan')
    rt_mae = float(np.abs(r2['RT'][tgood] - rc['RT'][tgood]).mean()) if tgood.sum() else float('nan')
    print(f"{sid.split('__')[-1][:16]}: HR~{60000/np.nanmedian(rr):.0f}bpm RR~{np.nanmedian(rr):.0f}ms "
          f"| T[{ts},{te}] qrs_good={qgood.sum()} t_good={tgood.sum()}/{len(tgood)} "
          f"| RT edge%={rt_edge:.2f} | AT_mae={at_mae:.1f}(n={qgood.sum()}) "
          f"RT_mae={rt_mae:.1f}(n={tgood.sum()})")

    if ri < 2:  # render 6 good vertices for the first 2 recordings
        gv = np.where(tgood)[0]
        if len(gv) < 6:
            gv = np.where(qgood)[0]
        picks = gv[np.linspace(0, len(gv) - 1, 6).astype(int)]
        t = np.arange(L)
        fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
        for ax, v in zip(axes, picks):
            a = (r2['sigavg'][v] - r2['sigavg'][v].mean()); a /= (a.std() + 1e-9)
            b = (rc['sigavg'][v] - rc['sigavg'][v].mean()); b /= (b.std() + 1e-9)
            ax.plot(t, b, 'steelblue', lw=1.0, label='commercial')
            ax.plot(t, a, 'tomato', lw=1.0, ls='--', label='ours(src2)')
            for (arr, res, c) in [('AT', rc, 'red'), ('RT', rc, 'green')]:
                ax.axvline(res[arr][v], color=c, lw=1.2, alpha=0.7)
            ax.axvline(r2['AT'][v], color='red', lw=1.0, ls=':', alpha=0.6)
            ax.axvline(r2['RT'][v], color='green', lw=1.0, ls=':', alpha=0.6)
            ax.axvspan(ts, te, color='green', alpha=0.06)   # T window
            ax.axvspan(qs, qe, color='red', alpha=0.06)     # QRS window
            ax.set_ylabel(f"v{v}", fontsize=8); ax.tick_params(labelsize=7)
        axes[0].legend(fontsize=8, loc='upper right')
        axes[0].set_title(f"{sid.split('__')[-1][:20]}  AT(red)/RT(green): solid=commercial, dotted=ours; "
                          f"shaded=windows", fontsize=9)
        axes[-1].set_xlabel('sample (ms) in signal-averaged beat', fontsize=9)
        fig.savefig(os.path.join(OUT, f"atrt_{sid.split('__')[-1][:16]}.png"), dpi=110); plt.close(fig)
print(f"\nfigures -> {OUT}")
