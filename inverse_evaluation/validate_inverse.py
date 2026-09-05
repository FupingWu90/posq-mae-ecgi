"""
Validate the MFS inverse itself, BEFORE trusting any AT/RT comparison.

Sanity check: our MFS inverse from ALL real valid channels (source 2) should
agree with the commercial .potential HSP (the clinical gold standard). If it
does (high per-vertex temporal CC, scale ratio ~O(1), sensible L-curve lambda,
physiological QRS/T electrogram morphology), the reconstructed HSP is valid and
the downstream AT/RT comparison is meaningful.

  python validate_inverse.py --run_dir <run> --split test_internal --n 12 [--figs 3]

Writes validate_out/<split>/summary.csv and electrogram-overlay PNGs.
"""
import os, sys, csv, json, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction')
sys.path.insert(0, _CODE)

from build_sources import build_context, CROP
from inverse import assemble_solver, solve_from_field, load_commercial
from dataset import collect_split

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validate_out')


def per_vertex_temporal_cc(a, b):
    am = a - a.mean(1, keepdims=True); bm = b - b.mean(1, keepdims=True)
    den = np.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1))
    return np.divide((am * bm).sum(1), den, out=np.full(a.shape[0], np.nan), where=den > 1e-12)


def per_timestep_spatial_cc(a, b):
    am = a - a.mean(0, keepdims=True); bm = b - b.mean(0, keepdims=True)
    den = np.sqrt((am ** 2).sum(0) * (bm ** 2).sum(0))
    return np.divide((am * bm).sum(0), den, out=np.full(a.shape[1], np.nan), where=den > 1e-12)


def _fig(phi2, phicom, cc_v, sid, out_png, k=6):
    """Overlay our vs commercial electrograms at k vertices spanning CC quality."""
    order = np.argsort(np.nan_to_num(cc_v, nan=-1))[::-1]
    valid = order[np.isfinite(cc_v[order])]
    if len(valid) < k:
        return
    picks = valid[np.linspace(0, len(valid) - 1, k).astype(int)]   # best..worst
    t = np.arange(phi2.shape[1]) / 1000.0
    fig, axes = plt.subplots(k, 1, figsize=(11, 1.7 * k), sharex=True, constrained_layout=True)
    for ax, v in zip(axes, picks):
        a = (phi2[v] - phi2[v].mean()) / (phi2[v].std() + 1e-9)
        b = (phicom[v] - phicom[v].mean()) / (phicom[v].std() + 1e-9)
        ax.plot(t, b, color='steelblue', lw=0.9, label='commercial')
        ax.plot(t, a, color='tomato', lw=0.9, ls='--', label='ours (MFS, all-valid)')
        ax.set_ylabel(f"v{v}\nCC={cc_v[v]:.2f}", fontsize=8); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('time (s)', fontsize=9)
    fig.suptitle(f"{sid}  — our MFS inverse vs commercial HSP (normalized)", fontsize=10)
    fig.savefig(out_png, dpi=110); plt.close(fig)


def validate(run_dir, split='test_internal', n=12, figs=3, seed=42):
    args = json.load(open(os.path.join(run_dir, 'args.json')))
    keep_idx = np.load(os.path.join(run_dir, 'keep_idx.npy')).astype(np.int64)
    buckets, _ = collect_split(seed=args.get('seed', 42))
    recs = buckets[split][:n]
    odir = os.path.join(OUT, split); os.makedirs(odir, exist_ok=True)

    solver_cache = {}; rows = []; nfig = 0
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        if torsopath is None:
            continue
        try:
            gd = os.path.dirname(torsopath)
            if gd not in solver_cache:
                solver_cache[gd] = assemble_solver(gd)
            solver, torso_ch_idx, heart_V, heart_F = solver_cache[gd]
            n_h = heart_V.shape[0]
            ctx = build_context(ecgpath, torsopath, keep_idx, args, seed=seed)
            if ctx is None:
                continue
            tw = (ctx['crop_start'], ctx['crop_start'] + CROP)
            phi2, lam, _, _, na = solve_from_field(solver, torso_ch_idx, *ctx['src2_allvalid'])
            phicom = load_commercial(os.path.splitext(ecgpath)[0] + '.potential', n_h, time_window=tw)
            if phi2 is None or phicom is None:
                continue
            cc_v = per_vertex_temporal_cc(phi2, phicom)
            cc_s = per_timestep_spatial_cc(phi2, phicom)
            sr = float(np.linalg.norm(phi2) / (np.linalg.norm(phicom) + 1e-12))
            row = dict(sid=sid, n_active=ctx['n_active'], n_h=n_h, lam=lam,
                       cc_temporal_median=float(np.nanmedian(cc_v)),
                       cc_temporal_mean=float(np.nanmean(cc_v)),
                       cc_temporal_q10=float(np.nanpercentile(cc_v, 10)),
                       cc_spatial_median=float(np.nanmedian(cc_s)),
                       scale_ratio=sr,
                       our_amp=float(np.abs(phi2).mean()), com_amp=float(np.abs(phicom).mean()))
            rows.append(row)
            print(f"  [{i+1}/{len(recs)}] {sid[:40]:40s} n_act={ctx['n_active']:3d} "
                  f"CCt_med={row['cc_temporal_median']:.3f} CCs_med={row['cc_spatial_median']:.3f} "
                  f"scale={sr:.2f} lam={lam:.1e}", flush=True)
            if nfig < figs:
                _fig(phi2, phicom, cc_v, sid.replace('/', '_'),
                     os.path.join(odir, f"eg_{sid.replace('/', '_')[:40]}.png"))
                nfig += 1
        except Exception as e:
            print(f"  [{i+1}] {sid[:40]} FAILED: {type(e).__name__}: {e}")

    if rows:
        with open(os.path.join(odir, 'summary.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        arr = lambda k: np.array([r[k] for r in rows])
        print("\n=== MFS INVERSE VALIDITY (source 2 all-valid vs commercial .potential) ===")
        print(f"  recordings                : {len(rows)}")
        print(f"  temporal CC  median       : {np.median(arr('cc_temporal_median')):.3f} "
              f"(mean {np.mean(arr('cc_temporal_mean')):.3f}, q10 {np.median(arr('cc_temporal_q10')):.3f})")
        print(f"  spatial  CC  median       : {np.median(arr('cc_spatial_median')):.3f}")
        print(f"  scale ratio (ours/com)    : median {np.median(arr('scale_ratio')):.2f} "
              f"[{arr('scale_ratio').min():.2f}, {arr('scale_ratio').max():.2f}]")
        print(f"  L-curve lambda            : median {np.median(arr('lam')):.1e} "
              f"[{arr('lam').min():.1e}, {arr('lam').max():.1e}]")
        print(f"  figures + summary.csv -> {odir}")
        good = np.median(arr('cc_temporal_median'))
        print(f"\n  VERDICT: MFS HSP is {'VALID (agrees with commercial)' if good >= 0.7 else 'QUESTIONABLE — investigate (low agreement)'} "
              f"[median temporal CC {good:.3f}]")
    return rows


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--split', default='test_internal')
    p.add_argument('--n', type=int, default=12)
    p.add_argument('--figs', type=int, default=3)
    a = p.parse_args()
    validate(a.run_dir, split=a.split, n=a.n, figs=a.figs)
